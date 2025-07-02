import type { EpochDate, OrderSide, OrderStatus, OrderType } from '@trdr/shared'
import { epochDateNow, type IsoDate, isoDateNow, isoToEpoch, toIsoDate } from '@trdr/shared'
import type { ConnectionManager } from '../db/connection-manager'
import type { Order } from '../types/orders'
import { BaseRepository } from './base-repository'

/**
 * Database order dto
 */
interface OrderDto {
  id: string
  symbol: string
  side: string
  type: string
  status: string
  price?: number
  size: number
  filled_size: number
  average_fill_price?: number
  stop_price?: number
  trail_distance?: number
  agent_id?: string
  metadata?: unknown
  created_at: IsoDate
  updated_at: IsoDate
  submitted_at?: IsoDate
  filled_at?: IsoDate
  cancelled_at?: IsoDate
}

/**
 * Repository for order management
 */
export class OrderRepository extends BaseRepository<OrderDto> {
  protected readonly tableName = 'orders'

  constructor(connectionManager: ConnectionManager) {
    super(connectionManager)
  }

  /**
   * Create a new order
   */
  async createOrder(order: Order): Promise<void> {
    const model: Partial<OrderDto> = {
      id: order.id,
      symbol: order.symbol,
      side: order.side,
      type: order.type,
      status: order.status,
      price: order.price,
      size: order.size,
      filled_size: order.filledSize || 0,
      average_fill_price: order.averageFillPrice,
      stop_price: order.stopPrice,
      trail_distance: order.trailPercent || order.trailAmount,
      agent_id: order.agentId,
      metadata: order.metadata ? JSON.stringify(order.metadata) : null,
      submitted_at: order.submittedAt ? toIsoDate(order.submittedAt) : undefined,
      created_at: order.createdAt ? toIsoDate(order.createdAt) : isoDateNow(),
      updated_at: order.updatedAt ? toIsoDate(order.updatedAt) : isoDateNow(),
    }

    await this.insert(model)
  }

  /**
   * Update an existing order
   */
  async updateOrder(orderId: string, updates: Partial<Order>): Promise<void> {
    const updateData: Partial<OrderDto> = {
      updated_at: isoDateNow(),
    }

    if (updates.status !== undefined) updateData.status = updates.status
    if (updates.filledSize !== undefined) updateData.filled_size = updates.filledSize
    if (updates.averageFillPrice !== undefined) updateData.average_fill_price = updates.averageFillPrice
    if (updates.submittedAt !== undefined) updateData.submitted_at = toIsoDate(updates.submittedAt)
    if (updates.filledAt !== undefined) updateData.filled_at = toIsoDate(updates.filledAt)
    if (updates.cancelledAt !== undefined) updateData.cancelled_at = toIsoDate(updates.cancelledAt)
    if (updates.metadata !== undefined) updateData.metadata = JSON.stringify(updates.metadata)

    await this.update(updateData, 'id = ?', [orderId])
  }

  /**
   * Get an order by ID
   */
  async getOrder(orderId: string): Promise<Order | null> {
    const model = await this.findOne('id = ?', [orderId])
    return model ? this.dtoToOrder(model) : null
  }

  /**
   * Get orders by status
   */
  async getOrdersByStatus(status: OrderStatus, limit?: number): Promise<Order[]> {
    const models = await this.findMany(
      'status = ?',
      [status],
      'created_at DESC',
      limit,
    )

    return models.map(model => this.dtoToOrder(model))
  }

  /**
   * Get active orders for a symbol
   */
  async getActiveOrders(symbol?: string): Promise<Order[]> {
    const activeStatuses = ['pending', 'submitted', 'partial']
    let where = `status IN (${activeStatuses.map(() => '?').join(', ')})`
    const params: unknown[] = [...activeStatuses]

    if (symbol) {
      where += ' AND symbol = ?'
      params.push(symbol)
    }

    const models = await this.findMany(where, params, 'created_at DESC')
    return models.map(model => this.dtoToOrder(model))
  }

  /**
   * Get orders for an agent
   */
  async getOrdersByAgent(agentId: string, limit?: number): Promise<Order[]> {
    const models = await this.findMany(
      'agent_id = ?',
      [agentId],
      'created_at DESC',
      limit,
    )

    return models.map(model => this.dtoToOrder(model))
  }

  /**
   * Get order history within a time range
   */
  async getOrderHistory(
    symbol: string,
    startTime: EpochDate | Date,
    endTime: EpochDate | Date,
    statuses?: OrderStatus[],
  ): Promise<Order[]> {
    const st = startTime instanceof Date ? toIsoDate(startTime) : toIsoDate(startTime)
    const et = endTime instanceof Date ? toIsoDate(endTime) : toIsoDate(endTime)

    let where = 'symbol = ? AND created_at >= ? AND created_at <= ?'
    const params: unknown[] = [symbol, st, et]

    if (statuses && statuses.length > 0) {
      where += ` AND status IN (${statuses.map(() => '?').join(', ')})`
      params.push(...statuses)
    }

    const models = await this.findMany(where, params, 'created_at DESC')
    return models.map(model => this.dtoToOrder(model))
  }

  /**
   * Cancel an order
   */
  async cancelOrder(orderId: string, reason?: string): Promise<void> {
    const metadata = reason ? { cancellationReason: reason } : undefined

    await this.updateOrder(orderId, {
      status: 'cancelled',
      cancelledAt: epochDateNow(),
      metadata,
    })
  }

  /**
   * Get order statistics
   */
  async getOrderStats(symbol?: string, days = 30): Promise<{
    totalOrders: number
    filledOrders: number
    cancelledOrders: number
    averageFillRate: number
    ordersByType: Record<OrderType, number>
    ordersBySide: Record<OrderSide, number>
  }> {
    const startTime = epochDateNow() - (days * 24 * 60 * 60 * 1000)

    let whereClause = 'created_at >= ?'
    const params: unknown[] = [toIsoDate(startTime)]

    if (symbol) {
      whereClause += ' AND symbol = ?'
      params.push(symbol)
    }

    // Get order counts by status
    const statusCounts = await this.query<{ status: string; count: number }>(`
      SELECT status, COUNT(*) as count
      FROM ${this.tableName}
      WHERE ${whereClause}
      GROUP BY status
    `, params)

    const totalOrders = statusCounts.reduce((sum, row) => sum + row.count, 0)
    const filledOrders = statusCounts.find(row => row.status === 'filled')?.count || 0
    const cancelledOrders = statusCounts.find(row => row.status === 'cancelled')?.count || 0

    // Get order counts by type
    const typeCounts = await this.query<{ type: string; count: number }>(`
      SELECT type, COUNT(*) as count
      FROM ${this.tableName}
      WHERE ${whereClause}
      GROUP BY type
    `, params)

    const ordersByType = typeCounts.reduce((acc, row) => {
      acc[row.type as OrderType] = row.count
      return acc
    }, {} as Record<OrderType, number>)

    // Get order counts by side
    const sideCounts = await this.query<{ side: string; count: number }>(`
      SELECT side, COUNT(*) as count
      FROM ${this.tableName}
      WHERE ${whereClause}
      GROUP BY side
    `, params)

    const ordersBySide = sideCounts.reduce((acc, row) => {
      acc[row.side as OrderSide] = row.count
      return acc
    }, {} as Record<OrderSide, number>)

    return {
      totalOrders,
      filledOrders,
      cancelledOrders,
      averageFillRate: totalOrders > 0 ? filledOrders / totalOrders : 0,
      ordersByType,
      ordersBySide,
    }
  }

  /**
   * Cleanup old orders
   */
  async cleanup(daysToKeep = 90): Promise<number> {
    const cutoffDate = epochDateNow() - (daysToKeep * 24 * 60 * 60 * 1000)

    const countBefore = await this.count()

    await this.delete(
      'created_at < ? AND status IN (?, ?, ?)',
      [toIsoDate(cutoffDate), 'filled', 'cancelled', 'rejected'],
    )

    const countAfter = await this.count()
    return countBefore - countAfter
  }

  /**
   * Convert database dto to Order
   */
  private dtoToOrder(model: OrderDto): Order {
    return {
      id: model.id,
      symbol: model.symbol,
      side: model.side as OrderSide,
      type: model.type as OrderType,
      status: model.status as OrderStatus,
      price: model.price,
      size: model.size,
      filledSize: model.filled_size,
      averageFillPrice: model.average_fill_price,
      stopPrice: model.stop_price,
      trailPercent: model.trail_distance,
      trailAmount: model.trail_distance,
      agentId: model.agent_id,
      metadata: model.metadata ? JSON.parse(model.metadata as string) as Record<string, unknown> : undefined,
      createdAt: isoToEpoch(model.created_at),
      updatedAt: isoToEpoch(model.updated_at),
      submittedAt: model.submitted_at ? isoToEpoch(model.submitted_at) : undefined,
      filledAt: model.filled_at ? isoToEpoch(model.filled_at) : undefined,
      cancelledAt: model.cancelled_at ? isoToEpoch(model.cancelled_at) : undefined,
    }
  }
}
