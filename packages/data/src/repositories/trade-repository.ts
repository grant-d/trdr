import type { EpochDate, OrderSide, StockSymbol } from '@trdr/shared'
import { type IsoDate, toEpochDate, toIsoDate } from '@trdr/shared'
import type { ConnectionManager } from '../db/connection-manager'
import { BaseRepository } from './base-repository'

/**
 * Trade interface matching PRD
 */
export interface Trade {
  readonly id: string
  readonly orderId: string
  readonly symbol: StockSymbol
  readonly side: OrderSide
  readonly price: number
  readonly size: number
  readonly fee: number
  readonly feeCurrency?: string
  readonly pnl?: number
  readonly metadata?: Record<string, unknown>
  readonly executedAt: EpochDate
}

/**
 * Database trade dto
 */
interface TradeDto {
  id: string
  order_id: string
  symbol: string
  side: string
  price: number
  size: number
  fee: number
  fee_currency?: string
  pnl?: number
  metadata?: unknown
  executed_at: IsoDate
  created_at: IsoDate
}

/**
 * Repository for trade management
 */
export class TradeRepository extends BaseRepository<TradeDto> {
  protected readonly tableName = 'trades'

  constructor(connectionManager: ConnectionManager) {
    super(connectionManager)
  }

  /**
   * Record a new trade
   */
  async recordTrade(trade: Trade): Promise<void> {
    const model: Partial<TradeDto> = {
      id: trade.id,
      order_id: trade.orderId,
      symbol: trade.symbol,
      side: trade.side,
      price: trade.price,
      size: trade.size,
      fee: trade.fee,
      fee_currency: trade.feeCurrency,
      pnl: trade.pnl,
      metadata: trade.metadata ? JSON.stringify(trade.metadata) : null,
      executed_at: toIsoDate(trade.executedAt),
    }

    await this.insert(model)
  }

  /**
   * Record multiple trades in batch
   */
  async recordTradesBatch(trades: Trade[]): Promise<void> {
    const models: Partial<TradeDto>[] = trades.map(trade => ({
      id: trade.id,
      order_id: trade.orderId,
      symbol: trade.symbol,
      side: trade.side,
      price: trade.price,
      size: trade.size,
      fee: trade.fee,
      fee_currency: trade.feeCurrency,
      pnl: trade.pnl,
      metadata: trade.metadata ? JSON.stringify(trade.metadata) : null,
      executed_at: toIsoDate(trade.executedAt),
    }))

    await this.insertBatch(models)
  }

  /**
   * Get a trade by ID
   */
  async getTrade(tradeId: string): Promise<Trade | null> {
    const model = await this.findOne('id = ?', [tradeId])
    return model ? this.dtoToTrade(model) : null
  }

  /**
   * Get trades for an order
   */
  async getTradesByOrder(orderId: string): Promise<Trade[]> {
    const models = await this.findMany(
      'order_id = ?',
      [orderId],
      'executed_at ASC',
    )

    return models.map(model => this.dtoToTrade(model))
  }

  /**
   * Get trade history within a time range
   */
  async getTradeHistory(
    symbol: string,
    startTime: Date,
    endTime: Date,
    side?: OrderSide,
  ): Promise<Trade[]> {
    let where = 'symbol = ? AND executed_at >= ? AND executed_at <= ?'
    const params: unknown[] = [symbol, toIsoDate(startTime), toIsoDate(endTime)]

    if (side) {
      where += ' AND side = ?'
      params.push(side)
    }

    const models = await this.findMany(where, params, 'executed_at DESC')
    return models.map(model => this.dtoToTrade(model))
  }

  /**
   * Calculate P&L for a symbol
   */
  async calculatePnL(
    symbol: string,
    startTime?: Date,
    endTime?: Date,
  ): Promise<{
    totalPnL: number
    realizedPnL: number
    totalFees: number
    tradeCount: number
    winRate: number
  }> {
    let where = 'symbol = ?'
    const params: unknown[] = [symbol]

    if (startTime) {
      where += ' AND executed_at >= ?'
      params.push(toIsoDate(startTime))
    }

    if (endTime) {
      where += ' AND executed_at <= ?'
      params.push(toIsoDate(endTime))
    }

    const result = await this.query<{
      total_pnl: number
      total_fees: number
      trade_count: number
      winning_trades: number
    }>(`
      SELECT 
        COALESCE(SUM(pnl), 0) as total_pnl,
        COALESCE(SUM(fee), 0) as total_fees,
        COUNT(*) as trade_count,
        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades
      FROM ${this.tableName}
      WHERE ${where}
    `, params)

    const stats = result[0]
    return {
      totalPnL: stats?.total_pnl || 0,
      realizedPnL: (stats?.total_pnl || 0) - (stats?.total_fees || 0),
      totalFees: stats?.total_fees || 0,
      tradeCount: stats?.trade_count || 0,
      winRate: stats && stats.trade_count > 0
        ? stats.winning_trades / stats.trade_count
        : 0,
    }
  }

  /**
   * Get trade statistics by time period
   */
  async getTradeStatsByPeriod(
    symbol: string,
    period: 'hour' | 'day' | 'week' | 'month',
    limit = 30,
  ): Promise<Array<{
    period: string
    tradeCount: number
    volume: number
    avgPrice: number
    pnl: number
    fees: number
  }>> {
    const dateFormat = {
      hour: '%Y-%m-%d %H:00:00',
      day: '%Y-%m-%d',
      week: '%Y-%W',
      month: '%Y-%m',
    }[period]

    const results = await this.query<{
      period: string
      trade_count: number
      volume: number
      avg_price: number
      pnl: number
      fees: number
    }>(`
      SELECT 
        strftime('${dateFormat}', executed_at) as period,
        COUNT(*) as trade_count,
        SUM(size) as volume,
        AVG(price) as avg_price,
        COALESCE(SUM(pnl), 0) as pnl,
        COALESCE(SUM(fee), 0) as fees
      FROM ${this.tableName}
      WHERE symbol = ?
      GROUP BY period
      ORDER BY period DESC
      LIMIT ?
    `, [symbol, limit])

    return results.map(row => ({
      period: row.period,
      tradeCount: row.trade_count,
      volume: row.volume,
      avgPrice: row.avg_price,
      pnl: row.pnl,
      fees: row.fees,
    }))
  }

  /**
   * Get top performing trades
   */
  async getTopTrades(
    symbol?: string,
    limit = 10,
    orderBy: 'pnl' | 'size' | 'recent' = 'pnl',
  ): Promise<Trade[]> {
    let where = '1=1'
    const params: unknown[] = []

    if (symbol) {
      where += ' AND symbol = ?'
      params.push(symbol)
    }

    const orderClause = {
      pnl: 'pnl DESC',
      size: 'size DESC',
      recent: 'executed_at DESC',
    }[orderBy]

    const models = await this.findMany(where, params, orderClause, limit)
    return models.map(model => this.dtoToTrade(model))
  }

  /**
   * Cleanup old trades
   */
  async cleanup(daysToKeep = 365): Promise<number> {
    const cutoffDate = new Date()
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep)

    const countBefore = await this.count()
    await this.delete('executed_at < ?', [toIsoDate(cutoffDate)])
    const countAfter = await this.count()

    return countBefore - countAfter
  }

  /**
   * Convert database dto to Trade
   */
  private dtoToTrade(model: TradeDto): Trade {
    return {
      id: model.id,
      orderId: model.order_id,
      symbol: model.symbol,
      side: model.side as OrderSide,
      price: model.price,
      size: model.size,
      fee: model.fee,
      feeCurrency: model.fee_currency,
      pnl: model.pnl,
      metadata: model.metadata ? JSON.parse(model.metadata as string) as Record<string, unknown> : undefined,
      executedAt: toEpochDate(new Date(model.executed_at)),
    }
  }
}
