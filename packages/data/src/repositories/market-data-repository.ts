import { BaseRepository } from './base-repository'
import { ConnectionManager } from '../db/connection-manager'
import { Candle, PriceTick } from '../types/market-data'
import { type IsoDate, toIsoDate } from '@trdr/shared'

/**
 * Database candle dto
 */
interface CandleDto {
  id: number
  symbol: string
  interval: string
  open_time: IsoDate
  close_time: IsoDate
  open: number
  high: number
  low: number
  close: number
  volume: number
  quote_volume?: number
  trades_count?: number
  created_at: IsoDate
}

/**
 * Database price tick dto
 */
interface PriceTickDto {
  id: number
  symbol: string
  price: number
  volume: number
  timestamp: IsoDate
  bid?: number
  ask?: number
  bid_size?: number
  ask_size?: number
  created_at: IsoDate
}

/**
 * Repository for market data (candles and ticks)
 */
export class MarketDataRepository extends BaseRepository<CandleDto> {
  protected readonly tableName = 'candles'
  private ticksTableName = 'market_ticks'
  private idCounter = Date.now()

  constructor(connectionManager: ConnectionManager) {
    super(connectionManager)
  }

  /**
   * Save a single candle
   */
  async saveCandle(candle: Candle): Promise<void> {
    const model: Partial<CandleDto> = {
      id: ++this.idCounter,
      symbol: candle.symbol,
      interval: candle.interval,
      open_time: toIsoDate(candle.openTime),
      close_time: toIsoDate(candle.closeTime),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      quote_volume: candle.quoteVolume,
      trades_count: candle.tradesCount,
    }

    await this.insert(model)
  }

  /**
   * Save multiple candles in batch
   */
  async saveCandlesBatch(candles: Candle[]): Promise<void> {
    const models: Partial<CandleDto>[] = candles.map(candle => ({
      id: ++this.idCounter,
      symbol: candle.symbol,
      interval: candle.interval,
      open_time: toIsoDate(candle.openTime),
      close_time: toIsoDate(candle.closeTime),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      quote_volume: candle.quoteVolume,
      trades_count: candle.tradesCount,
    }))

    await this.insertBatch(models)
  }

  /**
   * Get candles for a symbol within a time range
   */
  async getCandles(
    symbol: string,
    interval: string,
    startTime: Date,
    endTime: Date,
    limit?: number,
  ): Promise<Candle[]> {
    let sql = `
      SELECT * FROM ${this.tableName}
      WHERE symbol = ? 
        AND interval = ?
        AND open_time >= ?
        AND open_time <= ?
      ORDER BY open_time ASC
    `

    if (limit) {
      sql += ` LIMIT ${limit}`
    }

    const params = [symbol, interval, startTime.toISOString(), endTime.toISOString()]
    const models = await this.query<CandleDto>(sql, params)

    return models.map(this.dtoToCandle)
  }

  /**
   * Get the latest candle for a symbol
   */
  async getLatestCandle(symbol: string, interval: string): Promise<Candle | null> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE symbol = ? AND interval = ?
      ORDER BY open_time DESC
      LIMIT 1
    `

    const models = await this.query<CandleDto>(sql, [symbol, interval])
    return models.length > 0 && models[0] ? this.dtoToCandle(models[0]) : null
  }

  /**
   * Save a price tick
   */
  async saveTick(tick: PriceTick): Promise<void> {
    const model: Partial<PriceTickDto> = {
      id: ++this.idCounter,
      symbol: tick.symbol,
      price: tick.price,
      volume: tick.volume,
      timestamp: toIsoDate(tick.timestamp),
      bid: tick.bid,
      ask: tick.ask,
      bid_size: tick.bidSize,
      ask_size: tick.askSize,
    }

    const fields = Object.keys(model)
    const values = Object.values(model)
    const placeholders = fields.map(() => '?').join(', ')

    const sql = `INSERT INTO ${this.ticksTableName} (${fields.join(', ')}) VALUES (${placeholders})`
    await this.connectionManager.execute(sql, values)
  }

  /**
   * Save multiple ticks in batch
   */
  async saveTicksBatch(ticks: PriceTick[]): Promise<void> {
    if (ticks.length === 0) return

    // SQLite doesn't support multi-row inserts with a single statement efficiently
    // Use a transaction for batch insert instead
    const models: Partial<PriceTickDto>[] = ticks.map(tick => ({
      id: ++this.idCounter,
      symbol: tick.symbol,
      price: tick.price,
      volume: tick.volume,
      timestamp: toIsoDate(tick.timestamp),
      bid: tick.bid,
      ask: tick.ask,
      bid_size: tick.bidSize,
      ask_size: tick.askSize,
    }))

    await this.connectionManager.transaction((db) => {
      const firstModel = models[0]
      if (!firstModel) return 0

      const fields = Object.keys(firstModel)
      const placeholders = fields.map(() => '?').join(', ')
      const sql = `INSERT INTO ${this.ticksTableName} (${fields.join(', ')}) VALUES (${placeholders})`

      const stmt = db.prepare(sql)
      let count = 0
      for (const model of models) {
        const values = fields.map(field => (model as any)[field])
        stmt.run(...values)
        count++
      }
      return count
    })
  }

  /**
   * Get ticks for a symbol within a time range
   */
  async getTicks(
    symbol: string,
    startTime: Date,
    endTime: Date,
    limit?: number,
  ): Promise<PriceTick[]> {
    let sql = `
      SELECT * FROM ${this.ticksTableName}
      WHERE symbol = ?
        AND timestamp >= ?
        AND timestamp <= ?
      ORDER BY timestamp ASC
    `

    if (limit) {
      sql += ` LIMIT ${limit}`
    }

    const params = [symbol, toIsoDate(startTime), toIsoDate(endTime)]
    const models = await this.query<PriceTickDto>(sql, params)

    return models.map(this.dtoToTick)
  }

  /**
   * Get the latest tick for a symbol
   */
  async getLatestTick(symbol: string): Promise<PriceTick | null> {
    const sql = `
      SELECT * FROM ${this.ticksTableName}
      WHERE symbol = ?
      ORDER BY timestamp DESC
      LIMIT 1
    `

    const models = await this.query<PriceTickDto>(sql, [symbol])
    return models.length > 0 && models[0] ? this.dtoToTick(models[0]) : null
  }

  /**
   * Get aggregated stats for a symbol
   */
  async getMarketStats(
    symbol: string,
    interval: string,
    days: number = 30,
  ): Promise<{
    avgVolume: number
    avgPrice: number
    priceRange: { min: number; max: number }
    volatility: number
  }> {
    const startTime = new Date()
    startTime.setDate(startTime.getDate() - days)

    // SQLite doesn't have STDDEV, calculate volatility using variance
    const sql = `
      SELECT 
        AVG(volume) as avg_volume,
        AVG((high + low + close) / 3) as avg_price,
        MIN(low) as min_price,
        MAX(high) as max_price,
        AVG((high + low + close) / 3) as mean_price,
        AVG(((high + low + close) / 3) * ((high + low + close) / 3)) as mean_price_squared
      FROM ${this.tableName}
      WHERE symbol = ?
        AND interval = ?
        AND open_time >= ?
    `

    const results = await this.query<{
      avg_volume: number
      avg_price: number
      min_price: number
      max_price: number
      mean_price: number
      mean_price_squared: number
    }>(sql, [symbol, interval, toIsoDate(startTime)])

    const stats = results[0]
    // Calculate standard deviation manually: sqrt(E[X^2] - E[X]^2)
    const variance = (stats?.mean_price_squared || 0) - Math.pow(stats?.mean_price || 0, 2)
    const volatility = Math.sqrt(Math.max(0, variance))

    return {
      avgVolume: stats?.avg_volume || 0,
      avgPrice: stats?.avg_price || 0,
      priceRange: {
        min: stats?.min_price || 0,
        max: stats?.max_price || 0,
      },
      volatility,
    }
  }

  /**
   * Delete old market data
   */
  async cleanup(daysToKeep: number = 90): Promise<{ candlesDeleted: number; ticksDeleted: number }> {
    const cutoffDate = new Date()
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep)

    // Count before deletion
    const candlesBefore = await this.count()
    const ticksBefore = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.ticksTableName}`,
    ).then(r => r[0]?.count || 0)

    // Delete old data
    await this.delete('open_time < ?', [toIsoDate(cutoffDate)])
    await this.connectionManager.execute(
      `DELETE FROM ${this.ticksTableName} WHERE timestamp < ?`,
      [toIsoDate(cutoffDate)],
    )

    // Count after deletion
    const candlesAfter = await this.count()
    const ticksAfter = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.ticksTableName}`,
    ).then(r => r[0]?.count || 0)

    return {
      candlesDeleted: candlesBefore - candlesAfter,
      ticksDeleted: ticksBefore - ticksAfter,
    }
  }

  /**
   * Convert database dto to Candle
   */
  private dtoToCandle(model: CandleDto): Candle {
    return {
      symbol: model.symbol,
      interval: model.interval,
      timestamp: new Date(model.open_time),
      openTime: new Date(model.open_time),
      closeTime: new Date(model.close_time),
      open: model.open,
      high: model.high,
      low: model.low,
      close: model.close,
      volume: model.volume,
      quoteVolume: model.quote_volume,
      tradesCount: model.trades_count,
    }
  }

  /**
   * Convert database dto to PriceTick
   */
  private dtoToTick(model: PriceTickDto): PriceTick {
    return {
      symbol: model.symbol,
      price: model.price,
      volume: model.volume,
      timestamp: new Date(model.timestamp),
      bid: model.bid,
      ask: model.ask,
      bidSize: model.bid_size,
      askSize: model.ask_size,
    }
  }
}
