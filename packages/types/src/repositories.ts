import type { EpochDate, StockSymbol } from '@trdr/shared'
import type { Candle } from './market-data.js'

/**
 * Market data repository interface
 * Contract for storing and retrieving market data (candles, ticks)
 * Implement in data layer for database operations
 */
export interface MarketDataRepository {
  /**
   * Save a single candle to storage
   * @param candle - Candle to save
   */
  saveCandle(candle: Candle): Promise<void>
  
  /**
   * Save multiple candles in a batch
   * @param candles - Array of candles to save
   */
  saveCandlesBatch(candles: Candle[]): Promise<void>
  
  /**
   * Get candles for a symbol and interval within a time range
   * @param symbol - Trading symbol
   * @param interval - Candle interval (e.g. '1m')
   * @param startTime - Start time (epoch)
   * @param endTime - End time (epoch)
   * @param limit - Max number of candles
   * @returns Array of candles
   */
  getCandles(
    symbol: StockSymbol | string,
    interval: string,
    startTime: EpochDate,
    endTime: EpochDate,
    limit?: number
  ): Promise<Candle[]>
  
  /**
   * Get the latest candle for a symbol and interval
   * @param symbol - Trading symbol
   * @param interval - Candle interval
   * @returns Latest candle or null
   */
  getLatestCandle(symbol: StockSymbol | string, interval: string): Promise<Candle | null>
  
  /**
   * Clean up old data
   * @param olderThan - Delete data older than this epoch
   * @returns Number of candles and ticks deleted
   */
  cleanup(olderThan: EpochDate): Promise<{ candlesDeleted: number; ticksDeleted: number }>
}