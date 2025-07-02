/**
 * Core trading engine exports
 */

// Event system
export * from './events'

// Interfaces
export * from './interfaces'

// Market data
export * from './market-data'

// Network
export * from './network'

// Database
export { createDatabaseWithEventBus } from './database/database-factory'

// Grid Trading
export * from './grid'

// Re-export shared types for convenience
export type {
  // Market data types
  Candle,
  MarketUpdate,
  MarketDataPipeline,
  PriceTick,
  Ticker,
  OrderBook,
  Trade,

  // Order types
  Order,
  OrderSide,
  OrderType,
  OrderStatus,
  OrderResult,
  OrderState,
  OrderEvent,
  Position,

  // Agent types
  ITradeAgent,
  AgentSignal,
  AgentContext,
  AgentState,
  AgentType,

  // Config types
  SystemConfig,
  TradingConfig,
  GridConfig,
  RiskConfig,
  MinimalConfig,
} from '@trdr/shared'

export const version = '1.0.0'
