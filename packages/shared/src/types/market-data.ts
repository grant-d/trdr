export interface Candle {
  readonly timestamp: number
  readonly open: number
  readonly high: number
  readonly low: number
  readonly close: number
  readonly volume: number
}

export interface OrderBook {
  readonly timestamp: number
  readonly bids: readonly OrderBookLevel[]
  readonly asks: readonly OrderBookLevel[]
}

export interface OrderBookLevel {
  readonly price: number
  readonly size: number
}

export interface Trade {
  readonly id: string
  readonly timestamp: number
  readonly price: number
  readonly size: number
  readonly side: 'buy' | 'sell'
  readonly maker: boolean
}

export interface Ticker {
  readonly symbol: string
  readonly timestamp: number
  readonly bid: number
  readonly ask: number
  readonly last: number
  readonly volume24h: number
}

export interface MarketDataSnapshot {
  readonly symbol: string
  readonly timestamp: number
  readonly candle: Candle
  readonly orderBook: OrderBook
  readonly ticker: Ticker
}

export type TimeInterval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

export interface MarketDataSubscription {
  readonly id: string
  readonly symbol: string
  readonly type: 'candle' | 'orderbook' | 'trade' | 'ticker'
  readonly interval?: TimeInterval
}