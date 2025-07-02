export type OrderSide = 'buy' | 'sell'
export type OrderStatus = 'pending' | 'open' | 'partial' | 'filled' | 'cancelled' | 'rejected'
export type OrderType = 'market' | 'limit' | 'stop' | 'trailing'

export interface OrderBase {
  readonly id: string
  readonly symbol: string
  readonly side: OrderSide
  readonly type: OrderType
  readonly size: number
  readonly createdAt: number
  readonly updatedAt: number
  readonly status: OrderStatus
}

export interface LimitOrder extends OrderBase {
  readonly type: 'limit'
  readonly price: number
}

export interface MarketOrder extends OrderBase {
  readonly type: 'market'
}

export interface StopOrder extends OrderBase {
  readonly type: 'stop'
  readonly stopPrice: number
  readonly limitPrice?: number
}

export interface TrailingOrder extends OrderBase {
  readonly type: 'trailing'
  readonly trailPercent: number
  readonly trailAmount?: number
  readonly limitPrice?: number
  readonly activationPrice?: number
  readonly highWaterMark?: number
  readonly lowWaterMark?: number
}

export type Order = LimitOrder | MarketOrder | StopOrder | TrailingOrder

export interface OrderFill {
  readonly id: string
  readonly orderId: string
  readonly timestamp: number
  readonly price: number
  readonly size: number
  readonly fee: number
  readonly feeCurrency: string
}

export interface Position {
  readonly symbol: string
  readonly size: number
  readonly avgEntryPrice: number
  readonly realizedPnl: number
  readonly unrealizedPnl: number
  readonly updatedAt: number
}

export interface OrderRequest {
  readonly symbol: string
  readonly side: OrderSide
  readonly type: OrderType
  readonly size: number
  readonly price?: number
  readonly stopPrice?: number
  readonly limitPrice?: number
  readonly trailPercent?: number
  readonly trailAmount?: number
  readonly timeInForce?: 'GTC' | 'IOC' | 'FOK'
  readonly postOnly?: boolean
}

export interface OrderUpdate {
  readonly orderId: string
  readonly price?: number
  readonly size?: number
  readonly stopPrice?: number
  readonly limitPrice?: number
}