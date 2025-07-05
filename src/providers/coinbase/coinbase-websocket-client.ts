import WebSocket from 'ws'
import logger from '../../utils/logger'
import type { OhlcvDto } from '../../models'
import type { WebSocketMessageType, WebSocketSubscribeMessage } from './types'

/**
 * Coinbase WebSocket client for real-time market data
 * Note: Coinbase Advanced Trade currently doesn't provide candle/OHLCV data via WebSocket
 * This implementation prepares for future WebSocket support or can be adapted for ticker data
 */
export class CoinbaseWebSocketClient {
  private ws?: WebSocket
  private readonly url: string
  private reconnectAttempts = 0
  private readonly maxReconnectAttempts = 5
  private readonly reconnectDelay = 1000
  private readonly subscriptions = new Map<string, Set<string>>()
  private messageQueue: OhlcvDto[] = []
  private isConnecting = false
  private isDisconnecting = false
  private heartbeatInterval?: NodeJS.Timeout
  private reconnectTimeoutId?: NodeJS.Timeout
  
  constructor(config: { sandbox?: boolean } = {}) {
    // Coinbase WebSocket endpoints
    this.url = config.sandbox 
      ? 'wss://ws-feed-public.sandbox.exchange.coinbase.com'
      : 'wss://ws-feed.exchange.coinbase.com'
  }
  
  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      logger.info('WebSocket already connected')
      return
    }
    
    if (this.isConnecting) {
      logger.info('WebSocket connection already in progress')
      return
    }
    
    this.isConnecting = true
    
    return new Promise((resolve, reject) => {
      try {
        logger.info('Connecting to Coinbase WebSocket', { url: this.url })
        
        this.ws = new WebSocket(this.url)
        
        this.ws.on('open', () => {
          logger.info('WebSocket connected')
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.startHeartbeat()
          
          // Resubscribe to previous subscriptions if reconnecting
          this.resubscribeAll()
          
          resolve()
        })
        
        this.ws.on('message', (data: WebSocket.Data) => {
          this.handleMessage(data)
        })
        
        this.ws.on('error', (error: Error) => {
          logger.error('WebSocket error', { error })
          this.isConnecting = false
          reject(error)
        })
        
        this.ws.on('close', (code: number, reason: Buffer) => {
          logger.info('WebSocket closed', { code, reason: reason.toString() })
          this.isConnecting = false
          this.stopHeartbeat()
          
          // Attempt reconnection if not manually closed
          if (code !== 1000) {
            this.attemptReconnect()
          }
        })
        
      } catch (error) {
        this.isConnecting = false
        logger.error('Failed to create WebSocket', { error })
        reject(error)
      }
    })
  }
  
  /**
   * Disconnect from WebSocket server
   */
  async disconnect(): Promise<void> {
    this.isDisconnecting = true
    this.stopHeartbeat()
    
    // Clear pending reconnection timeout
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId)
      this.reconnectTimeoutId = undefined
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = undefined
    }
    
    this.subscriptions.clear()
    this.messageQueue = []
    logger.info('WebSocket disconnected')
  }
  
  /**
   * Subscribe to a product and timeframe
   */
  async subscribe(symbol: string, granularity: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }
    
    // Track subscription
    if (!this.subscriptions.has(symbol)) {
      this.subscriptions.set(symbol, new Set())
    }
    this.subscriptions.get(symbol)!.add(granularity)
    
    // Send subscription message
    const message: WebSocketSubscribeMessage = {
      type: 'subscribe' as WebSocketMessageType.SUBSCRIBE,
      channels: ['ticker', 'matches'], // Coinbase doesn't provide candle channel yet
      product_ids: [symbol]
    }
    
    this.ws.send(JSON.stringify(message))
    logger.info('Subscribed to symbol', { symbol, granularity })
  }
  
  /**
   * Unsubscribe from a product
   */
  async unsubscribe(symbol: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return
    }
    
    this.subscriptions.delete(symbol)
    
    const message = {
      type: 'unsubscribe',
      channels: ['ticker', 'matches'],
      product_ids: [symbol]
    }
    
    this.ws.send(JSON.stringify(message))
    logger.info('Unsubscribed from symbol', { symbol })
  }
  
  /**
   * Get async iterator for data stream
   */
  async *getDataStream(): AsyncIterableIterator<{ symbol: string; data: any }> {
    while (this.ws?.readyState === WebSocket.OPEN && !this.isDisconnecting) {
      // Yield queued messages
      while (this.messageQueue.length > 0) {
        const data = this.messageQueue.shift()!
        yield { symbol: data.symbol, data }
      }
      
      // Wait for new messages with timeout to allow breaking from loop
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // Additional check to break loop when disconnecting
      if (this.isDisconnecting) {
        break
      }
    }
  }
  
  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: WebSocket.Data): void {
    try {
      const message = JSON.parse(data.toString())
      
      switch (message.type) {
        case 'ticker':
          // Ticker data could be aggregated into candles
          logger.debug('Received ticker', { product_id: message.product_id })
          break
          
        case 'match':
          // Trade data could be aggregated into candles
          logger.debug('Received match', { product_id: message.product_id })
          break
          
        case 'subscriptions':
          logger.info('Subscription confirmed', { channels: message.channels })
          break
          
        case 'error':
          logger.error('WebSocket error message', { message: message.message })
          break
          
        default:
          logger.debug('Unknown message type', { type: message.type })
      }
    } catch (error) {
      logger.error('Failed to parse WebSocket message', { error, data: data.toString() })
    }
  }
  
  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.ping()
      }
    }, 30000) // 30 seconds
  }
  
  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = undefined
    }
  }
  
  /**
   * Attempt to reconnect with exponential backoff
   */
  private async attemptReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached')
      return
    }
    
    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    
    logger.info('Attempting reconnection', { 
      attempt: this.reconnectAttempts, 
      delay,
      maxAttempts: this.maxReconnectAttempts 
    })
    
    this.reconnectTimeoutId = setTimeout(async () => {
      this.reconnectTimeoutId = undefined
      try {
        await this.connect()
      } catch (error) {
        logger.error('Reconnection failed', { error })
      }
    }, delay)
  }
  
  /**
   * Resubscribe to all previous subscriptions after reconnection
   */
  private resubscribeAll(): void {
    for (const [symbol, granularities] of this.subscriptions) {
      for (const granularity of granularities) {
        this.subscribe(symbol, granularity).catch(error => {
          logger.error('Failed to resubscribe', { symbol, granularity, error })
        })
      }
    }
  }
}