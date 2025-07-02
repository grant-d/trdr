import { epochDateNow, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { LiveDataSource } from './live-data-collector'

/**
 * Coinbase WebSocket message types
 */
interface CoinbaseWebSocketMessage {
  type: string
  product_id?: string
  time?: string
  price?: string
  size?: string
  side?: 'buy' | 'sell'
  best_bid?: string
  best_ask?: string
  channels?: Array<string | { name: string; product_ids: string[] }>
}

interface CoinbaseTickerMessage extends CoinbaseWebSocketMessage {
  type: 'ticker'
  product_id: string
  time: string
  price: string
  open_24h: string
  volume_24h: string
  low_24h: string
  high_24h: string
  volume_30d: string
  best_bid: string
  best_ask: string
  side: 'buy' | 'sell'
  last_size: string
}

interface CoinbaseMatchMessage extends CoinbaseWebSocketMessage {
  type: 'match'
  trade_id: number
  product_id: string
  time: string
  price: string
  size: string
  side: 'buy' | 'sell'
}

/**
 * Configuration for Coinbase WebSocket source
 */
export interface CoinbaseWebSocketConfig {
  /** WebSocket URL (default: wss://ws-feed.exchange.coinbase.com) */
  readonly wsUrl?: string
  /** Channels to subscribe to (default: ['ticker', 'matches']) */
  readonly channels?: string[]
  /** Enable heartbeat monitoring */
  readonly enableHeartbeat?: boolean
  /** Heartbeat timeout in ms */
  readonly heartbeatTimeoutMs?: number
  /** Max reconnection attempts */
  readonly maxReconnectAttempts?: number
  /** Reconnect delay in ms */
  readonly reconnectDelayMs?: number
}

/**
 * Coinbase WebSocket data source implementation
 * Provides real-time market data via WebSocket connection
 */
export class CoinbaseWebSocketSource implements LiveDataSource {
  readonly name = 'coinbase-websocket'
  readonly type = 'websocket' as const
  
  private readonly config: Required<CoinbaseWebSocketConfig>
  private readonly eventBus: EventBus
  private readonly logger?: Logger
  
  private ws?: WebSocket
  private readonly subscribedSymbols = new Set<string>()
  private connected = false
  private reconnectAttempts = 0
  private heartbeatTimer?: NodeJS.Timeout
  private lastHeartbeat?: number
  private reconnectTimer?: NodeJS.Timeout
  
  constructor(config: CoinbaseWebSocketConfig = {}, logger?: Logger) {
    this.config = {
      wsUrl: config.wsUrl ?? 'wss://ws-feed.exchange.coinbase.com',
      channels: config.channels ?? ['ticker', 'matches'],
      enableHeartbeat: config.enableHeartbeat ?? true,
      heartbeatTimeoutMs: config.heartbeatTimeoutMs ?? 30000,
      maxReconnectAttempts: config.maxReconnectAttempts ?? 5,
      reconnectDelayMs: config.reconnectDelayMs ?? 1000
    }
    
    this.eventBus = EventBus.getInstance()
    this.logger = logger
  }

  /**
   * Connect to WebSocket
   */
  async connect(): Promise<void> {
    if (this.ws && this.connected) {
      this.logger?.warn('Already connected to Coinbase WebSocket')
      return
    }
    
    return new Promise((resolve, reject) => {
      try {
        this.logger?.info('Connecting to Coinbase WebSocket', { url: this.config.wsUrl })
        
        // In Node.js environment, we'd need to use a WebSocket library
        // For browser environment, we can use native WebSocket
        if (typeof WebSocket === 'undefined') {
          throw new Error('WebSocket not available. Use ws package in Node.js environment.')
        }
        
        this.ws = new WebSocket(this.config.wsUrl)
        
        this.ws.onopen = () => {
          this.logger?.info('Connected to Coinbase WebSocket')
          this.connected = true
          this.reconnectAttempts = 0
          
          this.emitConnectionStatus('connected')
          
          // Start heartbeat monitoring
          if (this.config.enableHeartbeat) {
            this.startHeartbeatMonitoring()
          }
          
          // Resubscribe to previously subscribed symbols
          if (this.subscribedSymbols.size > 0) {
            this.sendSubscriptionMessage(Array.from(this.subscribedSymbols), 'subscribe')
          }
          
          resolve()
        }
        
        this.ws.onmessage = (event) => {
          this.handleMessage(event.data)
        }
        
        this.ws.onerror = (error) => {
          this.logger?.error('WebSocket error', { error })
          this.emitConnectionStatus('error')
        }
        
        this.ws.onclose = (event) => {
          this.logger?.info('WebSocket closed', { 
            code: event.code, 
            reason: event.reason 
          })
          
          this.connected = false
          this.stopHeartbeatMonitoring()
          this.emitConnectionStatus('disconnected')
          
          // Auto-reconnect if not a normal closure
          if (event.code !== 1000 && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }
        
        // Set connection timeout
        setTimeout(() => {
          if (!this.connected) {
            this.ws?.close()
            reject(new Error('Connection timeout'))
          }
        }, 10000)
        
      } catch (error) {
        this.logger?.error('Failed to create WebSocket', { error })
        reject(error)
      }
    })
  }

  /**
   * Subscribe to symbols
   */
  async subscribe(symbols: string[]): Promise<void> {
    if (!this.ws || !this.connected) {
      // Store for later subscription when connected
      symbols.forEach(symbol => this.subscribedSymbols.add(symbol))
      
      if (!this.ws) {
        await this.connect()
      }
      return
    }
    
    // Add to subscribed set
    symbols.forEach(symbol => this.subscribedSymbols.add(symbol))
    
    // Send subscription message
    this.sendSubscriptionMessage(symbols, 'subscribe')
  }

  /**
   * Unsubscribe from symbols
   */
  async unsubscribe(symbols: string[]): Promise<void> {
    if (!this.ws || !this.connected) {
      // Just remove from set
      symbols.forEach(symbol => this.subscribedSymbols.delete(symbol))
      return
    }
    
    // Remove from subscribed set
    symbols.forEach(symbol => this.subscribedSymbols.delete(symbol))
    
    // Send unsubscribe message
    this.sendSubscriptionMessage(symbols, 'unsubscribe')
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected && this.ws?.readyState === WebSocket.OPEN
  }

  /**
   * Reconnect to WebSocket
   */
  async reconnect(): Promise<void> {
    this.logger?.info('Attempting to reconnect')
    
    // Close existing connection
    if (this.ws) {
      this.ws.close()
      this.ws = undefined
    }
    
    this.connected = false
    this.reconnectAttempts++
    
    await this.connect()
  }

  /**
   * Disconnect from WebSocket
   */
  async disconnect(): Promise<void> {
    this.logger?.info('Disconnecting from Coinbase WebSocket')
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
    
    this.stopHeartbeatMonitoring()
    
    if (this.ws) {
      this.ws.close(1000, 'Normal closure')
      this.ws = undefined
    }
    
    this.connected = false
    this.subscribedSymbols.clear()
  }

  /**
   * Send subscription message
   */
  private sendSubscriptionMessage(symbols: string[], action: 'subscribe' | 'unsubscribe'): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.logger?.warn('Cannot send message, WebSocket not open')
      return
    }
    
    const message = {
      type: action,
      product_ids: symbols,
      channels: this.config.channels
    }
    
    this.logger?.debug('Sending subscription message', { action, symbols })
    
    try {
      this.ws.send(JSON.stringify(message))
    } catch (error) {
      this.logger?.error('Failed to send subscription message', { error })
    }
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data) as CoinbaseWebSocketMessage
      
      // Update heartbeat
      this.lastHeartbeat = Date.now()
      
      switch (message.type) {
        case 'ticker':
          this.handleTickerMessage(message as CoinbaseTickerMessage)
          break
          
        case 'match':
          this.handleMatchMessage(message as CoinbaseMatchMessage)
          break
          
        case 'heartbeat':
          // Just update heartbeat timestamp
          break
          
        case 'subscriptions':
          this.logger?.debug('Subscription confirmed', { message })
          break
          
        case 'error':
          this.logger?.error('Received error message', { message })
          break
          
        default:
          this.logger?.debug('Unhandled message type', { type: message.type })
      }
      
    } catch (error) {
      this.logger?.error('Failed to parse message', { error, data })
    }
  }

  /**
   * Handle ticker message
   */
  private handleTickerMessage(message: CoinbaseTickerMessage): void {
    const candle = {
      symbol: toStockSymbol(message.product_id),
      interval: '1m', // Ticker data represents current state
      timestamp: toEpochDate(new Date(message.time).getTime()),
      openTime: toEpochDate(new Date(message.time).getTime()),
      closeTime: toEpochDate(new Date(message.time).getTime() + 60000),
      open: parseFloat(message.price),
      high: parseFloat(message.high_24h),
      low: parseFloat(message.low_24h),
      close: parseFloat(message.price),
      volume: parseFloat(message.volume_24h),
      bid: parseFloat(message.best_bid),
      ask: parseFloat(message.best_ask)
    }
    
    // Emit candle event
    this.eventBus.emit(EventTypes.CANDLE, {
      candle,
      source: this.name,
      timestamp: epochDateNow()
    })
    
    // Also emit tick event
    this.eventBus.emit(EventTypes.TICK, {
      symbol: message.product_id,
      price: parseFloat(message.price),
      bid: parseFloat(message.best_bid),
      ask: parseFloat(message.best_ask),
      timestamp: epochDateNow()
    })
  }

  /**
   * Handle match (trade) message
   */
  private handleMatchMessage(message: CoinbaseMatchMessage): void {
    // Emit trade event
    this.eventBus.emit(EventTypes.TICK, {
      symbol: message.product_id,
      price: parseFloat(message.price),
      size: parseFloat(message.size),
      side: message.side,
      timestamp: toEpochDate(new Date(message.time).getTime())
    })
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeatMonitoring(): void {
    this.stopHeartbeatMonitoring()
    
    this.lastHeartbeat = Date.now()
    
    this.heartbeatTimer = setInterval(() => {
      if (!this.lastHeartbeat || Date.now() - this.lastHeartbeat > this.config.heartbeatTimeoutMs) {
        this.logger?.warn('Heartbeat timeout, connection may be dead')
        
        // Force reconnection
        if (this.ws) {
          this.ws.close()
        }
      }
    }, this.config.heartbeatTimeoutMs / 2)
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeatMonitoring(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = undefined
    }
  }

  /**
   * Schedule reconnection
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return // Already scheduled
    }
    
    const delay = this.config.reconnectDelayMs * Math.pow(2, this.reconnectAttempts)
    
    this.logger?.info('Scheduling reconnection', { 
      attempt: this.reconnectAttempts + 1,
      delay 
    })
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined
      this.reconnect().catch(error => {
        this.logger?.error('Reconnection failed', { error })
      })
    }, delay)
  }

  /**
   * Emit connection status event
   */
  private emitConnectionStatus(status: 'connected' | 'disconnected' | 'error'): void {
    this.eventBus.emit(EventTypes.CONNECTION_STATUS, {
      source: this.name,
      status,
      timestamp: epochDateNow()
    })
  }
}