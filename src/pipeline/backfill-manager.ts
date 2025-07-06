import type { DataProvider, HistoricalParams } from '../interfaces'
import type { OhlcvDto } from '../models'
import type { Gap } from './gap-detector'
import { EventEmitter } from 'node:events'

/**
 * Backfill request
 */
export interface BackfillRequest {
  id: string
  symbol: string
  startTime: number
  endTime: number
  priority: 'low' | 'medium' | 'high'
  retries: number
  maxRetries: number
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  error?: string
  recordsReceived?: number
  createdAt: number
  startedAt?: number
  completedAt?: number
}

/**
 * Backfill configuration
 */
export interface BackfillConfig {
  /** Maximum concurrent backfill requests */
  maxConcurrent?: number
  /** Maximum retries per request */
  maxRetries?: number
  /** Retry delay in milliseconds */
  retryDelay?: number
  /** Request timeout in milliseconds */
  requestTimeout?: number
  /** Batch size for backfill requests */
  batchSize?: number
  /** Rate limit delay between requests in milliseconds */
  rateLimitDelay?: number
}

/**
 * Backfill events
 */
export interface BackfillEvents {
  requestCreated: [request: BackfillRequest]
  requestStarted: [request: BackfillRequest]
  requestCompleted: [request: BackfillRequest]
  requestFailed: [request: BackfillRequest, error: Error]
  requestRetrying: [request: BackfillRequest, attempt: number]
  data: [records: OhlcvDto[]]
}

/**
 * Backfill manager for handling gap filling
 */
export class BackfillManager extends EventEmitter {
  private readonly config: Required<BackfillConfig>
  private readonly provider: DataProvider
  private requests: Map<string, BackfillRequest> = new Map()
  private queue: BackfillRequest[] = []
  private activeRequests: Set<string> = new Set()
  private nextRequestId = 1

  constructor(provider: DataProvider, config: BackfillConfig = {}) {
    super()
    this.provider = provider
    this.config = {
      maxConcurrent: config.maxConcurrent ?? 3,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 5000,
      requestTimeout: config.requestTimeout ?? 60000,
      batchSize: config.batchSize ?? 1000,
      rateLimitDelay: config.rateLimitDelay ?? 1000
    }
  }

  /**
   * Create backfill requests from gaps
   */
  createRequestsFromGaps(gaps: Gap[]): BackfillRequest[] {
    const requests: BackfillRequest[] = []

    for (const gap of gaps) {
      if (gap.type === 'time_gap' || gap.type === 'missing_data') {
        const request = this.createRequest({
          symbol: gap.symbol,
          startTime: gap.startTime,
          endTime: gap.endTime,
          priority: gap.severity as 'low' | 'medium' | 'high'
        })
        requests.push(request)
      }
    }

    return requests
  }

  /**
   * Create a backfill request
   */
  createRequest(params: {
    symbol: string
    startTime: number
    endTime: number
    priority?: 'low' | 'medium' | 'high'
  }): BackfillRequest {
    const request: BackfillRequest = {
      id: `backfill-${this.nextRequestId++}`,
      symbol: params.symbol,
      startTime: params.startTime,
      endTime: params.endTime,
      priority: params.priority || 'medium',
      retries: 0,
      maxRetries: this.config.maxRetries,
      status: 'pending',
      createdAt: Date.now()
    }

    this.requests.set(request.id, request)
    this.queue.push(request)
    this.sortQueue()
    
    this.emit('requestCreated', request)
    this.processQueue()

    return request
  }

  /**
   * Process the backfill queue
   */
  private async processQueue(): Promise<void> {
    while (this.activeRequests.size < this.config.maxConcurrent && this.queue.length > 0) {
      const request = this.queue.shift()
      if (request) {
        this.activeRequests.add(request.id)
        this.processRequest(request).catch(error => {
          console.error(`Error processing backfill request ${request.id}:`, error)
        })
      }
    }
  }

  /**
   * Process a single backfill request
   */
  private async processRequest(request: BackfillRequest): Promise<void> {
    try {
      request.status = 'in_progress'
      request.startedAt = Date.now()
      this.emit('requestStarted', request)

      // Ensure provider is connected
      if (!this.provider.isConnected()) {
        await this.provider.connect()
      }

      // Create historical params
      const params: HistoricalParams = {
        symbols: [request.symbol],
        start: request.startTime,
        end: request.endTime,
        timeframe: '1m'
      }

      // Set up timeout
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout')), this.config.requestTimeout)
      })

      // Fetch data with timeout
      const dataPromise = this.fetchData(params)
      const records = await Promise.race([dataPromise, timeoutPromise])

      // Update request
      request.recordsReceived = records.length
      request.status = 'completed'
      request.completedAt = Date.now()
      
      this.emit('requestCompleted', request)
      this.emit('data', records)

    } catch (error) {
      await this.handleRequestError(request, error as Error)
    } finally {
      this.activeRequests.delete(request.id)
      await this.delay(this.config.rateLimitDelay)
      this.processQueue()
    }
  }

  /**
   * Fetch data from provider
   */
  private async fetchData(params: HistoricalParams): Promise<OhlcvDto[]> {
    const records: OhlcvDto[] = []
    const stream = this.provider.getHistoricalData(params)

    for await (const record of stream) {
      records.push(record)
      
      if (records.length >= this.config.batchSize) {
        break
      }
    }

    return records
  }

  /**
   * Handle request error
   */
  private async handleRequestError(request: BackfillRequest, error: Error): Promise<void> {
    request.retries++
    request.error = error.message

    if (request.retries < request.maxRetries) {
      request.status = 'pending'
      this.emit('requestRetrying', request, request.retries)
      
      // Add back to queue with delay
      await this.delay(this.config.retryDelay * request.retries)
      this.queue.push(request)
      this.sortQueue()
    } else {
      request.status = 'failed'
      request.completedAt = Date.now()
      this.emit('requestFailed', request, error)
    }
  }

  /**
   * Sort queue by priority and creation time
   */
  private sortQueue(): void {
    const priorityWeight = { high: 3, medium: 2, low: 1 }
    
    this.queue.sort((a, b) => {
      const priorityDiff = priorityWeight[b.priority] - priorityWeight[a.priority]
      if (priorityDiff !== 0) return priorityDiff
      
      return a.createdAt - b.createdAt
    })
  }

  /**
   * Get request status
   */
  getRequest(id: string): BackfillRequest | undefined {
    return this.requests.get(id)
  }

  /**
   * Get all requests
   */
  getAllRequests(): BackfillRequest[] {
    return Array.from(this.requests.values())
  }

  /**
   * Get queue status
   */
  getQueueStatus(): {
    pending: number
    active: number
    completed: number
    failed: number
    total: number
  } {
    const requests = this.getAllRequests()
    
    return {
      pending: requests.filter(r => r.status === 'pending').length,
      active: requests.filter(r => r.status === 'in_progress').length,
      completed: requests.filter(r => r.status === 'completed').length,
      failed: requests.filter(r => r.status === 'failed').length,
      total: requests.length
    }
  }

  /**
   * Cancel a request
   */
  cancelRequest(id: string): boolean {
    const request = this.requests.get(id)
    if (!request || request.status !== 'pending') {
      return false
    }

    // Remove from queue
    const index = this.queue.findIndex(r => r.id === id)
    if (index !== -1) {
      this.queue.splice(index, 1)
    }

    // Update status
    request.status = 'failed'
    request.error = 'Cancelled by user'
    request.completedAt = Date.now()

    return true
  }

  /**
   * Clear completed requests
   */
  clearCompleted(): number {
    const completed = Array.from(this.requests.entries())
      .filter(([_, request]) => request.status === 'completed')

    for (const [id] of completed) {
      this.requests.delete(id)
    }

    return completed.length
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * Type-safe event emitter methods
   */
  emit<K extends keyof BackfillEvents>(
    event: K,
    ...args: BackfillEvents[K]
  ): boolean {
    return super.emit(event, ...args)
  }

  on<K extends keyof BackfillEvents>(
    event: K,
    listener: (...args: BackfillEvents[K]) => void
  ): this {
    return super.on(event, listener)
  }

  once<K extends keyof BackfillEvents>(
    event: K,
    listener: (...args: BackfillEvents[K]) => void
  ): this {
    return super.once(event, listener)
  }

  off<K extends keyof BackfillEvents>(
    event: K,
    listener: (...args: BackfillEvents[K]) => void
  ): this {
    return super.off(event, listener)
  }
}