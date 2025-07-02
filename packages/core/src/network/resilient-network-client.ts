import type {
  NetworkClient,
  NetworkResponse,
  RequestOptions,
  RetryConfig,
  NetworkError,
} from '../interfaces/network-client'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Default retry configuration
 */
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 5,
  initialDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: 0.1,
}

/**
 * Resilient network client with retry logic and exponential backoff
 *
 * @deprecated This client is deprecated in favor of using official exchange SDKs
 * (e.g., @coinbase-sample/advanced-trade-sdk-ts for Coinbase). Keep for potential
 * future use with data sources that don't provide official SDKs.
 */
export class ResilientNetworkClient implements NetworkClient {
  private retryConfig: RetryConfig
  private eventBus: EventBus

  constructor(retryConfig?: Partial<RetryConfig>) {
    this.retryConfig = { ...DEFAULT_RETRY_CONFIG, ...retryConfig }
    this.eventBus = EventBus.getInstance()
  }

  /**
   * Execute a request with retry logic
   */
  async request<T = unknown>(url: string, options?: RequestOptions): Promise<NetworkResponse<T>> {
    const requestOptions = {
      method: options?.method || 'GET',
      headers: options?.headers || {},
      body: options?.body ? JSON.stringify(options.body) : undefined,
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    }

    return this.executeWithRetry(
      async () => {
        const response = await fetch(url, requestOptions as RequestInit)

        if (!response.ok) {
          const error: NetworkError = new Error(`HTTP ${response.status}: ${response.statusText}`)
          error.status = response.status
          error.response = await response.text().catch(() => null)
          throw error
        }

        const data = await response.json() as T

        return {
          data,
          status: response.status,
          headers: Object.fromEntries(response.headers.entries()),
        }
      },
      `Request to ${url}`,
    )
  }

  /**
   * Execute an operation with retry logic
   */
  async executeWithRetry<T>(
    operation: () => Promise<T>,
    context: string,
  ): Promise<T> {
    let lastError: Error | undefined
    const config = this.retryConfig

    for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
      try {
        return await operation()
      } catch (error) {
        lastError = error as Error

        if (!this.isRetryable(error) || attempt === config.maxRetries) {
          this.logError(context, error as Error, attempt)
          throw error
        }

        const delay = this.calculateBackoff(attempt)
        this.logRetry(context, error as Error, attempt, delay)

        await this.sleep(delay)
      }
    }

    throw lastError!
  }

  /**
   * Check if an error is retryable
   */
  isRetryable(error: unknown): boolean {
    if (!(error instanceof Error)) {
      return false
    }

    const networkError = error as NetworkError

    // Network errors
    if (networkError.code === 'ECONNRESET' ||
      networkError.code === 'ETIMEDOUT' ||
      networkError.code === 'ENOTFOUND' ||
      networkError.code === 'ECONNREFUSED' ||
      networkError.code === 'EHOSTUNREACH') {
      return true
    }

    // HTTP status codes
    const status = networkError.status
    if (status === 429 || // Rate limited
      status === 502 || // Bad gateway
      status === 503 || // Service unavailable
      status === 504) { // Gateway timeout
      return true
    }

    // Exchange-specific errors
    if (error.message?.includes('request timestamp expired') ||
      error.message?.includes('nonce') ||
      error.message?.includes('rate limit')) {
      return true
    }

    return false
  }

  /**
   * Get current retry configuration
   */
  getRetryConfig(): RetryConfig {
    return { ...this.retryConfig }
  }

  /**
   * Update retry configuration
   */
  setRetryConfig(config: Partial<RetryConfig>): void {
    this.retryConfig = { ...this.retryConfig, ...config }
  }

  /**
   * Calculate backoff delay with jitter
   */
  private calculateBackoff(attempt: number): number {
    const exponentialDelay = this.retryConfig.initialDelay *
      Math.pow(this.retryConfig.backoffMultiplier, attempt)

    const clampedDelay = Math.min(exponentialDelay, this.retryConfig.maxDelay)

    // Add jitter to avoid thundering herd
    const jitter = clampedDelay * this.retryConfig.jitter * (Math.random() - 0.5) * 2

    return Math.round(clampedDelay + jitter)
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * Log retry attempt
   */
  private logRetry(context: string, error: Error, attempt: number, delay: number): void {
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      timestamp: new Date(),
      message: `${context} failed, retrying in ${delay}ms`,
      context: 'NetworkClient',
      details: {
        attempt: attempt + 1,
        error: error.message,
        delay,
      },
    })
  }

  /**
   * Log error
   */
  private logError(context: string, error: Error, attempts: number): void {
    this.eventBus.emit(EventTypes.SYSTEM_ERROR, {
      timestamp: new Date(),
      error,
      context: `NetworkClient: ${context}`,
      severity: attempts > 0 ? 'high' : 'medium',
    })
  }
}
