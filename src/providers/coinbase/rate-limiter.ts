import logger from '../../utils/logger'

/**
 * Rate limiter configuration
 */
export interface RateLimiterConfig {
  /** Maximum requests per second */
  maxRequestsPerSecond: number

  /** Maximum retry attempts */
  maxRetries: number

  /** Initial retry delay in milliseconds */
  initialRetryDelayMs: number

  /** Maximum retry delay in milliseconds */
  maxRetryDelayMs: number

  /** Backoff multiplier for exponential backoff */
  backoffMultiplier: number
}

/**
 * Default rate limiter configuration
 */
export const DEFAULT_RATE_LIMITER_CONFIG: RateLimiterConfig = {
  maxRequestsPerSecond: 10,
  maxRetries: 5,
  initialRetryDelayMs: 1000,
  maxRetryDelayMs: 60000,
  backoffMultiplier: 2
}

/**
 * Rate limiter with exponential backoff for API requests
 */
export class RateLimiter {
  private readonly config: RateLimiterConfig
  private requestCount = 0
  private resetTime = Date.now()

  constructor(config: Partial<RateLimiterConfig> = {}) {
    this.config = { ...DEFAULT_RATE_LIMITER_CONFIG, ...config }
  }

  /**
   * Execute a function with rate limiting
   */
  async execute<T>(fn: () => Promise<T>, context?: string): Promise<T> {
    await this.waitForRateLimit()

    let lastError: Error | undefined

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        const result = await fn()
        return result
      } catch (error) {
        lastError = error as Error

        // Check if it's a rate limit error
        if (this.isRateLimitError(error)) {
          const delay = this.calculateBackoffDelay(attempt)
          logger.warn('Rate limit hit, backing off', {
            attempt: attempt + 1,
            maxAttempts: this.config.maxRetries,
            delayMs: delay,
            context
          })

          await this.delay(delay)
        } else {
          // If it's not a rate limit error, throw immediately
          throw error
        }
      }
    }

    // If we've exhausted all retries, throw the last error
    throw new Error(
      `Rate limit retry exhausted after ${this.config.maxRetries} attempts: ${lastError?.message}`
    )
  }

  /**
   * Wait for rate limit window if necessary
   */
  private async waitForRateLimit(): Promise<void> {
    const now = Date.now()

    // Reset counter if we're in a new second
    if (now - this.resetTime >= 1000) {
      this.requestCount = 0
      this.resetTime = now
    }

    // If we've hit the rate limit, wait
    if (this.requestCount >= this.config.maxRequestsPerSecond) {
      const waitTime = 1000 - (now - this.resetTime)
      if (waitTime > 0) {
        logger.debug('Rate limiting, waiting', { waitTimeMs: waitTime })
        await this.delay(waitTime)

        // Reset after waiting
        this.requestCount = 0
        this.resetTime = Date.now()
      }
    }

    this.requestCount++
  }

  /**
   * Calculate exponential backoff delay
   */
  private calculateBackoffDelay(attempt: number): number {
    const delay = Math.min(
      this.config.initialRetryDelayMs *
      Math.pow(this.config.backoffMultiplier, attempt),
      this.config.maxRetryDelayMs
    )

    // Add jitter (±10%) to prevent thundering herd
    const jitter = delay * 0.1 * (Math.random() * 2 - 1)

    return Math.floor(delay + jitter)
  }

  /**
   * Check if error is a rate limit error
   */
  private isRateLimitError(error: any): boolean {
    // Common rate limit error patterns
    if (error?.response?.status === 429) {
      return true
    }

    if (error?.message?.toLowerCase().includes('rate limit')) {
      return true
    }

    if (error?.code === 'RATE_LIMIT_EXCEEDED') {
      return true
    }

    // Coinbase-specific rate limit errors
    if (error?.response?.data?.message?.includes('rate limit')) {
      return true
    }

    return false
  }

  /**
   * Delay execution for specified milliseconds
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  /**
   * Get current rate limit status
   */
  getStatus(): { requestCount: number; resetTime: number } {
    return {
      requestCount: this.requestCount,
      resetTime: this.resetTime
    }
  }
}
