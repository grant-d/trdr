/**
 * Configuration for retry behavior
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  readonly maxRetries: number
  /** Initial delay in milliseconds before first retry */
  readonly initialDelay: number
  /** Maximum delay in milliseconds between retries */
  readonly maxDelay: number
  /** Multiplier for exponential backoff */
  readonly backoffMultiplier: number
  /** Jitter factor (0-1) to randomize delays */
  readonly jitter: number
}

/**
 * Network request options
 */
export interface RequestOptions {
  /** HTTP method */
  readonly method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  /** Request headers */
  readonly headers?: Record<string, string>
  /** Request body */
  readonly body?: unknown
  /** Request timeout in milliseconds */
  readonly timeout?: number
  /** Custom retry configuration for this request */
  readonly retryConfig?: Partial<RetryConfig>
}

/**
 * Network response
 */
export interface NetworkResponse<T = unknown> {
  /** Response data */
  readonly data: T
  /** HTTP status code */
  readonly status: number
  /** Response headers */
  readonly headers: Record<string, string>
}

/**
 * Network error with additional context
 */
export interface NetworkError extends Error {
  /** Error code (e.g., 'ECONNRESET', 'ETIMEDOUT') */
  code?: string
  /** HTTP status code if applicable */
  status?: number
  /** Response body if available */
  response?: unknown
  /** Number of retry attempts made */
  retryAttempts?: number
}

/**
 * Network client interface for resilient HTTP requests
 *
 * @deprecated This interface is deprecated in favor of using official exchange SDKs
 * (e.g., @coinbase-sample/advanced-trade-sdk-ts for Coinbase). Keep for potential
 * future use with data sources that don't provide official SDKs.
 */
export interface NetworkClient {
  /**
   * Execute a request with retry logic
   * @param url - Request URL
   * @param options - Request options
   * @returns Response data
   */
  request<T = unknown>(url: string, options?: RequestOptions): Promise<NetworkResponse<T>>

  /**
   * Execute an operation with retry logic
   * @param operation - Async operation to execute
   * @param context - Context string for logging
   * @returns Operation result
   */
  executeWithRetry<T>(operation: () => Promise<T>, context: string): Promise<T>

  /**
   * Check if an error is retryable
   * @param error - Error to check
   * @returns True if the error is retryable
   */
  isRetryable(error: unknown): boolean

  /**
   * Get current retry configuration
   */
  getRetryConfig(): RetryConfig

  /**
   * Update retry configuration
   * @param config - Partial configuration to merge
   */
  setRetryConfig(config: Partial<RetryConfig>): void
}
