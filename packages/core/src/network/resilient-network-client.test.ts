import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { ResilientNetworkClient } from './resilient-network-client'
import type { NetworkError } from '../interfaces/network-client'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

// Skip tests as we're not using the ResilientNetworkClient right now
describe.skip('ResilientNetworkClient', () => {
  let client: ResilientNetworkClient
  let fetchCallCount: number
  let originalFetch: typeof global.fetch

  beforeEach(() => {
    client = new ResilientNetworkClient()
    fetchCallCount = 0
    originalFetch = global.fetch

    // Register required event types
    const eventBus = EventBus.getInstance()
    eventBus.registerEvent(EventTypes.SYSTEM_INFO)
    eventBus.registerEvent(EventTypes.SYSTEM_ERROR)
  })

  afterEach(() => {
    global.fetch = originalFetch
    EventBus.getInstance().reset()
  })

  describe('request', () => {
    it('should make successful request', async () => {
      const mockResponse = { data: 'test' }
      global.fetch = async () => {
        fetchCallCount++
        return {
          ok: true,
          status: 200,
          json: async () => mockResponse,
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response
      }

      const response = await client.request('https://api.example.com/test')

      assert.equal(fetchCallCount, 1)
      assert.deepEqual(response.data, mockResponse)
      assert.equal(response.status, 200)
    })

    it('should retry on network error', async () => {
      let attempts = 0
      global.fetch = async () => {
        attempts++
        if (attempts < 3) {
          const error = new Error('Network error') as NetworkError
          error.code = 'ECONNRESET'
          throw error
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
          headers: new Headers(),
        } as Response
      }

      const response = await client.request('https://api.example.com/test')

      assert.equal(attempts, 3)
      assert.deepEqual(response.data, { success: true })
    })

    it('should throw after max retries', async () => {
      global.fetch = async () => {
        const error = new Error('Network error') as NetworkError
        error.code = 'ECONNRESET'
        throw error
      }

      await assert.rejects(
        () => client.request('https://api.example.com/test'),
        {
          message: 'Network error',
        },
      )
    })

    it('should not retry on non-retryable error', async () => {
      let attempts = 0
      global.fetch = async () => {
        attempts++
        const error = new Error('Bad request') as NetworkError
        error.status = 400
        throw error
      }

      await assert.rejects(
        () => client.request('https://api.example.com/test'),
        {
          message: 'Bad request',
        },
      )

      assert.equal(attempts, 1)
    })
  })

  describe('isRetryable', () => {
    it('should identify network errors as retryable', () => {
      const testCases = [
        { code: 'ECONNRESET', expected: true },
        { code: 'ETIMEDOUT', expected: true },
        { code: 'ENOTFOUND', expected: true },
        { code: 'ECONNREFUSED', expected: true },
        { code: 'EHOSTUNREACH', expected: true },
        { code: 'UNKNOWN', expected: false },
      ]

      for (const { code, expected } of testCases) {
        const error = new Error('Network error') as NetworkError
        error.code = code
        assert.equal(client.isRetryable(error), expected, `Code ${code} should be ${expected ? 'retryable' : 'not retryable'}`)
      }
    })

    it('should identify HTTP status codes as retryable', () => {
      const testCases = [
        { status: 429, expected: true }, // Rate limited
        { status: 502, expected: true }, // Bad gateway
        { status: 503, expected: true }, // Service unavailable
        { status: 504, expected: true }, // Gateway timeout
        { status: 400, expected: false }, // Bad request
        { status: 401, expected: false }, // Unauthorized
        { status: 404, expected: false },  // Not found
      ]

      for (const { status, expected } of testCases) {
        const error = new Error('HTTP error') as NetworkError
        error.status = status
        assert.equal(client.isRetryable(error), expected, `Status ${status} should be ${expected ? 'retryable' : 'not retryable'}`)
      }
    })

    it('should identify exchange-specific errors as retryable', () => {
      const testCases = [
        { message: 'request timestamp expired', expected: true },
        { message: 'nonce too small', expected: true },
        { message: 'rate limit exceeded', expected: true },
        { message: 'invalid signature', expected: false },
      ]

      for (const { message, expected } of testCases) {
        const error = new Error(message)
        assert.equal(client.isRetryable(error), expected, `Message "${message}" should be ${expected ? 'retryable' : 'not retryable'}`)
      }
    })

    it('should return false for non-Error objects', () => {
      assert.equal(client.isRetryable('not an error'), false)
      assert.equal(client.isRetryable(null), false)
      assert.equal(client.isRetryable(undefined), false)
      assert.equal(client.isRetryable({}), false)
    })
  })

  describe('executeWithRetry', () => {
    it('should execute operation successfully on first try', async () => {
      let attempts = 0
      const result = await client.executeWithRetry(
        async () => {
          attempts++
          return 'success'
        },
        'test operation',
      )

      assert.equal(result, 'success')
      assert.equal(attempts, 1)
    })

    it('should retry and eventually succeed', async () => {
      let attempts = 0
      const result = await client.executeWithRetry(
        async () => {
          attempts++
          if (attempts < 3) {
            const error = new Error('Temporary failure') as NetworkError
            error.code = 'ETIMEDOUT'
            throw error
          }
          return 'success after retries'
        },
        'test operation',
      )

      assert.equal(result, 'success after retries')
      assert.equal(attempts, 3)
    })

    it('should apply exponential backoff', async () => {
      const delays: number[] = []

      // Override sleep to track delays
      const originalSetTimeout = global.setTimeout
      global.setTimeout = ((fn: Function, delay: number) => {
        delays.push(delay)
        fn()
        return 0
      }) as any

      try {
        await client.executeWithRetry(
          async () => {
            if (delays.length < 3) {
              const error = new Error('Network error') as NetworkError
              error.code = 'ECONNRESET'
              throw error
            }
            return 'done'
          },
          'test operation',
        )

        // Check that delays follow exponential pattern with jitter
        assert.equal(delays.length, 3)
        assert.ok(delays[0]! >= 900 && delays[0]! <= 1100) // ~1000ms ±10%
        assert.ok(delays[1]! >= 1800 && delays[1]! <= 2200) // ~2000ms ±10%
        assert.ok(delays[2]! >= 3600 && delays[2]! <= 4400) // ~4000ms ±10%
      } finally {
        global.setTimeout = originalSetTimeout
      }
    })
  })

  describe('configuration', () => {
    it('should use custom retry configuration', async () => {
      client = new ResilientNetworkClient({
        maxRetries: 2,
        initialDelay: 100,
        maxDelay: 500,
        backoffMultiplier: 1.5,
        jitter: 0,
      })

      let attempts = 0
      global.fetch = async () => {
        attempts++
        const error = new Error('Network error') as NetworkError
        error.code = 'ECONNRESET'
        throw error
      }

      await assert.rejects(
        () => client.request('https://api.example.com/test'),
      )

      assert.equal(attempts, 3) // initial + 2 retries
    })

    it('should get and set retry configuration', () => {
      const initialConfig = client.getRetryConfig()
      assert.equal(initialConfig.maxRetries, 5)
      assert.equal(initialConfig.initialDelay, 1000)

      client.setRetryConfig({
        maxRetries: 10,
        initialDelay: 500,
      })

      const updatedConfig = client.getRetryConfig()
      assert.equal(updatedConfig.maxRetries, 10)
      assert.equal(updatedConfig.initialDelay, 500)
      assert.equal(updatedConfig.backoffMultiplier, initialConfig.backoffMultiplier)
    })
  })
})
