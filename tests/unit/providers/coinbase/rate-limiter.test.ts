import { ok, rejects, strictEqual } from 'node:assert'
import { afterEach, beforeEach, describe, it } from 'node:test'
import { RateLimiter } from '../../../../src/providers/coinbase'
import { forceCleanupAsyncHandles } from '../../../helpers/test-cleanup'

describe('RateLimiter', () => {
  let rateLimiter: RateLimiter

  beforeEach(() => {
    rateLimiter = new RateLimiter({
      maxRequestsPerSecond: 2,
      maxRetries: 3,
      initialRetryDelayMs: 100,
      maxRetryDelayMs: 1000,
      backoffMultiplier: 2
    })
  })

  afterEach(() => {
    forceCleanupAsyncHandles()
  })

  it('should limit requests per second', async () => {
    const startTime = Date.now()
    const results: number[] = []

    // Try to make 3 requests quickly (limit is 2 per second)
    for (let i = 0; i < 3; i++) {
      await rateLimiter.execute(async () => {
        results.push(Date.now() - startTime)
        return i
      })
    }

    // First 2 should be immediate, 3rd should wait ~1 second
    ok(results[0]! < 100, 'First request should be immediate')
    ok(results[1]! < 100, 'Second request should be immediate')
    ok(results[2]! >= 900, 'Third request should wait for rate limit')
  })

  it('should retry with exponential backoff on rate limit errors', async () => {
    let attempts = 0
    const delays: number[] = []
    let lastTime = Date.now()

    const result = await rateLimiter.execute(async () => {
      attempts++
      const now = Date.now()
      if (attempts > 1) {
        delays.push(now - lastTime)
      }
      lastTime = now

      if (attempts < 3) {
        const error: any = new Error('Rate limit exceeded')
        error.response = { status: 429 }
        throw error
      }

      return 'success'
    })

    strictEqual(result, 'success')
    strictEqual(attempts, 3)
    strictEqual(delays.length, 2)

    // Check exponential backoff (with jitter tolerance)
    ok(delays[0]! >= 90 && delays[0]! <= 110, 'First retry ~100ms')
    ok(delays[1]! >= 180 && delays[1]! <= 220, 'Second retry ~200ms')
  })

  it('should throw non-rate-limit errors immediately', async () => {
    let attempts = 0

    await rejects(async () => {
      await rateLimiter.execute(async () => {
        attempts++
        throw new Error('Some other error')
      })
    }, /Some other error/)

    strictEqual(attempts, 1, 'Should not retry non-rate-limit errors')
  })

  it('should throw after max retries', async () => {
    let attempts = 0

    await rejects(async () => {
      await rateLimiter.execute(async () => {
        attempts++
        const error: any = new Error('Rate limit exceeded')
        error.response = { status: 429 }
        throw error
      })
    }, /Rate limit retry exhausted after 3 attempts/)

    strictEqual(attempts, 3)
  })

  it('should detect various rate limit error formats', async () => {
    const rateLimitErrors = [
      { response: { status: 429 } },
      { message: 'Rate limit exceeded' },
      { message: 'You have hit the rate limit' },
      { code: 'RATE_LIMIT_EXCEEDED' },
      { response: { data: { message: 'API rate limit exceeded' } } }
    ]

    for (const errorFormat of rateLimitErrors) {
      let attempts = 0

      await rateLimiter.execute(async () => {
        attempts++
        if (attempts === 1) {
          const error: any = new Error('Test error')
          Object.assign(error, errorFormat)
          throw error
        }
        return 'success'
      })

      strictEqual(
        attempts,
        2,
        `Should retry for error format: ${JSON.stringify(errorFormat)}`
      )
    }
  })

  it('should add jitter to prevent thundering herd', async () => {
    const delays: number[] = []

    // Run multiple instances to check jitter
    for (let i = 0; i < 5; i++) {
      const limiter = new RateLimiter({
        maxRequestsPerSecond: 1,
        maxRetries: 2,
        initialRetryDelayMs: 100,
        backoffMultiplier: 2
      })

      let attemptTime = 0
      await limiter.execute(async () => {
        if (attemptTime === 0) {
          attemptTime = Date.now()
          const error: any = new Error('Rate limit')
          error.response = { status: 429 }
          throw error
        }
        delays.push(Date.now() - attemptTime)
        return 'success'
      })
    }

    // Check that delays have some variance due to jitter
    const uniqueDelays = new Set(delays)
    ok(uniqueDelays.size > 1, 'Delays should vary due to jitter')
  })
})
