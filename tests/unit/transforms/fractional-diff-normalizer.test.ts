import { describe, it } from 'node:test'
import { deepStrictEqual, doesNotThrow, ok, strictEqual, throws } from 'node:assert/strict'
import type { OhlcvDto } from '../../../src/models'
import { FractionalDiffNormalizer } from '../../../src/transforms'

describe('FractionalDiffNormalizer', () => {
  const createSampleData = (): OhlcvDto[] => [
    {
      timestamp: Date.parse('2024-01-01T00:00:00Z'),
      symbol: 'BTC-USD',
      exchange: 'test',
      open: 100,
      high: 110,
      low: 95,
      close: 105,
      volume: 1000,
    },
    {
      timestamp: Date.parse('2024-01-01T01:00:00Z'),
      symbol: 'BTC-USD',
      exchange: 'test',
      open: 105,
      high: 115,
      low: 100,
      close: 110,
      volume: 1200,
    },
    {
      timestamp: Date.parse('2024-01-01T02:00:00Z'),
      symbol: 'BTC-USD',
      exchange: 'test',
      open: 110,
      high: 120,
      low: 105,
      close: 115,
      volume: 1500,
    },
    {
      timestamp: Date.parse('2024-01-01T03:00:00Z'),
      symbol: 'BTC-USD',
      exchange: 'test',
      open: 115,
      high: 125,
      low: 110,
      close: 120,
      volume: 1800,
    },
  ]

  describe('validation', () => {
    it('should require d parameter', () => {
      throws(
        () => {
          const transform = new FractionalDiffNormalizer({} as any)
          transform.validate()
        },
        { message: /Differencing parameter d is required/ }
      )
    })

    it('should reject d values outside valid range', () => {
      throws(
        () => new FractionalDiffNormalizer({ d: -0.5 }),
        { message: /Differencing parameter d must be between 0 and 2/ }
      )

      throws(
        () => new FractionalDiffNormalizer({ d: 2.5 }),
        { message: /Differencing parameter d must be between 0 and 2/ }
      )
    })

    it('should accept valid d values', () => {
      doesNotThrow(() => new FractionalDiffNormalizer({ d: 0.5 }))
      doesNotThrow(() => new FractionalDiffNormalizer({ d: 0 }))
      doesNotThrow(() => new FractionalDiffNormalizer({ d: 1 }))
      doesNotThrow(() => new FractionalDiffNormalizer({ d: 2 }))
    })
  })

  describe('weight calculation', () => {
    it('should calculate correct weights for d=0.5', () => {
      const transform = new FractionalDiffNormalizer({ d: 0.5 })
      
      // Access weights through transform result
      const weights = (transform as any).weights
      
      // First few weights for d=0.5
      // w_0 = 1
      // w_1 = -w_0 * (d - 0) / 1 = -1 * 0.5 / 1 = -0.5
      // w_2 = -w_1 * (d - 1) / 2 = -(-0.5) * (0.5 - 1) / 2 = 0.5 * (-0.5) / 2 = -0.125
      // w_3 = -w_2 * (d - 2) / 3 = -(-0.125) * (0.5 - 2) / 3 = 0.125 * (-1.5) / 3 = -0.0625
      strictEqual(weights[0], 1)
      strictEqual(weights[1], -0.5)
      strictEqual(weights[2], -0.125)
      strictEqual(weights[3], -0.0625)
    })

    it('should respect maxWeights parameter', () => {
      const transform = new FractionalDiffNormalizer({ 
        d: 0.5, 
        maxWeights: 5 
      })
      
      const weights = (transform as any).weights
      ok(weights.length <= 5)
    })

    it('should respect minWeight threshold', () => {
      const transform = new FractionalDiffNormalizer({ 
        d: 0.5, 
        minWeight: 0.1 
      })
      
      const weights = (transform as any).weights
      // All weights should have magnitude >= minWeight (except possibly the last)
      for (let i = 0; i < weights.length - 1; i++) {
        ok(Math.abs(weights[i]) >= 0.1)
      }
    })
  })

  describe('transformation', () => {
    it('should apply fractional differentiation to all columns by default', async () => {
      const transform = new FractionalDiffNormalizer({ d: 0.5 })
      const data = createSampleData()
      
      const asyncData = async function* () {
        for (const item of data) {
          yield item
        }
      }()
      
      const result = await transform.apply(asyncData)
      const output: OhlcvDto[] = []
      
      let item = await result.data.next()
      while (!item.done) {
        output.push(item.value)
        item = await result.data.next()
      }
      
      strictEqual(output.length, data.length)
      
      // First data point uses only w_0 = 1
      strictEqual(output[0]!.open, 100)
      strictEqual(output[0]!.close, 105)
      
      // Second data point: w_0 * current + w_1 * previous
      // For open: 1 * 105 + (-0.5) * 100 = 55
      strictEqual(output[1]!.open, 55)
      // For close: 1 * 110 + (-0.5) * 105 = 57.5
      strictEqual(output[1]!.close, 57.5)
    })

    it('should handle in/out parameters correctly', async () => {
      const transform = new FractionalDiffNormalizer({ 
        d: 0.5,
        in: ['close', 'volume'],
        out: ['close_fd', 'volume_fd']
      })
      
      const data = createSampleData()
      const asyncData = async function* () {
        for (const item of data) {
          yield item
        }
      }()
      
      const result = await transform.apply(asyncData)
      const output: OhlcvDto[] = []
      
      let item = await result.data.next()
      while (!item.done) {
        output.push(item.value)
        item = await result.data.next()
      }
      
      // Check that only specified columns are transformed
      ok('close_fd' in output[0]!)
      ok('volume_fd' in output[0]!)
      
      // Original columns should remain unchanged
      strictEqual(output[0]!.open, 100)
      strictEqual(output[0]!.high, 110)
      strictEqual(output[0]!.low, 95)
    })

    it('should handle multiple symbols independently', async () => {
      const transform = new FractionalDiffNormalizer({ d: 0.5 })
      
      const multiSymbolData: OhlcvDto[] = [
        {
          timestamp: Date.parse('2024-01-01T00:00:00Z'),
          symbol: 'BTC-USD',
          exchange: 'test',
          open: 100,
          high: 110,
          low: 95,
          close: 105,
          volume: 1000,
        },
        {
          timestamp: Date.parse('2024-01-01T00:00:00Z'),
          symbol: 'ETH-USD',
          exchange: 'test',
          open: 50,
          high: 55,
          low: 45,
          close: 52,
          volume: 500,
        },
        {
          timestamp: Date.parse('2024-01-01T01:00:00Z'),
          symbol: 'BTC-USD',
          exchange: 'test',
          open: 105,
          high: 115,
          low: 100,
          close: 110,
          volume: 1200,
        },
        {
          timestamp: Date.parse('2024-01-01T01:00:00Z'),
          symbol: 'ETH-USD',
          exchange: 'test',
          open: 52,
          high: 57,
          low: 50,
          close: 55,
          volume: 600,
        },
      ]
      
      const asyncData = async function* () {
        for (const item of multiSymbolData) {
          yield item
        }
      }()
      
      const result = await transform.apply(asyncData)
      const output: OhlcvDto[] = []
      
      let item = await result.data.next()
      while (!item.done) {
        output.push(item.value)
        item = await result.data.next()
      }
      
      // BTC second data point
      const btcSecond = output[2]!
      strictEqual(btcSecond.symbol, 'BTC-USD')
      strictEqual(btcSecond.open, 55) // 1 * 105 + (-0.5) * 100
      
      // ETH second data point
      const ethSecond = output[3]!
      strictEqual(ethSecond.symbol, 'ETH-USD')
      strictEqual(ethSecond.open, 27) // 1 * 52 + (-0.5) * 50
    })

    it('should handle d=0 (no differentiation)', async () => {
      const transform = new FractionalDiffNormalizer({ d: 0 })
      const data = createSampleData()
      
      const asyncData = async function* () {
        for (const item of data) {
          yield item
        }
      }()
      
      const result = await transform.apply(asyncData)
      const output: OhlcvDto[] = []
      
      let item = await result.data.next()
      while (!item.done) {
        output.push(item.value)
        item = await result.data.next()
      }
      
      // With d=0, output should be identical to input
      for (let i = 0; i < data.length; i++) {
        strictEqual(output[i]!.open, data[i]!.open)
        strictEqual(output[i]!.close, data[i]!.close)
      }
    })

    it('should handle d=1 (standard differentiation)', async () => {
      const transform = new FractionalDiffNormalizer({ d: 1 })
      const data = createSampleData()
      
      const asyncData = async function* () {
        for (const item of data) {
          yield item
        }
      }()
      
      const result = await transform.apply(asyncData)
      const output: OhlcvDto[] = []
      
      let item = await result.data.next()
      while (!item.done) {
        output.push(item.value)
        item = await result.data.next()
      }
      
      // With d=1, this is standard first differencing
      // w_0 = 1, w_1 = -1, all other weights = 0
      strictEqual(output[0]!.close, 105) // First value unchanged
      strictEqual(output[1]!.close, 5) // 110 - 105 = 5
      strictEqual(output[2]!.close, 5) // 115 - 110 = 5
      strictEqual(output[3]!.close, 5) // 120 - 115 = 5
    })
  })

  describe('getters', () => {
    it('should return correct output fields', () => {
      const transform1 = new FractionalDiffNormalizer({ 
        d: 0.5,
        in: ['close'],
        out: ['close_fd']
      })
      deepStrictEqual(transform1.getOutputFields(), ['close_fd'])
      
      const transform2 = new FractionalDiffNormalizer({ d: 0.5 })
      deepStrictEqual(transform2.getOutputFields(), ['open', 'high', 'low', 'close', 'volume'])
    })

    it('should return correct required fields', () => {
      const transform1 = new FractionalDiffNormalizer({ 
        d: 0.5,
        in: ['close', 'volume'],
        out: ['close_fd', 'volume_fd']
      })
      deepStrictEqual(transform1.getRequiredFields(), ['close', 'volume'])
      
      const transform2 = new FractionalDiffNormalizer({ d: 0.5 })
      deepStrictEqual(transform2.getRequiredFields(), ['open', 'high', 'low', 'close', 'volume'])
    })

    it('should always be ready', () => {
      const transform = new FractionalDiffNormalizer({ d: 0.5 })
      strictEqual(transform.isReady(), true)
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new FractionalDiffNormalizer({ d: 0.5 })
      const updated = original.withParams({ d: 0.7 })
      
      ok(original !== updated)
      strictEqual((original as any).params.d, 0.5)
      strictEqual((updated as any).params.d, 0.7)
    })
  })
})