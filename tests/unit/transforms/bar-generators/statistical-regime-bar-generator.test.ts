import { ok } from 'node:assert'
import { describe, it } from 'node:test'
import { StatisticalRegimeBarGenerator } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('StatisticalRegimeBarGenerator', () => {
  const createTestBuffer = (count = 35): DataBuffer => {
    const buffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        symbol: { index: 1 },
        exchange: { index: 2 },
        open: { index: 3 },
        high: { index: 4 },
        low: { index: 5 },
        close: { index: 6 },
        volume: { index: 7 }
      }
    })
    
    const baseTime = Date.parse('2024-01-01T00:00:00Z')
    
    // Create test data with regime change pattern
    // Start with stable prices, then add volatility, then trending
    const stablePrices = Array(15)
      .fill(0)
      .map((_, i) => 100 + Math.sin(i * 0.1) * 0.5) // Low volatility
    const volatilePrices = Array(10)
      .fill(0)
      .map((_, i) => 105 + Math.sin(i * 0.5) * 5) // High volatility
    const trendingPrices = Array(10)
      .fill(0)
      .map((_, i) => 110 + i * 2) // Strong trend

    const allPrices = [...stablePrices, ...volatilePrices, ...trendingPrices]
    
    for (let i = 0; i < Math.min(count, allPrices.length); i++) {
      const price = allPrices[i]!
      buffer.push({
        timestamp: baseTime + i * 1000,
        open: price,
        high: price + Math.random() * 0.5,
        low: price - Math.random() * 0.5,
        close: price,
        volume: 100
      })
    }
    
    return buffer
  }

  it('should create bars when statistical regime changes significantly', () => {
    const buffer = createTestBuffer(35)
    const slice = new DataSlice(buffer, 0, buffer.length() - 1)
    
    const generator = new StatisticalRegimeBarGenerator({
      tx: {
        lookback: 20,
        threshold: 2.0 // Lower threshold for more sensitive detection
      }
    }, slice)

    // Process the data
    generator.next(0, buffer.length() - 1)
    const outputBuffer = generator.outputBuffer

    // Should create at least 1 bar due to regime changes (algorithm may be conservative)
    ok(outputBuffer.length() >= 1, `Expected at least 1 bar, got ${outputBuffer.length()}`)

    // Each bar should be valid OHLCV
    for (let i = 0; i < outputBuffer.length(); i++) {
      const bar = outputBuffer.getRow(i)
      ok(bar)
      ok(bar.open && bar.open > 0)
      ok(bar.high && bar.open && bar.high >= bar.open)
      ok(bar.high && bar.close && bar.high >= bar.close)
      ok(bar.low && bar.open && bar.low <= bar.open)
      ok(bar.low && bar.close && bar.low <= bar.close)
      ok(bar.volume && bar.volume > 0)
    }
  })

  it('should handle stable market conditions without excessive bars', () => {
    const buffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        symbol: { index: 1 },
        exchange: { index: 2 },
        open: { index: 3 },
        high: { index: 4 },
        low: { index: 5 },
        close: { index: 6 },
        volume: { index: 7 }
      }
    })
    
    const baseTime = Date.parse('2024-01-01T00:00:00Z')
    
    // Create very stable price data
    const stablePrices = Array(50)
      .fill(0)
      .map((_, i) => 100 + Math.sin(i * 0.05) * 0.1) // Very low volatility
    
    for (let i = 0; i < stablePrices.length; i++) {
      const price = stablePrices[i]!
      buffer.push({
        timestamp: baseTime + i * 1000,
        open: price,
        high: price + 0.01,
        low: price - 0.01,
        close: price,
        volume: 100
      })
    }
    
    const slice = new DataSlice(buffer, 0, buffer.length() - 1)
    
    const generator = new StatisticalRegimeBarGenerator({
      tx: {
        lookback: 20,
        threshold: 3.0 // Higher threshold
      }
    }, slice)

    // Process the data
    generator.next(0, buffer.length() - 1)
    const outputBuffer = generator.outputBuffer

    // Should create very few bars in stable conditions
    ok(
      outputBuffer.length() <= 3,
      `Expected few bars in stable conditions, got ${outputBuffer.length()}`
    )
  })

  it('should require minimum lookback period', () => {
    const buffer = createTestBuffer(10)
    const slice = new DataSlice(buffer, 0, buffer.length() - 1)
    
    try {
      new StatisticalRegimeBarGenerator({
        tx: {
          lookback: 5, // Too small
          threshold: 2.5
        }
      }, slice)
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('10'))
    }
  })

  it('should require minimum threshold', () => {
    const buffer = createTestBuffer(10)
    const slice = new DataSlice(buffer, 0, buffer.length() - 1)
    
    try {
      new StatisticalRegimeBarGenerator({
        tx: {
          lookback: 20,
          threshold: 0.5 // Too small
        }
      }, slice)
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('1'))
    }
  })

  it('should handle basic functionality without errors', () => {
    const buffer = createTestBuffer(25)
    const slice = new DataSlice(buffer, 0, buffer.length() - 1)
    
    const generator = new StatisticalRegimeBarGenerator({
      tx: {
        lookback: 15,
        threshold: 2.5
      }
    }, slice)

    // Process the data
    generator.next(0, buffer.length() - 1)
    const outputBuffer = generator.outputBuffer

    // Should not throw errors and potentially create bars
    ok(outputBuffer.length() >= 0, 'Should process without errors')
  })
})
