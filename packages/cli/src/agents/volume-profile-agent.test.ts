import { describe, it, beforeEach } from 'node:test'
import assert from 'node:assert'
import { VolumeProfileAgent } from './volume-profile-agent'
import type { MarketContext } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('VolumeProfileAgent', () => {
  let agent: VolumeProfileAgent
  let mockContext: MarketContext

  beforeEach(async () => {
    agent = new VolumeProfileAgent({
      id: 'test-volume-profile-agent',
      name: 'Test Volume Profile Agent',
      version: '1.0.0',
      description: 'Test volume profile agent',
      type: 'volume',
      defaultConfig: {}
    })

    // Create mock candles with varying volume patterns
    const basePrice = 100
    const mockCandles = Array.from({ length: 50 }, (_, i) => {
      const baseVolume = 1000
      // Create volume spikes every 10 candles
      const volumeMultiplier = i % 10 === 0 ? 3 : 1
      
      return {
        timestamp: toEpochDate(Date.now() - (50 - i) * 60000), // 1 minute intervals
        open: basePrice + i * 0.5,
        high: basePrice + i * 0.5 + 1,
        low: basePrice + i * 0.5 - 0.5,
        close: basePrice + i * 0.5 + 0.2,
        volume: baseVolume * volumeMultiplier + Math.random() * 200
      }
    })

    mockContext = {
      symbol: 'TEST-USD' as any,
      currentPrice: mockCandles[mockCandles.length - 1]!.close,
      candles: mockCandles
    }
  })

  it('should initialize successfully', async () => {
    await agent.initialize()
    assert.ok(agent)
  })

  it('should return hold signal with insufficient data', async () => {
    await agent.initialize()
    
    const shortContext = {
      ...mockContext,
      candles: mockContext.candles.slice(0, 5) // Too few candles
    }

    const signal = await agent.analyze(shortContext)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence < 0.5)
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should detect volume spikes', async () => {
    await agent.initialize()
    
    // Create context with a significant volume spike
    const spikeCandles = [...mockContext.candles]
    const lastCandle = spikeCandles[spikeCandles.length - 1]!
    
    // Create a bullish volume spike
    spikeCandles[spikeCandles.length - 1] = {
      ...lastCandle,
      volume: 5000, // 5x normal volume
      open: lastCandle.close - 2,
      close: lastCandle.close,
      high: lastCandle.close + 0.5,
      low: lastCandle.close - 2.5
    }
    
    const spikeContext = {
      ...mockContext,
      candles: spikeCandles
    }
    
    const signal = await agent.analyze(spikeContext)
    assert.ok(['buy', 'sell'].includes(signal.action))
    assert.ok(signal.confidence > 0.6)
    assert.ok(signal.reason.toLowerCase().includes('volume spike'))
  })

  it('should identify support and resistance levels', async () => {
    await agent.initialize()
    
    // Create candles with clear volume concentration at certain price levels
    const supportResistanceCandles = Array.from({ length: 100 }, (_, i) => {
      const price = 100 + Math.sin(i / 10) * 10 // Oscillating price
      const volume = Math.abs(price - 100) < 2 ? 3000 : 1000 // High volume near 100
      
      return {
        timestamp: toEpochDate(Date.now() - (100 - i) * 60000),
        open: price,
        high: price + 0.5,
        low: price - 0.5,
        close: price + 0.1,
        volume
      }
    })
    
    const srContext = {
      ...mockContext,
      candles: supportResistanceCandles,
      currentPrice: 99.5 // Just below high volume area
    }
    
    const signal = await agent.analyze(srContext)
    assert.ok(signal)
    assert.ok(/support|resistance|volume/i.test(signal.reason))
  })

  it('should generate valid signals with sufficient data', async () => {
    await agent.initialize()
    
    const signal = await agent.analyze(mockContext)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(signal.confidence >= 0)
    assert.ok(signal.confidence <= 1)
    assert.ok(signal.reason)
    assert.ok(signal.timestamp)
  })

  it('should handle calculation errors gracefully', async () => {
    await agent.initialize()
    
    // Create context with invalid candles
    const invalidContext = {
      ...mockContext,
      candles: mockContext.candles.map(c => ({
        ...c,
        volume: NaN // Invalid volume data
      }))
    }

    const signal = await agent.analyze(invalidContext)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.toLowerCase().includes('error'))
  })

  it('should build volume profile correctly', async () => {
    await agent.initialize()
    
    // Run analysis to populate volume profile
    await agent.analyze(mockContext)
    
    const lastAnalysis = agent.getLastAnalysis()
    assert.ok(lastAnalysis)
    assert.ok(Array.isArray(lastAnalysis!.volumeProfile))
    assert.ok(lastAnalysis!.volumeProfile.length > 0)
    
    // Check for POC (Point of Control)
    const poc = lastAnalysis!.volumeProfile.find(level => level.type === 'poc')
    assert.ok(poc)
    assert.strictEqual(poc!.percentage, 1) // POC should have 100% relative volume
  })

  it('should track analysis history', async () => {
    await agent.initialize()
    
    // Run analysis multiple times
    await agent.analyze(mockContext)
    await agent.analyze(mockContext)
    
    const lastAnalysis = agent.getLastAnalysis()
    assert.ok(lastAnalysis)
    assert.strictEqual(typeof lastAnalysis!.averageVolume, 'number')
    assert.ok(Array.isArray(lastAnalysis!.supportResistance))
  })

  it('should reset state correctly', async () => {
    await agent.initialize()
    
    // Generate some analysis to populate internal state
    await agent.analyze(mockContext)
    assert.ok(agent.getLastAnalysis())
    
    // Reset
    await agent.reset()
    
    // Should be able to run analysis again without issues
    const signal = await agent.analyze(mockContext)
    assert.ok(signal)
  })

  it('should validate custom configuration', async () => {
    const customAgent = new VolumeProfileAgent({
      id: 'custom-volume-profile-agent',
      name: 'Custom Volume Profile Agent',
      version: '1.0.0',
      description: 'Custom test volume profile agent',
      type: 'volume',
      defaultConfig: {}
    }, undefined, {
      volumeThreshold: 1.5,
      priceResolution: 30,
      lookbackPeriod: 50,
      supportResistanceThreshold: 0.6,
      spikeDetectionPeriod: 15
    })

    await customAgent.initialize()
    const signal = await customAgent.analyze(mockContext)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should provide confidence levels within valid range', async () => {
    await agent.initialize()
    
    // Test multiple scenarios
    for (let i = 0; i < 5; i++) {
      const signal = await agent.analyze(mockContext)
      assert.ok(signal.confidence >= 0)
      assert.ok(signal.confidence <= 1)
    }
  })
})