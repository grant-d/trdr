import { describe, it, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { BaseAgent } from './base-agent'
import type { AgentMetadata, AgentSignal, MarketContext } from './types'
import { toStockSymbol, epochDateNow } from '@trdr/shared'

// Test implementation of BaseAgent
class TestAgent extends BaseAgent {
  private analysisCount = 0
  
  protected async onInitialize(): Promise<void> {
    // Test initialization
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    this.analysisCount++
    
    // Add small delay to ensure measurable execution time
    await new Promise(resolve => setTimeout(resolve, 1))
    
    // Simple test logic
    const lastCandle = context.candles[context.candles.length - 1]!
    const action = lastCandle.close > lastCandle.open ? 'buy' : 'sell'
    const confidence = 0.75
    
    return this.createSignal(
      action,
      confidence,
      'Test signal based on last candle',
      'Detailed test analysis'
    )
  }
  
  getAnalysisCount(): number {
    return this.analysisCount
  }
}

describe('BaseAgent', () => {
  let agent: TestAgent
  const metadata: AgentMetadata = {
    id: 'test-agent',
    name: 'Test Agent',
    version: '1.0.0',
    description: 'Test agent for unit tests',
    type: 'custom',
    requiredIndicators: ['RSI', 'MACD'],
    defaultConfig: { threshold: 0.5 }
  }
  
  const createContext = (): MarketContext => ({
    symbol: toStockSymbol('BTC-USD'),
    currentPrice: 50000,
    candles: [
      {
        timestamp: (epochDateNow() - 60000) as any,
        open: 49900,
        high: 50100,
        low: 49800,
        close: 50000,
        volume: 100
      },
      {
        timestamp: epochDateNow(),
        open: 50000,
        high: 50200,
        low: 49950,
        close: 50100,
        volume: 150
      }
    ],
    indicators: {
      RSI: { value: 55, timestamp: epochDateNow() },
      MACD: { value: 100, timestamp: epochDateNow() }
    }
  })
  
  beforeEach(() => {
    agent = new TestAgent(metadata)
  })
  
  describe('initialization', () => {
    it('should initialize with default config', async () => {
      await agent.initialize()
      
      const signal = await agent.analyze(createContext())
      assert.ok(signal)
      assert.equal(agent.getAnalysisCount(), 1)
    })
    
    it('should merge custom config with defaults', async () => {
      await agent.initialize({ threshold: 0.7, newParam: 'test' })
      
      // Agent should be initialized
      const signal = await agent.analyze(createContext())
      assert.ok(signal)
    })
    
    it('should prevent double initialization', async () => {
      await agent.initialize()
      
      // Second initialization should be ignored
      await agent.initialize({ different: 'config' })
      
      // Should still work
      const signal = await agent.analyze(createContext())
      assert.ok(signal)
    })
  })
  
  describe('analyze', () => {
    it('should throw if not initialized', async () => {
      await assert.rejects(
        agent.analyze(createContext()),
        /not initialized/
      )
    })
    
    it('should generate signal based on context', async () => {
      await agent.initialize()
      
      const context = createContext()
      const signal = await agent.analyze(context)
      
      assert.equal(signal.action, 'buy') // Last candle is green
      assert.equal(signal.confidence, 0.75)
      assert.ok(signal.reason)
      assert.ok(signal.timestamp)
    })
    
    it('should validate context', async () => {
      await agent.initialize()
      
      // Missing symbol
      await assert.rejects(
        agent.analyze({ ...createContext(), symbol: null as any }),
        /Missing symbol/
      )
      
      // Invalid price
      await assert.rejects(
        agent.analyze({ ...createContext(), currentPrice: 0 }),
        /Invalid current price/
      )
      
      // No candles
      await assert.rejects(
        agent.analyze({ ...createContext(), candles: [] }),
        /No candles provided/
      )
      
      // Missing required indicators
      const contextNoIndicators = { ...createContext() }
      delete contextNoIndicators.indicators
      await assert.rejects(
        agent.analyze(contextNoIndicators),
        /Required indicator RSI not provided/
      )
    })
    
    it('should track execution time', async () => {
      await agent.initialize()
      
      // Analyze multiple times
      for (let i = 0; i < 5; i++) {
        await agent.analyze(createContext())
      }
      
      const performance = agent.getPerformance()
      assert.equal(performance.totalSignals, 5)
      assert.ok(performance.avgExecutionTime > 0)
    })
  })
  
  describe('updateOnTrade', () => {
    it('should update performance on profitable trade', async () => {
      await agent.initialize()
      
      const signal = await agent.analyze(createContext())
      
      await agent.updateOnTrade({
        id: 'trade-1',
        timestamp: epochDateNow(),
        side: 'buy',
        price: 50000,
        size: 0.1,
        fee: 10,
        pnl: 100
      }, signal)
      
      const performance = agent.getPerformance()
      assert.equal(performance.profitableSignals, 1)
    })
    
    it('should track returns and calculate metrics', async () => {
      await agent.initialize()
      
      // Generate signals and trades
      for (let i = 0; i < 10; i++) {
        const signal = await agent.analyze(createContext())
        
        await agent.updateOnTrade({
          id: `trade-${i}`,
          timestamp: epochDateNow(),
          side: 'buy',
          price: 50000,
          size: 0.1,
          fee: 10,
          pnl: i % 2 === 0 ? 100 : -50 // Alternate profit/loss
        }, signal)
      }
      
      const performance = agent.getPerformance()
      assert.equal(performance.totalSignals, 10)
      assert.equal(performance.profitableSignals, 5)
      assert.equal(performance.winRate, 0.5)
      assert.ok(performance.averageReturn !== 0)
      assert.ok(performance.sharpeRatio !== 0)
      assert.ok(performance.maxDrawdown >= 0)
    })
  })
  
  describe('reset', () => {
    it('should reset performance metrics', async () => {
      await agent.initialize()
      
      // Generate some activity
      await agent.analyze(createContext())
      await agent.updateOnTrade({
        id: 'trade-1',
        timestamp: epochDateNow(),
        side: 'buy',
        price: 50000,
        size: 0.1,
        fee: 10,
        pnl: 100
      }, await agent.analyze(createContext()))
      
      // Reset
      await agent.reset()
      
      const performance = agent.getPerformance()
      assert.equal(performance.totalSignals, 0)
      assert.equal(performance.profitableSignals, 0)
      assert.equal(performance.winRate, 0)
    })
  })
  
  describe('shutdown', () => {
    it('should cleanup and prevent further analysis', async () => {
      await agent.initialize()
      
      // Should work before shutdown
      await agent.analyze(createContext())
      
      // Shutdown
      await agent.shutdown()
      
      // Should fail after shutdown
      await assert.rejects(
        agent.analyze(createContext()),
        /not initialized/
      )
    })
  })
  
  describe('createSignal helper', () => {
    it('should create valid signal with clamped confidence', async () => {
      await agent.initialize()
      
      // Test internal createSignal through a custom method
      class TestAgentWithHelper extends TestAgent {
        async testCreateSignal(): Promise<AgentSignal> {
          return this.createSignal(
            'buy',
            1.5, // Should be clamped to 1
            'Test reason',
            'Test analysis',
            51000,
            49000,
            0.1
          )
        }
      }
      
      const agentWithHelper = new TestAgentWithHelper(metadata)
      await agentWithHelper.initialize()
      
      const signal = await agentWithHelper.testCreateSignal()
      assert.equal(signal.confidence, 1) // Clamped from 1.5
      assert.equal(signal.priceTarget, 51000)
      assert.equal(signal.stopLoss, 49000)
      assert.equal(signal.positionSize, 0.1)
    })
  })
})