import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { AlpacaProvider } from '../../../src/providers/alpaca'
import type { HistoricalParams } from '../../../src/interfaces'
import { forceCleanupAsyncHandles } from '../../helpers/test-cleanup'

describe('AlpacaProvider Integration Tests', () => {
  let provider: AlpacaProvider
  
  beforeEach(() => {
    // Skip tests if API credentials not available
    const isPaper = process.env.ALPACA_PAPER?.toLowerCase() !== 'false'
    const apiKey = isPaper ? process.env.ALPACA_PAPER_API_KEY : process.env.ALPACA_LIVE_API_KEY
    const apiSecret = isPaper ? process.env.ALPACA_PAPER_API_SECRET : process.env.ALPACA_LIVE_API_SECRET
    
    if (!apiKey || !apiSecret) {
      console.log('Skipping Alpaca integration tests - API credentials not set')
      return
    }
    
    provider = new AlpacaProvider()
  })
  
  afterEach(async () => {
    if (provider && provider.isConnected()) {
      await provider.disconnect()
    }
    forceCleanupAsyncHandles()
  })
  
  it('should connect and disconnect successfully', async (t) => {
    const isPaper = process.env.ALPACA_PAPER?.toLowerCase() !== 'false'
    const apiKey = isPaper ? process.env.ALPACA_PAPER_API_KEY : process.env.ALPACA_LIVE_API_KEY
    
    if (!apiKey) {
      t.skip('API credentials not available')
      return
    }
    
    assert.strictEqual(provider.isConnected(), false)
    
    await provider.connect()
    assert.strictEqual(provider.isConnected(), true)
    
    await provider.disconnect()
    assert.strictEqual(provider.isConnected(), false)
  })
  
  it('should fetch historical data', async (t) => {
    const isPaper = process.env.ALPACA_PAPER?.toLowerCase() !== 'false'
    const apiKey = isPaper ? process.env.ALPACA_PAPER_API_KEY : process.env.ALPACA_LIVE_API_KEY
    
    if (!apiKey) {
      t.skip('API credentials not available')
      return
    }
    
    await provider.connect()
    
    // Use a recent date range from 2025
    const endDate = new Date('2025-07-04T00:00:00Z') // Yesterday
    const startDate = new Date('2025-06-01T00:00:00Z') // About a month ago
    
    const params: HistoricalParams = {
      symbols: ['AAPL'], // Apple stock
      start: startDate.getTime(),
      end: endDate.getTime(),
      timeframe: '1d' // Daily bars
    }
    
    const data: any[] = []
    for await (const ohlcv of provider.getHistoricalData(params)) {
      data.push(ohlcv)
    }
    
    assert.ok(data.length > 0, 'Should have received some data')
    
    // Verify data structure
    const firstBar = data[0]
    assert.strictEqual(firstBar.exchange, 'alpaca')
    assert.strictEqual(firstBar.symbol, 'AAPL')
    assert.ok(typeof firstBar.timestamp === 'number')
    assert.ok(typeof firstBar.open === 'number')
    assert.ok(typeof firstBar.high === 'number')
    assert.ok(typeof firstBar.low === 'number')
    assert.ok(typeof firstBar.close === 'number')
    assert.ok(typeof firstBar.volume === 'number')
    
    // Verify OHLC relationships
    assert.ok(firstBar.high >= firstBar.low)
    assert.ok(firstBar.high >= firstBar.open)
    assert.ok(firstBar.high >= firstBar.close)
    assert.ok(firstBar.low <= firstBar.open)
    assert.ok(firstBar.low <= firstBar.close)
  })
  
  it('should throw error when not connected', async () => {
    const params: HistoricalParams = {
      symbols: ['AAPL'],
      start: Date.now() - 60 * 60 * 1000,
      end: Date.now(),
      timeframe: '5m'
    }
    
    await assert.rejects(
      async () => {
        for await (const _ of provider.getHistoricalData(params)) {
          // Should not reach here
        }
      },
      /Provider not connected/
    )
  })
  
  it('should validate required environment variables', () => {
    const tempPaperKey = process.env.ALPACA_PAPER_API_KEY
    const tempPaperSecret = process.env.ALPACA_PAPER_API_SECRET
    const tempLiveKey = process.env.ALPACA_LIVE_API_KEY
    const tempLiveSecret = process.env.ALPACA_LIVE_API_SECRET
    
    try {
      // Clear all Alpaca env vars
      delete process.env.ALPACA_PAPER_API_KEY
      delete process.env.ALPACA_PAPER_API_SECRET
      delete process.env.ALPACA_LIVE_API_KEY
      delete process.env.ALPACA_LIVE_API_SECRET
      
      const newProvider = new AlpacaProvider()
      
      assert.throws(
        () => newProvider.validateEnvVars(),
        /Missing required environment variables/
      )
    } finally {
      // Restore env vars
      if (tempPaperKey) process.env.ALPACA_PAPER_API_KEY = tempPaperKey
      if (tempPaperSecret) process.env.ALPACA_PAPER_API_SECRET = tempPaperSecret
      if (tempLiveKey) process.env.ALPACA_LIVE_API_KEY = tempLiveKey
      if (tempLiveSecret) process.env.ALPACA_LIVE_API_SECRET = tempLiveSecret
    }
  })
  
  it('should support standard and arbitrary timeframes', (t) => {
    const isPaper = process.env.ALPACA_PAPER?.toLowerCase() !== 'false'
    const apiKey = isPaper ? process.env.ALPACA_PAPER_API_KEY : process.env.ALPACA_LIVE_API_KEY
    
    if (!apiKey) {
      t.skip('API credentials not available')
      return
    }
    
    const timeframes = provider.getSupportedTimeframes()
    // Standard timeframes
    assert.ok(timeframes.includes('1m'))
    assert.ok(timeframes.includes('5m'))
    assert.ok(timeframes.includes('15m'))
    assert.ok(timeframes.includes('30m'))
    assert.ok(timeframes.includes('1h'))
    assert.ok(timeframes.includes('2h'))
    assert.ok(timeframes.includes('4h'))
    assert.ok(timeframes.includes('1d'))
    assert.ok(timeframes.includes('1w'))
    assert.ok(timeframes.includes('1M'))
    // Arbitrary timeframes from PRD
    assert.ok(timeframes.includes('3m'))
    assert.ok(timeframes.includes('17m'))
    assert.ok(timeframes.includes('90s'))
  })
  
  it('should determine paper/live mode correctly', () => {
    // Test with paper mode
    const paperProvider = new AlpacaProvider({ paper: true })
    assert.deepStrictEqual(
      paperProvider.getRequiredEnvVars(),
      ['ALPACA_PAPER_API_KEY', 'ALPACA_PAPER_API_SECRET']
    )
    
    // Test with live mode
    const liveProvider = new AlpacaProvider({ paper: false })
    assert.deepStrictEqual(
      liveProvider.getRequiredEnvVars(),
      ['ALPACA_LIVE_API_KEY', 'ALPACA_LIVE_API_SECRET']
    )
  })
  
  it('should fetch crypto historical data', async (t) => {
    const isPaper = process.env.ALPACA_PAPER?.toLowerCase() !== 'false'
    const apiKey = isPaper ? process.env.ALPACA_PAPER_API_KEY : process.env.ALPACA_LIVE_API_KEY
    
    if (!apiKey) {
      t.skip('API credentials not available')
      return
    }
    
    await provider.connect()
    
    // Use a recent date range
    const endDate = new Date('2025-07-04T00:00:00Z')
    const startDate = new Date('2025-07-03T00:00:00Z')
    
    const params: HistoricalParams = {
      symbols: ['BTC-USD'], // Bitcoin with dash delimiter (will be normalized to BTC/USD)
      start: startDate.getTime(),
      end: endDate.getTime(),
      timeframe: '1h' // Hourly bars
    }
    
    const data: any[] = []
    for await (const ohlcv of provider.getHistoricalData(params)) {
      data.push(ohlcv)
    }
    
    assert.ok(data.length > 0, 'Should have received some crypto data')
    
    // Verify data structure
    const firstBar = data[0]
    assert.strictEqual(firstBar.exchange, 'alpaca')
    assert.strictEqual(firstBar.symbol, 'BTC-USD') // Original symbol is preserved
    assert.ok(typeof firstBar.timestamp === 'number')
    assert.ok(typeof firstBar.open === 'number')
    assert.ok(typeof firstBar.high === 'number')
    assert.ok(typeof firstBar.low === 'number')
    assert.ok(typeof firstBar.close === 'number')
    assert.ok(typeof firstBar.volume === 'number')
  })
})