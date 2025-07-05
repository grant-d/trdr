import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { CoinbaseProvider } from '../../../src/providers/coinbase'
import type { HistoricalParams } from '../../../src/interfaces'

describe('CoinbaseProvider Integration Tests', () => {
  let provider: CoinbaseProvider
  
  beforeEach(() => {
    // Skip tests if API credentials not available
    if (!process.env.COINBASE_API_KEY || !process.env.COINBASE_API_SECRET) {
      console.log('Skipping Coinbase integration tests - API credentials not set')
      return
    }
    
    provider = new CoinbaseProvider()
  })
  
  afterEach(async () => {
    if (provider?.isConnected()) {
      await provider.disconnect()
    }
  })
  
  it('should connect and disconnect successfully', async (t) => {
    if (!process.env.COINBASE_API_KEY) {
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
    if (!process.env.COINBASE_API_KEY) {
      t.skip('API credentials not available')
      return
    }
    
    await provider.connect()
    
    const params: HistoricalParams = {
      symbols: ['BTC-USD'],
      start: Date.now() - 24 * 60 * 60 * 1000, // 24 hours ago
      end: Date.now(),
      timeframe: '1h'
    }
    
    const data: any[] = []
    for await (const ohlcv of provider.getHistoricalData(params)) {
      data.push(ohlcv)
    }
    
    assert.ok(data.length > 0, 'Should have received some data')
    
    // Verify data structure
    const firstBar = data[0]
    assert.strictEqual(firstBar.exchange, 'coinbase')
    assert.strictEqual(firstBar.symbol, 'BTC-USD')
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
  
  it('should throw error when not connected', async (t) => {
    if (!process.env.COINBASE_API_KEY) {
      t.skip('API credentials not available')
      return
    }
    
    const params: HistoricalParams = {
      symbols: ['BTC-USD'],
      start: Date.now() - 60 * 60 * 1000,
      end: Date.now(),
      timeframe: '5m'
    }
    
    await assert.rejects(
      async () => {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for await (const _ of provider.getHistoricalData(params)) {
          // Should not reach here
        }
      },
      /Provider not connected/
    )
  })
  
  it('should validate required environment variables', () => {
    const tempKey = process.env.COINBASE_API_KEY
    const tempSecret = process.env.COINBASE_API_SECRET
    
    try {
      delete process.env.COINBASE_API_KEY
      delete process.env.COINBASE_API_SECRET
      
      const newProvider = new CoinbaseProvider()
      
      assert.throws(
        () => newProvider.validateEnvVars(),
        /Missing required environment variables/
      )
    } finally {
      if (tempKey) process.env.COINBASE_API_KEY = tempKey
      if (tempSecret) process.env.COINBASE_API_SECRET = tempSecret
    }
  })
  
  it('should support standard timeframes', (t) => {
    if (!process.env.COINBASE_API_KEY) {
      t.skip('API credentials not available')
      return
    }
    
    const timeframes = provider.getSupportedTimeframes()
    assert.ok(timeframes.includes('1m'))
    assert.ok(timeframes.includes('5m'))
    assert.ok(timeframes.includes('15m'))
    assert.ok(timeframes.includes('30m'))
    assert.ok(timeframes.includes('1h'))
    assert.ok(timeframes.includes('2h'))
    assert.ok(timeframes.includes('6h'))
    assert.ok(timeframes.includes('1d'))
  })
})