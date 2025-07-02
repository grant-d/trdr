import { StockSymbol, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Candle, Logger, MarketDataRepository } from '@trdr/types'
import assert from 'node:assert/strict'
import { beforeEach, describe, it, mock } from 'node:test'
import { DataValidator } from './data-validator'

describe('DataValidator', () => {
  let validator: DataValidator
  let mockRepository: MarketDataRepository
  let mockLogger: Logger
  let testSymbol: StockSymbol
  
  beforeEach(() => {
    testSymbol = toStockSymbol('BTC-USD')
    
    mockRepository = {
      saveCandle: mock.fn(),
      saveCandlesBatch: mock.fn(),
      getCandles: mock.fn(() => Promise.resolve([])),
      getLatestCandle: mock.fn(() => Promise.resolve(null)),
      cleanup: mock.fn(() => Promise.resolve({ candlesDeleted: 0, ticksDeleted: 0 }))
    }
    
    mockLogger = {
      debug: mock.fn(),
      info: mock.fn(),
      warn: mock.fn(),
      error: mock.fn()
    }
    
    validator = new DataValidator({
      repository: mockRepository,
      logger: mockLogger
    })
  })

  const createTestCandle = (timestamp: number, overrides?: Partial<Candle>): Candle => ({
    symbol: testSymbol,
    interval: '1m',
    timestamp: toEpochDate(timestamp),
    openTime: toEpochDate(timestamp),
    closeTime: toEpochDate(timestamp + 60000),
    open: 100,
    high: 105,
    low: 95,
    close: 102,
    volume: 1000,
    ...overrides
  })

  describe('validateDataContinuity', () => {
    it('should detect gaps in data', async () => {
      const candles = [
        createTestCandle(1000000),
        createTestCandle(1060000), // Normal 1m interval
        createTestCandle(1120000), // Normal 1m interval
        createTestCandle(1300000), // Gap! Missing 2 candles (1180000, 1240000)
        createTestCandle(1360000)
      ]

      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(candles))

      const issues = await validator.validateDataContinuity(
        testSymbol,
        '1m',
        toEpochDate(1000000),
        toEpochDate(1400000)
      )

      assert.equal(issues.length, 1)
      assert.equal(issues[0]!.type, 'gap')
      assert.equal(issues[0]!.affectedCandles, 2)
      assert.equal(issues[0]!.suggestedAction, 'backfill')
    })

    it('should not report gaps for continuous data', async () => {
      const candles = Array.from({ length: 10 }, (_, i) => 
        createTestCandle(1000000 + i * 60000)
      )

      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(candles))

      const issues = await validator.validateDataContinuity(
        testSymbol,
        '1m',
        toEpochDate(1000000),
        toEpochDate(1600000)
      )

      assert.equal(issues.length, 0)
    })

    it('should calculate gap severity correctly', async () => {
      const candles = [
        createTestCandle(1000000),
        createTestCandle(1060000),
        createTestCandle(7260000) // Huge gap - 102 candles missing
      ]

      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(candles))

      const issues = await validator.validateDataContinuity(
        testSymbol,
        '1m',
        toEpochDate(1000000),
        toEpochDate(8000000)
      )

      assert.equal(issues.length, 1)
      assert.equal(issues[0]!.severity, 'critical')
      assert.equal(issues[0]!.affectedCandles, 102)
    })
  })

  describe('detectOutliers', () => {
    describe('Z-score method', () => {
      it('should detect price outliers using z-score', async () => {
        const candles = [
          createTestCandle(1000000, { close: 100 }),
          createTestCandle(1060000, { close: 102 }),
          createTestCandle(1120000, { close: 98 }),
          createTestCandle(1180000, { close: 101 }),
          createTestCandle(1240000, { close: 99 }),
          createTestCandle(1300000, { close: 300 }), // Outlier!
          createTestCandle(1360000, { close: 100 }),
          createTestCandle(1420000, { close: 101 }),
          createTestCandle(1480000, { close: 99 }),
          createTestCandle(1540000, { close: 102 })
        ]

        const issues = await validator.detectOutliers(candles, {
          outlierMethod: 'zscore',
          zscoreThreshold: 2
        })

        assert.ok(issues.length > 0)
        assert.equal(issues[0]!.type, 'outlier')
        assert.ok(issues[0]!.description.includes('close'))
      })

      it('should not flag normal variations as outliers', async () => {
        const candles = Array.from({ length: 20 }, (_, i) => 
          createTestCandle(1000000 + i * 60000, { 
            close: 100 + Math.sin(i / 3) * 2 // Small variations
          })
        )

        const issues = await validator.detectOutliers(candles, {
          outlierMethod: 'zscore',
          zscoreThreshold: 3
        })

        assert.equal(issues.length, 0)
      })
    })

    describe('IQR method', () => {
      it('should detect outliers using IQR', async () => {
        const candles = [
          createTestCandle(1000000, { close: 100 }),
          createTestCandle(1060000, { close: 102 }),
          createTestCandle(1120000, { close: 98 }),
          createTestCandle(1180000, { close: 101 }),
          createTestCandle(1240000, { close: 99 }),
          createTestCandle(1300000, { close: 150 }), // Outlier!
          createTestCandle(1360000, { close: 100 }),
          createTestCandle(1420000, { close: 101 })
        ]

        const issues = await validator.detectOutliers(candles, {
          outlierMethod: 'iqr',
          iqrMultiplier: 1.5,
          minDataPoints: 5
        })

        assert.ok(issues.length > 0)
        assert.equal(issues[0]!.type, 'outlier')
      })
    })

    describe('Isolation method', () => {
      it('should detect isolated anomalies', async () => {
        const candles = []
        // Create steady data
        for (let i = 0; i < 20; i++) {
          candles.push(createTestCandle(1000000 + i * 60000, { 
            close: 100,
            volume: 1000 
          }))
        }
        // Insert anomaly
        candles[10] = createTestCandle(1600000, { 
          close: 120, // 20% spike
          volume: 5000 // 500% volume spike
        })

        const issues = await validator.detectOutliers(candles, {
          outlierMethod: 'isolation'
        })

        assert.ok(issues.length > 0)
        assert.equal(issues[0]!.type, 'outlier')
        assert.ok(issues[0]!.description.includes('Isolated anomaly'))
      })
    })
  })

  describe('verifyDataIntegrity', () => {
    it('should return comprehensive validation report', async () => {
      const candles = [
        createTestCandle(1000000),
        createTestCandle(1060000),
        createTestCandle(1240000), // Gap
        createTestCandle(1300000, { high: 90, low: 95 }), // Invalid: high < low
        createTestCandle(1360000),
        createTestCandle(1360000) // Duplicate
      ]

      ;(mockRepository.getCandles as any).mock.mockImplementation(() => Promise.resolve(candles))

      const report = await validator.verifyDataIntegrity(
        testSymbol,
        '1m',
        toEpochDate(1000000),
        toEpochDate(1400000)
      )

      assert.equal(report.symbol, testSymbol)
      assert.equal(report.interval, '1m')
      assert.equal(report.totalCandles, 6)
      assert.ok(report.issues.length > 0)
      
      // Should detect gap, invalid data, and duplicate
      const issueTypes = report.issues.map(i => i.type)
      assert.ok(issueTypes.includes('gap'))
      assert.ok(issueTypes.includes('invalid'))
      assert.ok(issueTypes.includes('duplicate'))
    })

    it('should calculate health status correctly', async () => {
      // Healthy data
      const healthyCandles = Array.from({ length: 10 }, (_, i) => 
        createTestCandle(1000000 + i * 60000)
      )

      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(healthyCandles))

      const healthyReport = await validator.verifyDataIntegrity(
        testSymbol,
        '1m'
      )

      assert.equal(healthyReport.overallHealth, 'healthy')

      // Critical data
      const problematicCandles = [
        createTestCandle(1000000, { high: 90, low: 100 }), // Invalid
        createTestCandle(1060000, { open: -10 }), // Negative price
        createTestCandle(1120000, { volume: -100 }) // Negative volume
      ]

      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(problematicCandles))

      const criticalReport = await validator.verifyDataIntegrity(
        testSymbol,
        '1m'
      )

      assert.equal(criticalReport.overallHealth, 'critical')
    })
  })

  describe('repairCorruptedData', () => {
    it('should skip repairs when autoRepair is disabled', async () => {
      const issues = [{
        type: 'gap' as const,
        severity: 'high' as const,
        symbol: testSymbol,
        interval: '1m',
        startTime: toEpochDate(1000000),
        endTime: toEpochDate(1060000),
        description: 'Test gap',
        suggestedAction: 'backfill' as const
      }]

      const result = await validator.repairCorruptedData(
        testSymbol,
        '1m',
        issues,
        { autoRepair: false }
      )

      assert.equal(result.repaired, 0)
      assert.equal(result.failed, 1)
      assert.equal(result.results[0]!.success, false)
      assert.equal(result.results[0]!.error, 'Auto-repair disabled')
    })

    it('should attempt repairs when autoRepair is enabled', async () => {
      const issues = [{
        type: 'gap' as const,
        severity: 'high' as const,
        symbol: testSymbol,
        interval: '1m',
        startTime: toEpochDate(1000000),
        endTime: toEpochDate(1060000),
        description: 'Test gap',
        suggestedAction: 'backfill' as const
      }]

      const result = await validator.repairCorruptedData(
        testSymbol,
        '1m',
        issues,
        { autoRepair: true }
      )

      assert.equal(result.failed, 1) // Backfill not implemented
      assert.equal(result.repaired, 0)
      assert.ok((mockLogger.info as any).mock.calls.length > 0)
    })
  })

  describe('edge cases', () => {
    it('should handle empty data', async () => {
      ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve([]))

      const issues = await validator.validateDataContinuity(
        testSymbol,
        '1m',
        toEpochDate(1000000),
        toEpochDate(1400000)
      )

      assert.equal(issues.length, 0)
    })

    it('should handle insufficient data points', async () => {
      const candles = [createTestCandle(1000000)]

      const issues = await validator.detectOutliers(candles, {
        minDataPoints: 10
      })

      assert.equal(issues.length, 0)
      assert.ok((mockLogger.warn as any).mock.calls.length === 0) // No warning for detectOutliers
    })

    it('should validate different interval formats', async () => {
      const testCases = [
        { interval: '1m', expected: 60000 },
        { interval: '5m', expected: 300000 },
        { interval: '1h', expected: 3600000 },
        { interval: '1d', expected: 86400000 }
      ]

      for (const { interval, expected } of testCases) {
        const candles = [
          createTestCandle(1000000),
          createTestCandle(1000000 + expected * 3) // Gap of 2 intervals
        ]

        ;(mockRepository.getCandles as any).mock.mockImplementationOnce(() => Promise.resolve(candles))

        const issues = await validator.validateDataContinuity(
          testSymbol,
          interval,
          toEpochDate(1000000),
          toEpochDate(2000000)
        )

        assert.equal(issues.length, 1)
        assert.equal(issues[0]!.affectedCandles, 2)
      }
    })
  })
})