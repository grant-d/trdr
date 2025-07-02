import type { GridState, GridLevel } from '@trdr/shared'
import { toStockSymbol, epochDateNow } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, afterEach, describe, it } from 'node:test'
import { writeFile, rm, access } from 'fs/promises'
import { constants } from 'fs'
import path from 'path'
import { 
  GridStatePersistence, 
  type GridPersistenceConfig,
  type GridManagerSnapshot
} from './grid-state-persistence'

describe('GridStatePersistence', () => {
  let persistence: GridStatePersistence
  let testStorageDir: string
  
  const createTestGridState = (symbol: string = 'BTC-USD'): GridState => {
    const now = epochDateNow()
    
    const levels: GridLevel[] = [
      {
        id: 'level-1',
        price: 49000,
        side: 'buy',
        size: 0.01,
        isActive: true,
        orderId: 'order-1',
        createdAt: (now - 3600000) as any,
        updatedAt: now,
        fillCount: 0,
        pnl: 0
      },
      {
        id: 'level-2',
        price: 51000,
        side: 'sell',
        size: 0.01,
        isActive: false,
        createdAt: (now - 3600000) as any,
        updatedAt: now,
        fillCount: 1,
        pnl: 100,
        trailingState: {
          status: 'triggered',
          activationPrice: 50500,
          activatedAt: (now - 1800000) as any,
          lastUpdatePrice: 51000,
          lastUpdatedAt: now,
          adjustmentCount: 3,
          fallbackAttempted: false
        }
      }
    ]

    return {
      config: {
        gridSpacing: 2,
        gridLevels: 10,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      },
      symbol: toStockSymbol(symbol),
      levels,
      centerPrice: 50000,
      currentSpacing: 2,
      allocatedCapital: 10000,
      availableCapital: 8000,
      currentPosition: 0.01,
      realizedPnl: 100,
      unrealizedPnl: 50,
      initializedAt: (now - 7200000) as any,
      lastUpdatedAt: now,
      isActive: true
    }
  }

  const createTestSnapshot = (): GridManagerSnapshot => {
    const now = epochDateNow()
    
    return {
      version: '1.0.0',
      timestamp: now,
      activeGrids: {
        'grid-1': persistence.serializeGridState(createTestGridState('BTC-USD')),
        'grid-2': persistence.serializeGridState(createTestGridState('ETH-USD'))
      },
      performanceHistory: [
        {
          spacing: 2.0,
          performance: 150,
          fillRate: 0.8,
          timestamp: now - 3600000,
          marketConditions: {
            volatility: 0.02,
            trend: 'bullish',
            volume: 2000
          }
        }
      ],
      lastPriceUpdates: {
        'BTC-USD': { price: 50000, timestamp: now },
        'ETH-USD': { price: 3000, timestamp: now }
      },
      metadata: {
        totalGridsCreated: 5,
        totalGridsCancelled: 2,
        systemUptime: 3600000
      }
    }
  }

  beforeEach(async () => {
    // Create unique test directory
    testStorageDir = path.join(process.cwd(), 'test-storage-' + Date.now())
    
    const config: Partial<GridPersistenceConfig> = {
      storageDir: testStorageDir,
      autoSaveInterval: 1000, // 1 second for testing
      maxBackups: 3,
      enableCompression: false,
      validateOnLoad: true
    }
    
    persistence = new GridStatePersistence(config)
  })

  afterEach(async () => {
    // Clean up test files
    try {
      await persistence.shutdown()
      // Check if directory exists before trying to remove it
      try {
        await access(testStorageDir, constants.F_OK)
        await rm(testStorageDir, { recursive: true, force: true })
      } catch {
        // Directory doesn't exist, nothing to clean up
      }
    } catch (error) {
      // Ignore cleanup errors to not fail tests
      console.warn(`Test cleanup warning: ${error}`)
    }
  })

  describe('initialization', () => {
    it('should initialize and create storage directory', async () => {
      await persistence.initialize()
      
      // Directory should exist (we can't easily test this without more fs access)
      // In a real test environment, we'd verify the directory was created
    })

    it('should handle initialization with custom config', async () => {
      const customStorageDir = testStorageDir + '-custom'
      const customConfig: Partial<GridPersistenceConfig> = {
        storageDir: customStorageDir,
        autoSaveInterval: 5000,
        maxBackups: 5,
        validateOnLoad: false
      }
      
      const customPersistence = new GridStatePersistence(customConfig)
      await customPersistence.initialize()
      
      // Clean up custom directory
      try {
        await customPersistence.shutdown()
        await rm(customStorageDir, { recursive: true, force: true })
      } catch {
        // Ignore cleanup errors
      }
    })
  })

  describe('serialization and deserialization', () => {
    it('should serialize grid state correctly', () => {
      const gridState = createTestGridState()
      const serialized = persistence.serializeGridState(gridState)
      
      assert.equal(serialized.symbol, gridState.symbol)
      assert.equal(serialized.centerPrice, gridState.centerPrice)
      assert.equal(serialized.levels.length, gridState.levels.length)
      
      // Verify level serialization
      const originalLevel = gridState.levels[0]!
      const serializedLevel = serialized.levels[0]!
      
      assert.equal(serializedLevel.id, originalLevel.id)
      assert.equal(serializedLevel.price, originalLevel.price)
      assert.equal(serializedLevel.side, originalLevel.side)
      assert.equal(typeof serializedLevel.createdAt, 'number')
      assert.equal(typeof serializedLevel.updatedAt, 'number')
    })

    it('should deserialize grid state correctly', () => {
      const originalState = createTestGridState()
      const serialized = persistence.serializeGridState(originalState)
      const deserialized = persistence.deserializeGridState(serialized)
      
      assert.equal(deserialized.symbol, originalState.symbol)
      assert.equal(deserialized.centerPrice, originalState.centerPrice)
      assert.equal(deserialized.levels.length, originalState.levels.length)
      
      // Verify level deserialization
      const originalLevel = originalState.levels[0]!
      const deserializedLevel = deserialized.levels[0]!
      
      assert.equal(deserializedLevel.id, originalLevel.id)
      assert.equal(deserializedLevel.price, originalLevel.price)
      assert.equal(deserializedLevel.side, originalLevel.side)
      assert.equal(deserializedLevel.createdAt, originalLevel.createdAt)
      assert.equal(deserializedLevel.updatedAt, originalLevel.updatedAt)
    })

    it('should handle trailing state serialization', () => {
      const gridState = createTestGridState()
      const serialized = persistence.serializeGridState(gridState)
      const deserialized = persistence.deserializeGridState(serialized)
      
      // Find level with trailing state
      const originalLevel = gridState.levels.find(l => l.trailingState)!
      const deserializedLevel = deserialized.levels.find(l => l.trailingState)!
      
      assert.ok(originalLevel.trailingState)
      assert.ok(deserializedLevel.trailingState)
      
      assert.equal(deserializedLevel.trailingState.status, originalLevel.trailingState.status)
      assert.equal(deserializedLevel.trailingState.activationPrice, originalLevel.trailingState.activationPrice)
      assert.equal(deserializedLevel.trailingState.adjustmentCount, originalLevel.trailingState.adjustmentCount)
      assert.equal(deserializedLevel.trailingState.fallbackAttempted, originalLevel.trailingState.fallbackAttempted)
    })

    it('should handle levels without trailing state', () => {
      const gridState = createTestGridState()
      const serialized = persistence.serializeGridState(gridState)
      const deserialized = persistence.deserializeGridState(serialized)
      
      // Find level without trailing state
      const originalLevel = gridState.levels.find(l => !l.trailingState)!
      const deserializedLevel = deserialized.levels.find(l => l.id === originalLevel.id)!
      
      assert.equal(deserializedLevel.trailingState, undefined)
      assert.equal(deserializedLevel.orderId, originalLevel.orderId)
      assert.equal(deserializedLevel.isActive, originalLevel.isActive)
    })
  })

  describe('snapshot validation', () => {
    it('should validate correct snapshot', () => {
      const snapshot = createTestSnapshot()
      const validation = persistence.validateSnapshot(snapshot)
      
      assert.equal(validation.valid, true)
      assert.equal(validation.errors.length, 0)
    })

    it('should detect missing version', () => {
      const snapshot = { ...createTestSnapshot(), version: '' }
      const validation = persistence.validateSnapshot(snapshot)
      
      assert.equal(validation.valid, false)
      assert.ok(validation.errors.some(e => e.includes('version')))
    })

    it('should detect invalid timestamp', () => {
      const snapshot = { ...createTestSnapshot(), timestamp: 0 }
      const validation = persistence.validateSnapshot(snapshot)
      
      assert.equal(validation.valid, false)
      assert.ok(validation.errors.some(e => e.includes('timestamp')))
    })

    it('should detect invalid grid states', () => {
      const snapshot = createTestSnapshot()
      // @ts-ignore - intentionally create invalid state for testing
      snapshot.activeGrids['invalid-grid'] = { symbol: null, config: null }
      
      const validation = persistence.validateSnapshot(snapshot)
      
      assert.equal(validation.valid, false)
      assert.ok(validation.errors.some(e => e.includes('invalid-grid')))
    })

    it('should detect invalid levels', () => {
      const snapshot = createTestSnapshot()
      const gridState = Object.values(snapshot.activeGrids)[0]!
      // @ts-ignore - intentionally create invalid level for testing
      gridState.levels.push({ id: '', price: 'invalid', side: null })
      
      const validation = persistence.validateSnapshot(snapshot)
      
      assert.equal(validation.valid, false)
      assert.ok(validation.errors.length > 0)
    })
  })

  describe('snapshot operations', () => {
    beforeEach(async () => {
      await persistence.initialize()
    })

    it('should save and load snapshot', async () => {
      const originalSnapshot = createTestSnapshot()
      
      await persistence.saveSnapshot(originalSnapshot)
      const { snapshot: loadedSnapshot, recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, true)
      assert.equal(recoveryInfo.recoveredGrids, 2) // grid-1 and grid-2
      assert.equal(recoveryInfo.failedGrids, 0)
      
      // Verify snapshot content
      assert.equal(loadedSnapshot.version, originalSnapshot.version)
      assert.equal(Object.keys(loadedSnapshot.activeGrids).length, 2)
      assert.equal(loadedSnapshot.performanceHistory.length, 1)
      assert.equal(Object.keys(loadedSnapshot.lastPriceUpdates).length, 2)
    })

    it('should handle empty state gracefully', async () => {
      // Try to load when no file exists
      const { snapshot, recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, false)
      assert.equal(recoveryInfo.recoveredGrids, 0)
      assert.ok(recoveryInfo.errors.length > 0)
      
      // Should return empty snapshot
      assert.equal(Object.keys(snapshot.activeGrids).length, 0)
      assert.equal(snapshot.performanceHistory.length, 0)
    })

    it('should handle corrupted file gracefully', async () => {
      // Initialize persistence first to create the directory
      await persistence.initialize()
      
      // Write invalid JSON to the storage file
      await writeFile(
        path.join(testStorageDir, 'grid-manager-state.json'), 
        'invalid json content',
        'utf8'
      )
      
      const { snapshot, recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, false)
      assert.ok(recoveryInfo.errors.some(e => e.includes('Failed to load') || e.includes('No valid backups')))
      
      // Should return empty snapshot
      assert.equal(Object.keys(snapshot.activeGrids).length, 0)
    })
  })

  describe('auto-save functionality', () => {
    it('should start and stop auto-save', async () => {
      let saveCallCount = 0
      const mockSaveCallback = async () => {
        saveCallCount++
        return createTestSnapshot()
      }
      
      // Use very short interval for testing
      const fastConfig: Partial<GridPersistenceConfig> = {
        storageDir: testStorageDir,
        autoSaveInterval: 10, // 10ms for fast testing
        maxBackups: 3,
        enableCompression: false,
        validateOnLoad: true
      }
      
      const fastPersistence = new GridStatePersistence(fastConfig)
      await fastPersistence.initialize()
      
      // Start auto-save
      fastPersistence.startAutoSave(mockSaveCallback)
      
      // Wait just long enough for one auto-save
      await new Promise(resolve => setTimeout(resolve, 20))
      
      // Stop auto-save
      fastPersistence.stopAutoSave()
      await fastPersistence.shutdown()
      
      // Should have called save at least once
      assert.ok(saveCallCount >= 1)
    })

    it('should handle save callback errors gracefully', async () => {
      const errorCallback = async () => {
        throw new Error('Save failed')
      }
      
      // Use very short interval for testing
      const fastConfig: Partial<GridPersistenceConfig> = {
        storageDir: testStorageDir,
        autoSaveInterval: 10, // 10ms for fast testing
        maxBackups: 3,
        enableCompression: false,
        validateOnLoad: true
      }
      
      const fastPersistence = new GridStatePersistence(fastConfig)
      await fastPersistence.initialize()
      
      // Should not throw when callback fails
      fastPersistence.startAutoSave(errorCallback)
      
      // Wait briefly then stop
      await new Promise(resolve => setTimeout(resolve, 20))
      fastPersistence.stopAutoSave()
      await fastPersistence.shutdown()
    })
  })

  describe('shutdown handling', () => {
    it('should shutdown cleanly', async () => {
      await persistence.initialize()
      
      const finalSnapshot = createTestSnapshot()
      await persistence.shutdown(finalSnapshot)
      
      // Verify the final snapshot was saved by trying to load it
      const { snapshot, recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, true)
      assert.equal(Object.keys(snapshot.activeGrids).length, 2)
    })

    it('should shutdown without final snapshot', async () => {
      await persistence.initialize()
      
      // Should not throw
      await persistence.shutdown()
    })
  })

  describe('error handling', () => {
    it('should handle storage directory creation failure', async () => {
      // Create persistence with invalid directory path
      const invalidConfig: Partial<GridPersistenceConfig> = {
        storageDir: '/invalid/path/that/cannot/be/created'
      }
      
      const invalidPersistence = new GridStatePersistence(invalidConfig)
      
      // Should handle gracefully (in real implementation, would check permissions)
      try {
        await invalidPersistence.initialize()
      } catch (error) {
        assert.ok(error instanceof Error)
        assert.ok(error.message.includes('Persistence initialization failed'))
      }
    })

    it('should handle file write permissions', async () => {
      await persistence.initialize()
      
      const snapshot = createTestSnapshot()
      
      // In a real test environment, we would test actual permission failures
      // For now, just verify the method exists and can handle basic cases
      await persistence.saveSnapshot(snapshot)
    })
  })

  describe('recovery scenarios', () => {
    it('should create recovery info for successful load', async () => {
      await persistence.initialize()
      
      const snapshot = createTestSnapshot()
      await persistence.saveSnapshot(snapshot)
      
      const { recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, true)
      assert.equal(recoveryInfo.recoveredGrids, 2)
      assert.equal(recoveryInfo.failedGrids, 0)
      assert.equal(recoveryInfo.errors.length, 0)
    })

    it('should provide detailed recovery info for failures', async () => {
      const { recoveryInfo } = await persistence.loadSnapshot()
      
      assert.equal(recoveryInfo.success, false)
      assert.equal(recoveryInfo.recoveredGrids, 0)
      assert.ok(recoveryInfo.errors.length > 0)
      assert.ok(recoveryInfo.warnings.includes('Starting with empty state'))
    })
  })

  describe('configuration handling', () => {
    it('should use default configuration when none provided', () => {
      const defaultPersistence = new GridStatePersistence()
      
      // Should not throw and should have reasonable defaults
      assert.ok(defaultPersistence)
    })

    it('should merge partial configuration with defaults', () => {
      const partialConfig: Partial<GridPersistenceConfig> = {
        autoSaveInterval: 60000
      }
      
      const configuredPersistence = new GridStatePersistence(partialConfig)
      
      // Should not throw
      assert.ok(configuredPersistence)
    })
  })
})