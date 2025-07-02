import type { GridConfig, GridTrailingOrderConfig, Candle } from '@trdr/shared'
import { toStockSymbol, epochDateNow } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it, mock } from 'node:test'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { TrailingOrderManager } from '../orders/trailing-order-manager'
import { GridManager, type GridInitializationParams } from './grid-manager'

describe('GridManager', () => {
  let gridManager: GridManager
  let mockTrailingOrderManager: TrailingOrderManager
  let eventBus: EventBus

  const createMockTrailingOrderManager = (): TrailingOrderManager => ({
    createTrailingOrder: mock.fn(async (params) => ({
      id: 'test-order-id',
      symbol: params.symbol,
      side: params.side,
      type: 'trailing',
      size: params.size,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'pending',
      state: 'created',
      trailPercent: params.trailPercent,
      limitPrice: params.limitPrice,
      bestPrice: params.currentPrice,
      triggerPrice: params.currentPrice,
      filledSize: 0,
      averageFillPrice: 0,
      fees: 0,
      lastModified: Date.now(),
      fills: [],
      lastUpdateTime: Date.now(),
      isTriggered: false
    })),
    removeOrder: mock.fn(async () => {}),
    getActiveOrders: mock.fn(() => []),
    getOrder: mock.fn(() => undefined),
    processMarketUpdate: mock.fn(async () => {})
  } as any)

  beforeEach(() => {
    eventBus = EventBus.getInstance()

    // Register all event types
    Object.values(EventTypes).forEach(eventType => {
      eventBus.registerEvent(eventType)
    })

    mockTrailingOrderManager = createMockTrailingOrderManager()
    gridManager = new GridManager(eventBus, mockTrailingOrderManager, {}, {}, {})
  })

  describe('createGrid', () => {
    it('should create a new grid with specified configuration', async () => {
      const config: GridConfig = {
        gridSpacing: 2, // 2% spacing
        gridLevels: 10,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 10000,
        baseAmount: 0.1,
        quoteAmount: 5000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)

      assert.ok(result.gridId)
      assert.equal(result.totalLevels, 10)
      assert.equal(result.spacing, 2)
      assert.ok(result.levels.length > 0)

      // Verify grid state was created
      const gridState = gridManager.getGridState(result.gridId)
      assert.ok(gridState)
      assert.equal(gridState.symbol, 'BTC-USD')
      assert.equal(gridState.allocatedCapital, 10000)
      assert.equal(gridState.isActive, true)
    })

    it('should generate buy and sell levels', async () => {
      const config: GridConfig = {
        gridSpacing: 5, // 5% spacing
        gridLevels: 6, // 3 buy + 3 sell levels
        trailPercent: 1.0,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.15
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('ETH-USD'),
        allocatedCapital: 5000,
        baseAmount: 1.0,
        quoteAmount: 2500,
        riskLevel: 0.3
      }

      const result = await gridManager.createGrid(config, params)
      const gridState = gridManager.getGridState(result.gridId)

      assert.ok(gridState)
      
      const buyLevels = gridState.levels.filter(l => l.side === 'buy')
      const sellLevels = gridState.levels.filter(l => l.side === 'sell')

      assert.equal(buyLevels.length, 3)
      assert.equal(sellLevels.length, 3)

      // Verify levels are sorted by price
      const prices = gridState.levels.map(l => l.price)
      for (let i = 1; i < prices.length; i++) {
        assert.ok(prices[i]! >= prices[i - 1]!, 'Levels should be sorted by price')
      }
    })

    it('should emit grid creation event', async () => {
      const config: GridConfig = {
        gridSpacing: 3,
        gridLevels: 4,
        trailPercent: 0.8,
        minOrderSize: 0.01,
        maxOrderSize: 2.0,
        rebalanceThreshold: 0.12
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('DOGE-USD'),
        allocatedCapital: 1000,
        baseAmount: 100,
        quoteAmount: 500,
        riskLevel: 0.7
      }

      const eventHandler = mock.fn()
      gridManager.on('gridCreated', eventHandler)

      const result = await gridManager.createGrid(config, params)

      assert.equal(eventHandler.mock.calls.length, 1)
      assert.equal(eventHandler.mock.calls[0]?.arguments[0], result.gridId)
    })
  })

  describe('activateNearbyGrids', () => {
    it('should activate grid levels near the current price', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 10,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 10000,
        baseAmount: 0.1,
        quoteAmount: 5000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const currentPrice = 50000 // Same as placeholder center price

      await gridManager.activateNearbyGrids(result.gridId, currentPrice)

      // Verify trailing orders were created for nearby levels
      const createOrderCalls = (mockTrailingOrderManager.createTrailingOrder as any).mock.calls
      assert.ok(createOrderCalls.length > 0, 'Should have created trailing orders')
    })

    it('should throw error for non-existent grid', async () => {
      await assert.rejects(
        () => gridManager.activateNearbyGrids('non-existent-id', 50000),
        /Grid .* not found/
      )
    })
  })

  describe('cancelGrid', () => {
    it('should cancel all active orders and deactivate grid', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 6,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 5000,
        baseAmount: 0.05,
        quoteAmount: 2500,
        riskLevel: 0.4
      }

      const result = await gridManager.createGrid(config, params)
      
      // Activate some levels first
      await gridManager.activateNearbyGrids(result.gridId, 50000)

      const eventHandler = mock.fn()
      gridManager.on('gridCancelled', eventHandler)

      await gridManager.cancelGrid(result.gridId, 'Test cancellation')

      // Verify grid is no longer active
      const gridState = gridManager.getGridState(result.gridId)
      assert.ok(gridState)
      assert.equal(gridState.isActive, false)

      // Verify cancellation event was emitted
      assert.equal(eventHandler.mock.calls.length, 1)
      assert.equal(eventHandler.mock.calls[0]?.arguments[0], result.gridId)
      assert.equal(eventHandler.mock.calls[0]?.arguments[1], 'Test cancellation')

      // Verify orders were cancelled
      const removeOrderCalls = (mockTrailingOrderManager.removeOrder as any).mock.calls
      assert.ok(removeOrderCalls.length > 0, 'Should have cancelled orders')
    })

    it('should handle cancelling non-existent grid gracefully', async () => {
      // Should not throw
      await gridManager.cancelGrid('non-existent-id', 'Test')

      // No active grids should exist
      const activeGrids = gridManager.getActiveGrids()
      assert.equal(activeGrids.length, 0)
    })
  })

  describe('getActiveGrids', () => {
    it('should return only active grids', async () => {
      const config: GridConfig = {
        gridSpacing: 3,
        gridLevels: 8,
        trailPercent: 1.0,
        minOrderSize: 0.01,
        maxOrderSize: 0.5,
        rebalanceThreshold: 0.08
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('ETH-USD'),
        allocatedCapital: 3000,
        baseAmount: 1.0,
        quoteAmount: 1500,
        riskLevel: 0.6
      }

      // Create two grids
      const grid1 = await gridManager.createGrid(config, params)
      const grid2 = await gridManager.createGrid(config, {
        ...params,
        symbol: toStockSymbol('ADA-USD')
      })

      // Cancel one grid
      await gridManager.cancelGrid(grid1.gridId)

      const activeGrids = gridManager.getActiveGrids()
      assert.equal(activeGrids.length, 1)
      assert.equal(activeGrids[0]?.[0], grid2.gridId)
      assert.equal(activeGrids[0]?.[1].isActive, true)
    })

    it('should return empty array when no active grids', async () => {
      const activeGrids = gridManager.getActiveGrids()
      assert.equal(activeGrids.length, 0)
    })
  })

  describe('updateGrid', () => {
    it('should trigger rebalancing when price moves beyond threshold', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 6,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.05 // 5% threshold
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 8000,
        baseAmount: 0.08,
        quoteAmount: 4000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const originalState = gridManager.getGridState(result.gridId)

      // Move price beyond rebalance threshold (>5% from center)
      const newPrice = 50000 * 1.08 // 8% increase

      const eventHandler = mock.fn()
      gridManager.on('gridUpdated', eventHandler)

      // Add small delay to ensure different timestamp
      await new Promise(resolve => setTimeout(resolve, 1))

      await gridManager.updateGrid(result.gridId, newPrice)

      // Verify grid was updated
      const updatedState = gridManager.getGridState(result.gridId)
      assert.ok(updatedState)
      assert.ok(updatedState.lastUpdatedAt > originalState!.lastUpdatedAt)

      // Should have triggered rebalancing, which creates new levels and emits update event
      assert.equal(eventHandler.mock.calls.length, 1)
    })

    it('should not rebalance when price movement is within threshold', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 6,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1 // 10% threshold
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 6000,
        baseAmount: 0.06,
        quoteAmount: 3000,
        riskLevel: 0.4
      }

      const result = await gridManager.createGrid(config, params)
      const originalState = gridManager.getGridState(result.gridId)

      // Small price movement within threshold
      const newPrice = 50000 * 1.03 // 3% increase (below 10% threshold)

      const eventHandler = mock.fn()
      gridManager.on('gridUpdated', eventHandler)

      // Add small delay to ensure different timestamp
      await new Promise(resolve => setTimeout(resolve, 1))

      await gridManager.updateGrid(result.gridId, newPrice)

      // Should not have triggered full rebalancing
      assert.equal(eventHandler.mock.calls.length, 0)

      // But should have updated timestamp
      const updatedState = gridManager.getGridState(result.gridId)
      assert.ok(updatedState)
      assert.ok(updatedState.lastUpdatedAt > originalState!.lastUpdatedAt)
    })
  })

  describe('grid level generation', () => {
    it('should distribute capital progressively across levels', async () => {
      const config: GridConfig = {
        gridSpacing: 3,
        gridLevels: 6,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 12000,
        baseAmount: 0.1,
        quoteAmount: 6000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const gridState = gridManager.getGridState(result.gridId)

      assert.ok(gridState)

      const buyLevels = gridState.levels.filter(l => l.side === 'buy')
      const sellLevels = gridState.levels.filter(l => l.side === 'sell')

      // Verify we have both buy and sell levels
      assert.ok(buyLevels.length > 0, 'Should have buy levels')
      assert.ok(sellLevels.length > 0, 'Should have sell levels')

      // Sort buy levels by price (highest to lowest) to check levels closest to center first
      const sortedBuyLevels = buyLevels.sort((a, b) => b.price - a.price)

      // Verify capital distribution - levels closer to center (higher price for buy) should have more capital
      for (let i = 0; i < sortedBuyLevels.length - 1; i++) {
        const currentCapital = sortedBuyLevels[i]!.size * sortedBuyLevels[i]!.price
        const nextCapital = sortedBuyLevels[i + 1]!.size * sortedBuyLevels[i + 1]!.price
        assert.ok(currentCapital >= nextCapital, `Buy level ${i} (price: ${sortedBuyLevels[i]!.price.toFixed(2)}) should have more or equal capital than level ${i + 1} (price: ${sortedBuyLevels[i + 1]!.price.toFixed(2)})`)
      }

      // Verify total capital allocation is reasonable (within 1% of target)
      const totalUsedCapital = gridState.levels.reduce((sum, level) => {
        return sum + (level.side === 'buy' ? level.size * level.price : level.size * 50000)
      }, 0)
      
      const capitalDifference = Math.abs(totalUsedCapital - params.allocatedCapital)
      const percentDifference = capitalDifference / params.allocatedCapital
      assert.ok(percentDifference < 0.01, `Capital allocation should be within 1% of target (difference: ${percentDifference.toFixed(3)})`)
    })

    it('should validate grid parameters and reject invalid configurations', async () => {
      const baseParams: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 1000,
        baseAmount: 0.01,
        quoteAmount: 500,
        riskLevel: 0.5
      }

      // Test spacing too tight
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 0.05, // Too tight
          gridLevels: 10,
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, baseParams),
        /Grid spacing too tight/
      )

      // Test spacing too wide
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 60, // Too wide
          gridLevels: 10,
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, baseParams),
        /Grid spacing too wide/
      )

      // Test too few levels
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 2,
          gridLevels: 1, // Too few
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, baseParams),
        /Grid must have at least 2 levels/
      )

      // Test too many levels
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 2,
          gridLevels: 250, // Too many
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, baseParams),
        /Grid cannot have more than 200 levels/
      )

      // Test insufficient capital
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 2,
          gridLevels: 50,
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, {
          ...baseParams,
          allocatedCapital: 1 // Insufficient capital
        }),
        /Insufficient capital/
      )
    })

    it('should handle edge cases in level distribution', async () => {
      // Test odd number of levels
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 7, // Odd number
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 7000,
        baseAmount: 0.07,
        quoteAmount: 3500,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const gridState = gridManager.getGridState(result.gridId)

      assert.ok(gridState)
      assert.equal(gridState.levels.length, 7)

      const buyLevels = gridState.levels.filter(l => l.side === 'buy')
      const sellLevels = gridState.levels.filter(l => l.side === 'sell')

      // With 7 levels: 3 buy, 4 sell (or 3 buy, 4 sell)
      assert.ok(buyLevels.length === 3 || buyLevels.length === 4)
      assert.ok(sellLevels.length === 3 || sellLevels.length === 4)
      assert.equal(buyLevels.length + sellLevels.length, 7)
    })

    it('should maintain proper price spacing across all levels', async () => {
      const config: GridConfig = {
        gridSpacing: 4, // 4% spacing
        gridLevels: 8,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 8000,
        baseAmount: 0.08,
        quoteAmount: 4000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const gridState = gridManager.getGridState(result.gridId)

      assert.ok(gridState)

      // Check that spacing is consistent
      const centerPrice = result.centerPrice
      const expectedSpacing = config.gridSpacing / 100

      const buyLevels = gridState.levels
        .filter(l => l.side === 'buy')
        .sort((a, b) => b.price - a.price) // Sort buy levels from highest to lowest

      const sellLevels = gridState.levels
        .filter(l => l.side === 'sell')
        .sort((a, b) => a.price - b.price) // Sort sell levels from lowest to highest

      // Verify buy level spacing
      for (let i = 0; i < buyLevels.length; i++) {
        const expectedPrice = centerPrice * (1 - (i + 1) * expectedSpacing)
        const actualPrice = buyLevels[i]!.price
        const priceDifference = Math.abs(actualPrice - expectedPrice) / expectedPrice
        assert.ok(priceDifference < 0.001, `Buy level ${i} price should match expected spacing`)
      }

      // Verify sell level spacing
      for (let i = 0; i < sellLevels.length; i++) {
        const expectedPrice = centerPrice * (1 + (i + 1) * expectedSpacing)
        const actualPrice = sellLevels[i]!.price
        const priceDifference = Math.abs(actualPrice - expectedPrice) / expectedPrice
        assert.ok(priceDifference < 0.001, `Sell level ${i} price should match expected spacing`)
      }
    })

    it('should prevent extreme price range configurations', async () => {
      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 5000,
        baseAmount: 0.05,
        quoteAmount: 2500,
        riskLevel: 0.5
      }

      // Test configuration that would create extreme price ranges
      await assert.rejects(
        () => gridManager.createGrid({
          gridSpacing: 20, // 20% spacing
          gridLevels: 10, // 5 levels per side = 100% price deviation
          trailPercent: 0.5,
          minOrderSize: 0.001,
          maxOrderSize: 1.0,
          rebalanceThreshold: 0.1
        }, params),
        /Grid configuration would create extreme price ranges/
      )
    })

    it('should handle minimum capital requirements', async () => {
      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 5, // Very small capital
        baseAmount: 0.001,
        quoteAmount: 2.5,
        riskLevel: 0.5
      }

      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 4, // Should work with small capital
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      // This should work
      const result = await gridManager.createGrid(config, params)
      assert.ok(result.gridId)

      // But too many levels with small capital should fail
      await assert.rejects(
        () => gridManager.createGrid({
          ...config,
          gridLevels: 50, // Too many levels for small capital
          gridSpacing: 0.5  // Keep spacing small to avoid price range error
        }, params),
        /Insufficient capital/
      )
    })
  })

  describe('advanced trailing order integration', () => {
    it('should activate grid levels with proximity-based trailing', async () => {
      const trailingConfig: GridConfig['trailingOrderConfig'] = {
        activationThreshold: 2.0, // 2%
        enableProximityActivation: true,
        enableDirectFallback: true,
        activationTimeoutMs: 5000,
        trailUpdateThrottleMs: 100,
        activationStrategy: 'proximity'
      }

      const config: GridConfig = {
        gridSpacing: 3,
        gridLevels: 6,
        trailPercent: 1.0,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1,
        trailingOrderConfig: trailingConfig
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 6000,
        baseAmount: 0.06,
        quoteAmount: 3000,
        riskLevel: 0.5
      }

      const result = await gridManager.createGrid(config, params)
      const currentPrice = 49000 // Close to center price (50000) within 2% threshold

      await gridManager.activateNearbyGrids(result.gridId, currentPrice)

      // Verify trailing orders were created with activation prices
      const createOrderCalls = (mockTrailingOrderManager.createTrailingOrder as any).mock.calls
      assert.ok(createOrderCalls.length > 0, 'Should have created trailing orders')

      // Verify activation price was set for proximity activation
      const lastCall = createOrderCalls[createOrderCalls.length - 1]
      assert.ok(lastCall?.arguments[0].activationPrice !== undefined, 'Should set activation price for proximity activation')
    })

    it('should process market updates with trailing order monitoring', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 4,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1,
        trailingOrderConfig: {
          activationThreshold: 3.0, // Larger than grid spacing (2%) to ensure activation
          enableProximityActivation: true,
          enableDirectFallback: false,
          activationTimeoutMs: 10000,
          trailUpdateThrottleMs: 50,
          activationStrategy: 'price_approach'
        }
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('ETH-USD'),
        allocatedCapital: 4000,
        baseAmount: 2.0,
        quoteAmount: 2000,
        riskLevel: 0.4
      }

      const result = await gridManager.createGrid(config, params)
      
      // Activate nearby grids first
      await gridManager.activateNearbyGrids(result.gridId, 50000)

      // Add small delay to avoid throttling
      await new Promise(resolve => setTimeout(resolve, 60))

      // Process market update with volume data
      await gridManager.processMarketUpdate(result.gridId, 49500, 2500)

      // Verify trailing order manager was called for market updates
      const processUpdateCalls = (mockTrailingOrderManager.processMarketUpdate as any).mock.calls
      assert.ok(processUpdateCalls.length > 0, 'Should have processed market updates')
    })

    it('should handle different activation strategies', async () => {
      const strategies: Array<GridTrailingOrderConfig['activationStrategy']> = [
        'proximity', 'price_approach', 'volume_spike', 'combined'
      ]

      for (const strategy of strategies) {
        const config: GridConfig = {
          gridSpacing: 4,
          gridLevels: 4,
          trailPercent: 0.8,
          minOrderSize: 0.01,
          maxOrderSize: 0.5,
          rebalanceThreshold: 0.12,
          trailingOrderConfig: {
            activationThreshold: 3.0,
            enableProximityActivation: true,
            enableDirectFallback: true,
            activationTimeoutMs: 8000,
            trailUpdateThrottleMs: 200,
            activationStrategy: strategy
          }
        }

        const params: GridInitializationParams = {
          symbol: toStockSymbol(`TEST-${strategy?.toUpperCase()}`),
          allocatedCapital: 2000,
          baseAmount: 0.02,
          quoteAmount: 1000,
          riskLevel: 0.3
        }

        // Should not throw for any strategy
        const result = await gridManager.createGrid(config, params)
        assert.ok(result.gridId, `Should create grid with ${strategy} strategy`)

        await gridManager.activateNearbyGrids(result.gridId, 50000)
        await gridManager.processMarketUpdate(result.gridId, 49000, 1500)
      }
    })

    it('should provide sensible defaults when trailing config is not specified', async () => {
      const config: GridConfig = {
        gridSpacing: 2.5,
        gridLevels: 8,
        trailPercent: 0.6,
        minOrderSize: 0.005,
        maxOrderSize: 2.0,
        rebalanceThreshold: 0.08
        // No trailingOrderConfig specified
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('DEFAULT-CONFIG'),
        allocatedCapital: 5000,
        baseAmount: 0.05,
        quoteAmount: 2500,
        riskLevel: 0.6
      }

      // Should work with default configuration
      const result = await gridManager.createGrid(config, params)
      assert.ok(result.gridId, 'Should create grid with default trailing config')

      // Should activate nearby grids without errors
      await gridManager.activateNearbyGrids(result.gridId, 50000)
      
      const gridState = gridManager.getGridState(result.gridId)
      assert.ok(gridState, 'Grid state should be available')
      
      // Should have activated some levels
      const activeLevels = gridState.levels.filter(l => l.isActive)
      assert.ok(activeLevels.length > 0, 'Should have activated some grid levels')
    })

    it('should handle activation threshold edge cases', async () => {
      const config: GridConfig = {
        gridSpacing: 1,
        gridLevels: 6,
        trailPercent: 0.3,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.05,
        trailingOrderConfig: {
          activationThreshold: 10.0, // Very wide threshold
          enableProximityActivation: true,
          enableDirectFallback: true,
          activationTimeoutMs: 1000,
          trailUpdateThrottleMs: 10,
          activationStrategy: 'proximity'
        }
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('WIDE-THRESHOLD'),
        allocatedCapital: 3000,
        baseAmount: 0.03,
        quoteAmount: 1500,
        riskLevel: 0.7
      }

      const result = await gridManager.createGrid(config, params)
      
      // With 10% threshold, most levels should be within range
      await gridManager.activateNearbyGrids(result.gridId, 50000)
      
      const gridState = gridManager.getGridState(result.gridId)
      assert.ok(gridState, 'Grid state should be available')
      
      const activeLevels = gridState.levels.filter(l => l.isActive)
      assert.ok(activeLevels.length >= 4, 'Should activate most levels with wide threshold')
    })
  })

  describe('self-tuning grid spacing integration', () => {
    it('should use self-tuning spacing when calculating optimal spacing', async () => {
      const testCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        testCandles.push({
          timestamp: (epochDateNow() - (20 - i) * 60000) as any,
          open: 50000 + i * 100,
          high: 50000 + i * 100 + 50,
          low: 50000 + i * 100 - 50,
          close: 50000 + i * 100,
          volume: 1000
        })
      }

      const result = await gridManager.calculateOptimalSpacing(testCandles, 50000)
      
      assert.ok(result.spacing > 0)
      assert.ok(result.confidence >= 0 && result.confidence <= 1)
      assert.ok(result.reasoning.length > 0)
    })

    it('should record performance when grids are cancelled', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 6,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 5000,
        baseAmount: 0.05,
        quoteAmount: 2500,
        riskLevel: 0.4
      }

      const result = await gridManager.createGrid(config, params)
      
      // Get initial performance stats
      const initialStats = gridManager.getPerformanceStatistics()
      const initialHistoryCount = initialStats.historyCount

      // Cancel the grid (this should record performance)
      await gridManager.cancelGrid(result.gridId, 'Test performance recording')

      // Check that performance was recorded
      const finalStats = gridManager.getPerformanceStatistics()
      assert.equal(finalStats.historyCount, initialHistoryCount + 1)
    })

    it('should provide performance statistics', () => {
      const stats = gridManager.getPerformanceStatistics()
      
      assert.ok(typeof stats.historyCount === 'number')
      assert.ok(typeof stats.averageSpacing === 'number')
      assert.ok(typeof stats.bestPerformingSpacing === 'number')
      assert.ok(typeof stats.currentMetrics === 'object')
    })

    it('should reset performance history', () => {
      // Reset history
      gridManager.resetPerformanceHistory()
      
      // Check that history was reset
      const resetStats = gridManager.getPerformanceStatistics()
      assert.equal(resetStats.historyCount, 0)
      assert.equal(resetStats.averageSpacing, 0)
    })
  })

  describe('persistence integration', () => {
    it('should provide persistence statistics', () => {
      const stats = gridManager.getPersistenceStats()
      
      assert.ok(typeof stats.activeGrids === 'number')
      assert.ok(typeof stats.totalCreated === 'number')
      assert.ok(typeof stats.totalCancelled === 'number')
      assert.ok(typeof stats.systemUptime === 'number')
      assert.equal(stats.activeGrids, 0) // No grids created yet
      assert.equal(stats.totalCreated, 0)
      assert.equal(stats.totalCancelled, 0)
    })

    it('should track grid creation and cancellation counts', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 4,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 5000,
        baseAmount: 0.05,
        quoteAmount: 2500,
        riskLevel: 0.4
      }

      // Create a grid
      const result = await gridManager.createGrid(config, params)
      
      let stats = gridManager.getPersistenceStats()
      assert.equal(stats.activeGrids, 1)
      assert.equal(stats.totalCreated, 1)
      assert.equal(stats.totalCancelled, 0)

      // Cancel the grid
      await gridManager.cancelGrid(result.gridId, 'Test cancellation')
      
      stats = gridManager.getPersistenceStats()
      assert.equal(stats.activeGrids, 0) // Grid is deactivated but not removed
      assert.equal(stats.totalCreated, 1)
      assert.equal(stats.totalCancelled, 1)
    })

    it('should create snapshots with current state', async () => {
      const config: GridConfig = {
        gridSpacing: 3,
        gridLevels: 6,
        trailPercent: 1.0,
        minOrderSize: 0.01,
        maxOrderSize: 0.5,
        rebalanceThreshold: 0.08
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('ETH-USD'),
        allocatedCapital: 3000,
        baseAmount: 1.0,
        quoteAmount: 1500,
        riskLevel: 0.6
      }

      // Create a grid
      const result = await gridManager.createGrid(config, params)
      
      // Create a snapshot
      const snapshot = await gridManager.createSnapshot()
      
      assert.equal(snapshot.version, '1.0.0')
      assert.ok(snapshot.timestamp > 0)
      assert.equal(Object.keys(snapshot.activeGrids).length, 1)
      assert.ok(snapshot.activeGrids[result.gridId])
      assert.equal(snapshot.metadata.totalGridsCreated, 1)
      assert.equal(snapshot.metadata.totalGridsCancelled, 0)
      
      // Verify grid state in snapshot
      const gridInSnapshot = snapshot.activeGrids[result.gridId]!
      assert.equal(gridInSnapshot.symbol, 'ETH-USD')
      assert.equal(gridInSnapshot.centerPrice, 50000)
      assert.equal(gridInSnapshot.allocatedCapital, 3000)
      assert.equal(gridInSnapshot.isActive, true)
    })

    it('should handle manual state saving', async () => {
      // Should not throw even with no grids
      await gridManager.saveState()
      
      // Create a grid and save again
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 4,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 2000,
        baseAmount: 0.02,
        quoteAmount: 1000,
        riskLevel: 0.3
      }

      await gridManager.createGrid(config, params)
      
      // Should save successfully
      await gridManager.saveState()
    })

    it('should support clean shutdown', async () => {
      const config: GridConfig = {
        gridSpacing: 2,
        gridLevels: 4,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      }

      const params: GridInitializationParams = {
        symbol: toStockSymbol('BTC-USD'),
        allocatedCapital: 2000,
        baseAmount: 0.02,
        quoteAmount: 1000,
        riskLevel: 0.3
      }

      // Create a grid
      const result = await gridManager.createGrid(config, params)
      
      // Shutdown should cancel all grids and save state
      await gridManager.shutdown()
      
      // Grid should be cancelled
      const gridState = gridManager.getGridState(result.gridId)
      assert.ok(gridState)
      assert.equal(gridState.isActive, false)
      
      // Stats should reflect cancellation
      const stats = gridManager.getPersistenceStats()
      assert.equal(stats.totalCancelled, 1)
    })

    it('should handle persistence initialization', async () => {
      // Should be able to initialize persistence
      const recoveryInfo = await gridManager.initializePersistence()
      
      // Should return recovery info (likely empty for new system)
      assert.ok(typeof recoveryInfo.success === 'boolean')
      assert.ok(typeof recoveryInfo.recoveredGrids === 'number')
      assert.ok(typeof recoveryInfo.failedGrids === 'number')
      assert.ok(Array.isArray(recoveryInfo.errors))
      assert.ok(Array.isArray(recoveryInfo.warnings))
    })
  })
})