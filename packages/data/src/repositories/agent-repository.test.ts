import { epochDateNow, toEpochDate } from '@trdr/shared'
import assert from 'node:assert/strict'
import { afterEach, beforeEach, describe, it } from 'node:test'
import type { ConnectionManager } from '../db/connection-manager'
import { createConnectionManager } from '../db/connection-manager'
import { SCHEMA_STATEMENTS } from '../db/schema'
import type { AgentSignal } from '../types/agents'
import { AgentRepository } from './agent-repository'

describe('AgentRepository', () => {
  let repository: AgentRepository
  let connectionManager: ConnectionManager

  beforeEach(async () => {
    connectionManager = createConnectionManager({ databasePath: ':memory:' })
    await connectionManager.initialize()

    // Create tables
    await connectionManager.execute(SCHEMA_STATEMENTS.agent_decisions)

    await connectionManager.execute(SCHEMA_STATEMENTS.agent_consensus)

    await connectionManager.execute(SCHEMA_STATEMENTS.checkpoints)

    repository = new AgentRepository(connectionManager)
  })

  afterEach(async () => {
    await connectionManager.close()
  })

  describe('Agent Decisions', () => {
    const testDecision: AgentSignal & { agentType: 'momentum' } = {
      agentId: 'momentum-agent-1',
      agentType: 'momentum',
      symbol: 'BTC-USD',
      action: 'TRAIL_BUY',
      confidence: 0.85,
      trailDistance: 0.02,
      reasoning: { momentum: 'strong', trend: 'up' },
      marketContext: { volume: 'high', volatility: 'medium' },
      timestamp: epochDateNow(),
    }

    it('should record agent decision', async () => {
      await repository.recordDecision(testDecision)

      const decisions = await repository.getDecisions(testDecision.agentId)

      assert.equal(decisions.length, 1)
      assert.ok(decisions[0])
      assert.equal(decisions[0].agentId, testDecision.agentId)
      assert.equal(decisions[0].action, testDecision.action)
      assert.equal(decisions[0].confidence, testDecision.confidence)
    })

    it('should record decisions in batch', async () => {
      const decisions = Array.from({ length: 5 }, (_, i): AgentSignal & { agentType: 'momentum' } => ({
        agentId: `batch-agent-${i}`,
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: i % 3 === 0 ? 'TRAIL_BUY' : i % 3 === 1 ? 'TRAIL_SELL' : 'HOLD',
        confidence: 0.7 + i * 0.05,
        trailDistance: 0.01 + i * 0.005,
        reasoning: { batch: i },
        timestamp: toEpochDate(Date.now() - i * 1000)
      }))

      await repository.recordDecisionsBatch(decisions)

      const allDecisions = await repository.getDecisions()
      assert.equal(allDecisions.length, 5)
    })

    it('should filter decisions by agent', async () => {
      const agentId = 'filter-agent'
      const otherAgentId = 'other-agent'

      await repository.recordDecision({
        ...testDecision,
        agentId,
        timestamp: toEpochDate(Date.now() - 2000)
      })

      await repository.recordDecision({
        ...testDecision,
        agentId: otherAgentId,
        timestamp: toEpochDate(Date.now() - 1000)
      })

      const agentDecisions = await repository.getDecisions(agentId)
      assert.equal(agentDecisions.length, 1)
      assert.ok(agentDecisions[0])
      assert.equal(agentDecisions[0].agentId, agentId)
    })

    it('should filter decisions by symbol', async () => {
      await repository.recordDecision({
        ...testDecision,
        symbol: 'BTC-USD',
        timestamp: toEpochDate(Date.now() - 2000)
      })

      await repository.recordDecision({
        ...testDecision,
        symbol: 'ETH-USD',
        timestamp: toEpochDate(Date.now() - 1000)
      })

      const btcDecisions = await repository.getDecisions(undefined, 'BTC-USD')
      assert.equal(btcDecisions.length, 1)
      assert.ok(btcDecisions[0])
      assert.equal(btcDecisions[0].symbol, 'BTC-USD')
    })

    it('should filter decisions by time range', async () => {
      const now = Date.now()

      await repository.recordDecision({
        ...testDecision,
        timestamp: toEpochDate(now - 7200000), // 2 hours ago
      })

      await repository.recordDecision({
        ...testDecision,
        timestamp: toEpochDate(now - 1800000), // 30 minutes ago
      })

      const recentDecisions = await repository.getDecisions(
        undefined,
        undefined,
        toEpochDate(now - 3600000), // 1 hour ago
        epochDateNow()
      )

      assert.equal(recentDecisions.length, 1)
    })
  })

  describe('Agent Consensus', () => {
    it('should record consensus', async () => {
      const consensus = {
        symbol: 'BTC-USD',
        decision: 'buy' as const,
        confidence: 0.75,
        dissent: 0.15,
        votes: [
          { agentId: 'agent-1', action: 'buy', confidence: 0.8 },
          { agentId: 'agent-2', action: 'buy', confidence: 0.7 },
          { agentId: 'agent-3', action: 'sell', confidence: 0.6 },
        ],
        timestamp: epochDateNow()
      }

      await repository.recordConsensus(consensus)

      const history = await repository.getConsensusHistory('BTC-USD')

      assert.equal(history.length, 1)
      assert.ok(history[0])
      assert.equal(history[0].decision, 'buy')
      assert.equal(history[0].confidence, 0.75)
      assert.equal(history[0].votes.length, 3)
    })

    it('should get consensus history with time filter', async () => {
      const now = epochDateNow()
      const past = toEpochDate(now - 3600000)

      await repository.recordConsensus({
        symbol: 'BTC-USD',
        decision: 'buy',
        confidence: 0.8,
        dissent: 0.1,
        votes: [],
        timestamp: past,
      })

      await repository.recordConsensus({
        symbol: 'BTC-USD',
        decision: 'sell',
        confidence: 0.7,
        dissent: 0.2,
        votes: [],
        timestamp: now,
      })

      const recentHistory = await repository.getConsensusHistory(
        'BTC-USD',
        toEpochDate(now - 1800000),
        now,
      )

      assert.equal(recentHistory.length, 1)
      assert.ok(recentHistory[0])
      assert.equal(recentHistory[0].decision, 'sell')
    })
  })

  describe('Checkpoints', () => {
    beforeEach(async () => {
      // Clean up any existing checkpoints from previous tests
      await connectionManager.execute('DELETE FROM checkpoints')
    })

    const testCheckpoint = {
      id: 'test-checkpoint-1',
      type: 'agent-state',
      version: 1,
      state: {
        agentId: 'checkpoint-agent-1',
        position: 'long',
        entryPrice: 50000,
        parameters: {
          trailDistance: 0.02,
          confidence: 0.8,
        },
      },
      metadata: {
        timestamp: epochDateNow(),
        source: 'test',
      },
    }

    it('should save checkpoint', async () => {
      await repository.saveCheckpoint(testCheckpoint)

      const loaded = await repository.loadCheckpoint(testCheckpoint.id)

      assert.ok(loaded)
      assert.equal(loaded.id, testCheckpoint.id)
      assert.equal(loaded.type, testCheckpoint.type)
      assert.equal(loaded.version, testCheckpoint.version)
      assert.deepEqual(loaded.state, testCheckpoint.state)
    })

    it('should load latest checkpoint by type', async () => {
      // Save multiple checkpoints with slight delay
      await repository.saveCheckpoint({
        ...testCheckpoint,
        id: 'checkpoint-v1',
        version: 1,
      })

      // Larger delay to ensure different created_at timestamp
      await new Promise(resolve => setTimeout(resolve, 50))

      await repository.saveCheckpoint({
        ...testCheckpoint,
        id: 'checkpoint-v2',
        version: 2,
      })

      const latest = await repository.loadLatestCheckpoint('agent-state')

      assert.ok(latest)
      assert.equal(latest.version, 2)
    })

    it('should list checkpoints', async () => {
      // Save checkpoints of different types
      await repository.saveCheckpoint({
        ...testCheckpoint,
        id: 'agent-1',
        type: 'agent-state',
      })

      await repository.saveCheckpoint({
        ...testCheckpoint,
        id: 'system-1',
        type: 'system-state',
      })

      const agentCheckpoints = await repository.listCheckpoints('agent-state')
      assert.equal(agentCheckpoints.length, 1)
      assert.ok(agentCheckpoints[0])
      assert.equal(agentCheckpoints[0].type, 'agent-state')

      const allCheckpoints = await repository.listCheckpoints()
      assert.equal(allCheckpoints.length, 2)
    })

    it('should return null for non-existent checkpoint', async () => {
      const checkpoint = await repository.loadCheckpoint('non-existent')
      assert.equal(checkpoint, null)
    })
  })

  describe('Agent Statistics', () => {
    beforeEach(async () => {
      const agentId = 'stats-agent'

      // Record various decisions
      const decisions = [
        { action: 'TRAIL_BUY' as const, confidence: 0.8 },
        { action: 'TRAIL_BUY' as const, confidence: 0.7 },
        { action: 'TRAIL_SELL' as const, confidence: 0.9 },
        { action: 'HOLD' as const, confidence: 0.6 },
        { action: 'TRAIL_BUY' as const, confidence: 0.75 },
      ]

      for (let i = 0; i < decisions.length; i++) {
        const decision = decisions[i]
        if (decision) {
          await repository.recordDecision({
            agentId,
            agentType: 'momentum',
            symbol: 'BTC-USD',
            action: decision.action,
            confidence: decision.confidence,
            trailDistance: 0.02,
            reasoning: { index: i },
            timestamp: toEpochDate(Date.now() - (decisions.length - i) * 3600000),
          })
        }
      }
    })

    it('should calculate agent statistics', async () => {
      const stats = await repository.getAgentStats('stats-agent', 30)

      assert.equal(stats.totalDecisions, 5)
      assert.ok(stats.avgConfidence > 0.7)
      assert.ok(stats.actionDistribution)
      assert.equal(stats.actionDistribution['TRAIL_BUY'], 3)
      assert.equal(stats.actionDistribution['TRAIL_SELL'], 1)
      assert.equal(stats.actionDistribution['HOLD'], 1)
    })
  })

  describe('Cleanup Operations', () => {
    it('should cleanup old data', async () => {
      const oldDate = toEpochDate(Date.now() - 100 * 86400000)
      const recentDate = toEpochDate(Date.now() - 3600000)

      // Add old data
      await repository.recordDecision({
        agentId: 'cleanup-agent',
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: 'HOLD',
        confidence: 0.5,
        trailDistance: 0,
        reasoning: { old: true },
        timestamp: toEpochDate(oldDate),
      })

      await repository.recordConsensus({
        symbol: 'BTC-USD',
        decision: 'hold',
        confidence: 0.5,
        dissent: 0.3,
        votes: [],
        timestamp: toEpochDate(oldDate),
      })

      await repository.saveCheckpoint({
        id: 'old-checkpoint',
        type: 'test',
        version: 1,
        state: { old: true },
      })

      // Add recent data
      await repository.recordDecision({
        agentId: 'cleanup-agent',
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: 'TRAIL_BUY',
        confidence: 0.8,
        trailDistance: 0.02,
        reasoning: { recent: true },
        timestamp: toEpochDate(recentDate),
      })

      const cleanup = await repository.cleanup(90)

      assert.ok(cleanup.decisionsDeleted > 0)
      assert.ok(cleanup.consensusDeleted > 0)
      assert.ok(cleanup.checkpointsDeleted >= 0)

      // Verify recent data remains
      const recentDecisions = await repository.getDecisions('cleanup-agent')
      assert.equal(recentDecisions.length, 1)
      assert.ok(recentDecisions[0])
      assert.equal(recentDecisions[0].timestamp, recentDate)
    })
  })
})
