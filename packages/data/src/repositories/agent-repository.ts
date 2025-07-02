import type { AgentType, EpochDate, StockSymbol } from '@trdr/shared'
import { epochDateNow, type IsoDate, isoDateNow, isoToEpoch, toIsoDate } from '@trdr/shared'
import type { ConnectionManager } from '../db/connection-manager'
import type { AgentSignal } from '../types/agents'
import { BaseRepository } from './base-repository'

/**
 * Database agent decision dto
 */
interface AgentDecisionDto {
  id: number
  agent_id: string
  agent_type: string
  symbol: string
  action: string
  confidence: number
  trail_distance?: number
  reasoning: unknown
  market_context?: unknown
  timestamp: IsoDate
  created_at: IsoDate
}

/**
 * Agent consensus data
 */
export interface AgentConsensus {
  readonly id: number
  readonly symbol: StockSymbol
  readonly decision: 'buy' | 'sell' | 'hold'
  readonly confidence: number
  readonly dissent: number
  readonly votes: Array<{
    agentId: string
    action: string
    confidence: number
  }>
  readonly timestamp: EpochDate
}

/**
 * Database agent consensus dto
 */
interface AgentConsensusDto {
  id: number
  symbol: string
  decision: string
  confidence: number
  dissent: number
  votes: unknown
  timestamp: IsoDate
  created_at: IsoDate
}

/**
 * Agent checkpoint data
 */
export interface AgentCheckpoint {
  readonly id: string
  readonly type: string
  readonly version: number
  readonly state: Record<string, unknown>
  readonly metadata?: Record<string, unknown>
  readonly createdAt: EpochDate
}

/**
 * Database checkpoint dto
 */
interface CheckpointDto {
  id: string
  type: string
  version: number
  state: unknown
  metadata?: unknown
  created_at: IsoDate
}

/**
 * Repository for agent decisions and checkpoints
 */
export class AgentRepository extends BaseRepository<AgentDecisionDto> {
  protected readonly tableName = 'agent_decisions'
  private readonly consensusTableName = 'agent_consensus'
  private readonly checkpointsTableName = 'checkpoints'

  constructor(connectionManager: ConnectionManager) {
    super(connectionManager)
  }

  /**
   * Record an agent decision
   */
  async recordDecision(decision: AgentSignal & { agentType: AgentType }): Promise<void> {
    const model: Partial<AgentDecisionDto> = {
      agent_id: decision.agentId,
      agent_type: decision.agentType,
      symbol: decision.symbol,
      action: decision.action,
      confidence: decision.confidence,
      trail_distance: decision.trailDistance,
      reasoning: JSON.stringify(decision.reasoning),
      market_context: decision.marketContext ? JSON.stringify(decision.marketContext) : null,
      timestamp: toIsoDate(decision.timestamp),
    }

    await this.insert(model)
  }

  /**
   * Record multiple decisions in batch
   */
  async recordDecisionsBatch(decisions: Array<AgentSignal & { agentType: AgentType }>): Promise<void> {
    const models: Partial<AgentDecisionDto>[] = decisions.map(decision => ({
      agent_id: decision.agentId,
      agent_type: decision.agentType,
      symbol: decision.symbol,
      action: decision.action,
      confidence: decision.confidence,
      trail_distance: decision.trailDistance,
      reasoning: JSON.stringify(decision.reasoning),
      market_context: decision.marketContext ? JSON.stringify(decision.marketContext) : null,
      timestamp: toIsoDate(decision.timestamp),
    }))

    await this.insertBatch(models)
  }

  /**
   * Get agent decisions within a time range
   */
  async getDecisions(
    agentId?: string,
    symbol?: string,
    startTime?: EpochDate | Date,
    endTime?: EpochDate | Date,
    limit?: number,
  ): Promise<AgentSignal[]> {
    const whereParts: string[] = []
    const params: unknown[] = []

    if (agentId) {
      whereParts.push('agent_id = ?')
      params.push(agentId)
    }

    if (symbol) {
      whereParts.push('symbol = ?')
      params.push(symbol)
    }

    if (startTime) {
      const dt = startTime instanceof Date ? toIsoDate(startTime) : toIsoDate(startTime)
      whereParts.push('timestamp >= ?')
      params.push(dt)
    }

    if (endTime) {
      const dt = endTime instanceof Date ? toIsoDate(endTime) : toIsoDate(endTime)
      whereParts.push('timestamp <= ?')
      params.push(dt)
    }

    const where = whereParts.length > 0 ? whereParts.join(' AND ') : undefined
    const models = await this.findMany(where, params, 'timestamp DESC', limit)

    return models.map(model => this.dtoToDecision(model))
  }

  /**
   * Record agent consensus
   */
  async recordConsensus(consensus: Omit<AgentConsensus, 'id'>): Promise<void> {
    const model: Partial<AgentConsensusDto> = {
      symbol: consensus.symbol,
      decision: consensus.decision,
      confidence: consensus.confidence,
      dissent: consensus.dissent,
      votes: JSON.stringify(consensus.votes),
      timestamp: toIsoDate(consensus.timestamp),
    }

    const fields = Object.keys(model)
    const values = Object.values(model)
    const placeholders = fields.map(() => '?').join(', ')

    const sql = `INSERT INTO ${this.consensusTableName} (${fields.join(', ')}) VALUES (${placeholders})`
    await this.connectionManager.execute(sql, values)
  }

  /**
   * Get consensus history
   */
  async getConsensusHistory(
    symbol: string,
    startTime?: EpochDate | Date,
    endTime?: EpochDate | Date,
    limit?: number,
  ): Promise<AgentConsensus[]> {
    let sql = `SELECT * FROM ${this.consensusTableName} WHERE symbol = ?`
    const params: unknown[] = [symbol]

    if (startTime) {
      const dt = startTime instanceof Date ? toIsoDate(startTime) : toIsoDate(startTime)
      sql += ' AND timestamp >= ?'
      params.push(dt)
    }

    if (endTime) {
      const dt = endTime instanceof Date ? toIsoDate(endTime) : toIsoDate(endTime)
      sql += ' AND timestamp <= ?'
      params.push(dt)
    }

    sql += ' ORDER BY timestamp DESC'

    if (limit) {
      sql += ` LIMIT ${limit}`
    }

    const models = await this.query<AgentConsensusDto>(sql, params)
    return models.map(model => this.dtoToConsensus(model))
  }

  /**
   * Save an agent checkpoint
   */
  async saveCheckpoint(checkpoint: Omit<AgentCheckpoint, 'createdAt'>): Promise<void> {
    const model: Partial<CheckpointDto> = {
      id: checkpoint.id,
      type: checkpoint.type,
      version: checkpoint.version,
      state: JSON.stringify(checkpoint.state),
      metadata: checkpoint.metadata ? JSON.stringify(checkpoint.metadata) : null,
      created_at: isoDateNow(),
    }

    const fields = Object.keys(model)
    const values = Object.values(model)
    const placeholders = fields.map(() => '?').join(', ')

    const sql = `INSERT INTO ${this.checkpointsTableName} (${fields.join(', ')}) VALUES (${placeholders})`
    await this.connectionManager.execute(sql, values)
  }

  /**
   * Load the latest checkpoint
   */
  async loadLatestCheckpoint(type: string): Promise<AgentCheckpoint | null> {
    const sql = `
      SELECT * FROM ${this.checkpointsTableName}
      WHERE type = ?
      ORDER BY created_at DESC
      LIMIT 1
    `

    const models = await this.query<CheckpointDto>(sql, [type])
    return models.length > 0 && models[0] ? this.dtoToCheckpoint(models[0]) : null
  }

  /**
   * Load a specific checkpoint
   */
  async loadCheckpoint(id: string): Promise<AgentCheckpoint | null> {
    const sql = `SELECT * FROM ${this.checkpointsTableName} WHERE id = ?`
    const models = await this.query<CheckpointDto>(sql, [id])
    return models.length > 0 && models[0] ? this.dtoToCheckpoint(models[0]) : null
  }

  /**
   * List checkpoints
   */
  async listCheckpoints(
    type?: string,
    limit = 10,
  ): Promise<AgentCheckpoint[]> {
    let sql = `SELECT * FROM ${this.checkpointsTableName}`
    const params: unknown[] = []

    if (type) {
      sql += ' WHERE type = ?'
      params.push(type)
    }

    sql += ` ORDER BY created_at DESC LIMIT ${limit}`

    const models = await this.query<CheckpointDto>(sql, params)
    return models.map(model => this.dtoToCheckpoint(model))
  }

  /**
   * Get agent performance statistics
   */
  async getAgentStats(
    agentId: string,
    days = 30,
  ): Promise<{
    totalDecisions: number
    avgConfidence: number
    actionDistribution: Record<string, number>
    accuracyRate?: number
  }> {
    const startTime = epochDateNow() - (days * 24 * 60 * 60 * 1000)

    // Get overall stats
    const overallStats = await this.query<{
      total_decisions: number
      avg_confidence: number
    }>(`
      SELECT 
        COUNT(*) as total_decisions,
        AVG(confidence) as avg_confidence
      FROM ${this.tableName}
      WHERE agent_id = ?
        AND timestamp >= ?
    `, [agentId, toIsoDate(startTime)])

    // Get action distribution
    const actionStats = await this.query<{
      action: string
      action_count: number
    }>(`
      SELECT 
        action,
        COUNT(*) as action_count
      FROM ${this.tableName}
      WHERE agent_id = ?
        AND timestamp >= ?
      GROUP BY action
    `, [agentId, toIsoDate(startTime)])

    const totalDecisions = Number(overallStats[0]?.total_decisions || 0)
    const avgConfidence = Number(overallStats[0]?.avg_confidence || 0)

    const actionDistribution = actionStats.reduce((acc, row) => {
      acc[row.action] = Number(row.action_count)
      return acc
    }, {} as Record<string, number>)

    return {
      totalDecisions,
      avgConfidence,
      actionDistribution,
    }
  }

  /**
   * Cleanup old data
   */
  async cleanup(daysToKeep = 90): Promise<{
    decisionsDeleted: number
    consensusDeleted: number
    checkpointsDeleted: number
  }> {
    const cutoffDate = epochDateNow() - (daysToKeep * 24 * 60 * 60 * 1000)

    // Count before deletion
    const decisionsBefore = await this.count()
    const consensusBefore = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.consensusTableName}`,
    ).then(r => Number(r[0]?.count || 0))
    const checkpointsBefore = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.checkpointsTableName}`,
    ).then(r => Number(r[0]?.count || 0))

    // Delete old data
    await this.delete('timestamp < ?', [toIsoDate(cutoffDate)])
    await this.connectionManager.execute(
      `DELETE FROM ${this.consensusTableName} WHERE timestamp < ?`,
      [toIsoDate(cutoffDate)],
    )
    await this.connectionManager.execute(
      `DELETE FROM ${this.checkpointsTableName} WHERE created_at < ?`,
      [toIsoDate(cutoffDate)],
    )

    // Count after deletion
    const decisionsAfter = await this.count()
    const consensusAfter = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.consensusTableName}`,
    ).then(r => Number(r[0]?.count || 0))
    const checkpointsAfter = await this.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.checkpointsTableName}`,
    ).then(r => Number(r[0]?.count || 0))

    return {
      decisionsDeleted: decisionsBefore - decisionsAfter,
      consensusDeleted: consensusBefore - consensusAfter,
      checkpointsDeleted: checkpointsBefore - checkpointsAfter,
    }
  }

  /**
   * Convert database dto to AgentSignal
   */
  private dtoToDecision(model: AgentDecisionDto): AgentSignal {
    return {
      agentId: model.agent_id,
      symbol: model.symbol,
      action: model.action as 'TRAIL_BUY' | 'TRAIL_SELL' | 'HOLD',
      confidence: model.confidence,
      trailDistance: model.trail_distance || 0,
      reasoning: JSON.parse(model.reasoning as string) as Record<string, unknown>,
      marketContext: model.market_context ? JSON.parse(model.market_context as string) as Record<string, unknown> : undefined,
      timestamp: isoToEpoch(model.timestamp),
    }
  }

  /**
   * Convert database dto to AgentConsensus
   */
  private dtoToConsensus(model: AgentConsensusDto): AgentConsensus {
    return {
      id: model.id,
      symbol: model.symbol,
      decision: model.decision as 'buy' | 'sell' | 'hold',
      confidence: model.confidence,
      dissent: model.dissent,
      votes: JSON.parse(model.votes as string) as Array<{ agentId: string; action: string; confidence: number }>,
      timestamp: isoToEpoch(model.timestamp),
    }
  }

  /**
   * Convert database dto to AgentCheckpoint
   */
  private dtoToCheckpoint(model: CheckpointDto): AgentCheckpoint {
    return {
      id: model.id,
      type: model.type,
      version: model.version,
      state: JSON.parse(model.state as string) as Record<string, unknown>,
      metadata: model.metadata ? JSON.parse(model.metadata as string) as Record<string, unknown> : undefined,
      createdAt: isoToEpoch(model.created_at),
    }
  }
}
