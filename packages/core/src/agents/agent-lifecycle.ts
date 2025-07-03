import type { ITradeAgent } from './types'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import type { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Agent lifecycle phases
 */
export enum AgentPhase {
  /** Agent is created but not initialized */
  CREATED = 'created',
  
  /** Agent is being initialized */
  INITIALIZING = 'initializing',
  
  /** Agent is warming up (loading models, calibrating) */
  WARMING_UP = 'warming_up',
  
  /** Agent is ready but not active */
  READY = 'ready',
  
  /** Agent is actively processing */
  ACTIVE = 'active',
  
  /** Agent is cooling down (graceful shutdown) */
  COOLING_DOWN = 'cooling_down',
  
  /** Agent has been paused */
  PAUSED = 'paused',
  
  /** Agent encountered an error */
  ERROR = 'error',
  
  /** Agent has been retired */
  RETIRED = 'retired'
}

/**
 * Agent state information
 */
export interface AgentState {
  phase: AgentPhase
  phaseStartTime: number
  lastActivityTime: number
  errorCount: number
  restartCount: number
  configuration: Record<string, unknown>
  metadata: {
    version: string
    deploymentTime: number
    healthScore: number
  }
}

/**
 * Agent replacement information
 */
export interface AgentReplacement {
  oldAgentId: string
  newAgentId: string
  reason: string
  migrationData?: Record<string, unknown>
}

/**
 * Health check result
 */
export interface HealthCheckResult {
  healthy: boolean
  score: number // 0-1
  issues: string[]
  metrics: {
    responseTime: number
    memoryUsage: number
    errorRate: number
  }
}

/**
 * Configuration for agent lifecycle manager
 */
export interface AgentLifecycleConfig {
  /** Warm-up duration in milliseconds */
  warmUpDuration: number
  
  /** Cool-down duration in milliseconds */
  coolDownDuration: number
  
  /** Health check interval in milliseconds */
  healthCheckInterval: number
  
  /** Maximum error count before agent is retired */
  maxErrorCount: number
  
  /** Maximum restart attempts */
  maxRestartAttempts: number
  
  /** Timeout for phase transitions */
  phaseTransitionTimeout: number
  
  /** Enable automatic recovery */
  enableAutoRecovery: boolean
  
  /** Enable configuration persistence */
  enableConfigPersistence: boolean
}

/**
 * Default lifecycle configuration
 */
const DEFAULT_LIFECYCLE_CONFIG: AgentLifecycleConfig = {
  warmUpDuration: 5000, // 5 seconds
  coolDownDuration: 3000, // 3 seconds
  healthCheckInterval: 60000, // 1 minute
  maxErrorCount: 10,
  maxRestartAttempts: 3,
  phaseTransitionTimeout: 30000, // 30 seconds
  enableAutoRecovery: true,
  enableConfigPersistence: true
}

/**
 * Manages agent lifecycle from creation to retirement
 */
export class AgentLifecycleManager {
  private readonly agentStates = new Map<string, AgentState>()
  private readonly healthCheckTimers = new Map<string, NodeJS.Timeout>()
  private readonly config: AgentLifecycleConfig
  private readonly configStore = new Map<string, Record<string, unknown>>()
  
  constructor(
    private readonly eventBus: EventBus,
    config?: Partial<AgentLifecycleConfig>,
    private readonly logger?: Logger
  ) {
    this.config = { ...DEFAULT_LIFECYCLE_CONFIG, ...config }
    this.subscribeToEvents()
  }
  
  /**
   * Register an agent for lifecycle management
   */
  registerAgent(agent: ITradeAgent, initialConfig?: Record<string, unknown>): void {
    const agentId = agent.metadata.id
    
    if (this.agentStates.has(agentId)) {
      throw new Error(`Agent ${agentId} already registered`)
    }
    
    const state: AgentState = {
      phase: AgentPhase.CREATED,
      phaseStartTime: epochDateNow(),
      lastActivityTime: epochDateNow(),
      errorCount: 0,
      restartCount: 0,
      configuration: initialConfig || {},
      metadata: {
        version: agent.metadata.version,
        deploymentTime: epochDateNow(),
        healthScore: 1.0
      }
    }
    
    this.agentStates.set(agentId, state)
    
    // Store configuration if persistence is enabled
    if (this.config.enableConfigPersistence && initialConfig) {
      this.configStore.set(agentId, initialConfig)
    }
    
    this.logger?.info(`Registered agent ${agentId} for lifecycle management`, {
      phase: state.phase,
      version: state.metadata.version
    })
    
    // Emit lifecycle event
    this.eventBus.emit(EventTypes.AGENT_LIFECYCLE_CHANGED, {
      timestamp: epochDateNow(),
      agentId,
      phase: state.phase,
      previousPhase: null
    })
  }
  
  /**
   * Initialize agent with warm-up phase
   */
  async initializeAgent(agent: ITradeAgent): Promise<void> {
    const agentId = agent.metadata.id
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not registered`)
    }
    
    if (state.phase !== AgentPhase.CREATED) {
      throw new Error(`Agent ${agentId} already initialized`)
    }
    
    try {
      // Transition to initializing
      await this.transitionPhase(agentId, AgentPhase.INITIALIZING)
      
      // Load persisted configuration if available
      const savedConfig = this.configStore.get(agentId)
      if (savedConfig) {
        state.configuration = { ...savedConfig, ...state.configuration }
      }
      
      // Initialize the agent
      await agent.initialize(state.configuration)
      
      // Transition to warming up
      await this.transitionPhase(agentId, AgentPhase.WARMING_UP)
      
      // Perform warm-up
      await this.performWarmUp(agent)
      
      // Transition to ready
      await this.transitionPhase(agentId, AgentPhase.READY)
      
      // Update last activity time
      state.lastActivityTime = epochDateNow()
      
      // Start health checks
      this.startHealthChecks(agent)
      
      this.logger?.info(`Agent ${agentId} initialized successfully`)
      
    } catch (error) {
      await this.handleAgentError(agentId, error as Error)
      throw error
    }
  }
  
  /**
   * Activate agent for processing
   */
  async activateAgent(agentId: string): Promise<void> {
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    if (state.phase !== AgentPhase.READY && state.phase !== AgentPhase.PAUSED) {
      throw new Error(`Agent ${agentId} cannot be activated from phase ${state.phase}`)
    }
    
    await this.transitionPhase(agentId, AgentPhase.ACTIVE)
    
    this.logger?.info(`Agent ${agentId} activated`)
  }
  
  /**
   * Pause agent temporarily
   */
  async pauseAgent(agentId: string): Promise<void> {
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    if (state.phase !== AgentPhase.ACTIVE) {
      throw new Error(`Agent ${agentId} cannot be paused from phase ${state.phase}`)
    }
    
    await this.transitionPhase(agentId, AgentPhase.PAUSED)
    
    this.logger?.info(`Agent ${agentId} paused`)
  }
  
  /**
   * Retire agent gracefully
   */
  async retireAgent(agent: ITradeAgent, reason: string): Promise<void> {
    const agentId = agent.metadata.id
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    if (state.phase === AgentPhase.RETIRED) {
      return // Already retired
    }
    
    try {
      // Stop health checks
      this.stopHealthChecks(agentId)
      
      // Transition to cooling down
      await this.transitionPhase(agentId, AgentPhase.COOLING_DOWN)
      
      // Perform cool-down
      await this.performCoolDown(agent)
      
      // Shutdown agent
      if (agent.shutdown) {
        await agent.shutdown()
      }
      
      // Transition to retired
      await this.transitionPhase(agentId, AgentPhase.RETIRED)
      
      // Save final configuration if persistence is enabled
      if (this.config.enableConfigPersistence) {
        this.configStore.set(agentId, state.configuration)
      }
      
      this.logger?.info(`Agent ${agentId} retired`, { reason })
      
      // Emit retirement event
      this.eventBus.emit(EventTypes.AGENT_RETIRED, {
        timestamp: epochDateNow(),
        agentId,
        reason,
        finalState: state
      })
      
    } catch (error) {
      this.logger?.error(`Error retiring agent ${agentId}`, { error })
      // Force retirement even if graceful shutdown fails
      state.phase = AgentPhase.RETIRED
    }
  }
  
  /**
   * Replace agent with a new version
   */
  async replaceAgent(
    oldAgent: ITradeAgent,
    newAgent: ITradeAgent,
    reason: string
  ): Promise<void> {
    const oldId = oldAgent.metadata.id
    const newId = newAgent.metadata.id
    
    this.logger?.info(`Replacing agent ${oldId} with ${newId}`, { reason })
    
    // Get old agent state
    const oldState = this.agentStates.get(oldId)
    if (!oldState) {
      throw new Error(`Old agent ${oldId} not found`)
    }
    
    // Register new agent with old configuration
    this.registerAgent(newAgent, oldState.configuration)
    
    // Initialize new agent
    await this.initializeAgent(newAgent)
    
    // Pause old agent if active
    if (oldState.phase === AgentPhase.ACTIVE) {
      await this.pauseAgent(oldId)
    }
    
    // Activate new agent
    await this.activateAgent(newId)
    
    // Retire old agent
    await this.retireAgent(oldAgent, `Replaced by ${newId}: ${reason}`)
    
    // Emit replacement event
    this.eventBus.emit(EventTypes.AGENT_REPLACED, {
      timestamp: epochDateNow(),
      replacement: {
        oldAgentId: oldId,
        newAgentId: newId,
        reason,
        migrationData: oldState.configuration
      }
    })
  }
  
  /**
   * Perform health check on agent
   */
  async performHealthCheck(agent: ITradeAgent): Promise<HealthCheckResult> {
    const agentId = agent.metadata.id
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      return {
        healthy: false,
        score: 0,
        issues: ['Agent not registered'],
        metrics: {
          responseTime: 0,
          memoryUsage: 0,
          errorRate: 1
        }
      }
    }
    
    const startTime = Date.now()
    const issues: string[] = []
    let score = 1.0
    
    // Check phase
    if (state.phase === AgentPhase.ERROR) {
      issues.push('Agent in error state')
      score -= 0.5
    }
    
    // Check error count
    if (state.errorCount > 0) {
      const errorPenalty = Math.min(0.3, state.errorCount * 0.05)
      score -= errorPenalty
      issues.push(`Error count: ${state.errorCount}`)
    }
    
    // Check last activity
    const inactivityMs = epochDateNow() - state.lastActivityTime
    if (inactivityMs > 5 * 60 * 1000) { // 5 minutes
      issues.push('No recent activity')
      score -= 0.2
    }
    
    // Check agent-specific performance
    if (agent.getPerformance) {
      const performance = agent.getPerformance()
      if (performance.winRate < 0.3) {
        issues.push('Low win rate')
        score -= 0.2
      }
      if (performance.timeouts > 0) {
        issues.push(`Timeouts: ${performance.timeouts}`)
        score -= 0.1
      }
    }
    
    const responseTime = Date.now() - startTime
    
    // Update health score in state
    state.metadata.healthScore = Math.max(0, score)
    
    return {
      healthy: score >= 0.5,
      score,
      issues,
      metrics: {
        responseTime,
        memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024, // MB
        errorRate: state.errorCount / Math.max(1, state.restartCount + 1)
      }
    }
  }
  
  /**
   * Recover agent from error state
   */
  async recoverAgent(agent: ITradeAgent): Promise<void> {
    const agentId = agent.metadata.id
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    if (state.phase !== AgentPhase.ERROR) {
      throw new Error(`Agent ${agentId} not in error state`)
    }
    
    if (state.restartCount >= this.config.maxRestartAttempts) {
      throw new Error(`Agent ${agentId} exceeded maximum restart attempts`)
    }
    
    this.logger?.info(`Attempting to recover agent ${agentId}`, {
      restartCount: state.restartCount,
      errorCount: state.errorCount
    })
    
    try {
      // Reset agent
      if (agent.reset) {
        await agent.reset()
      }
      
      // Clear error count
      state.errorCount = 0
      state.restartCount++
      
      // Transition back to ready
      await this.transitionPhase(agentId, AgentPhase.READY)
      
      // Restart health checks
      this.startHealthChecks(agent)
      
      this.logger?.info(`Agent ${agentId} recovered successfully`)
      
    } catch (error) {
      this.logger?.error(`Failed to recover agent ${agentId}`, { error })
      throw error
    }
  }
  
  /**
   * Get agent state
   */
  getAgentState(agentId: string): AgentState | undefined {
    return this.agentStates.get(agentId)
  }
  
  /**
   * Get all agent states
   */
  getAllAgentStates(): Map<string, AgentState> {
    return new Map(this.agentStates)
  }
  
  /**
   * Update agent configuration
   */
  updateAgentConfiguration(
    agentId: string,
    config: Record<string, unknown>
  ): void {
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    state.configuration = { ...state.configuration, ...config }
    
    if (this.config.enableConfigPersistence) {
      this.configStore.set(agentId, state.configuration)
    }
    
    this.logger?.info(`Updated configuration for agent ${agentId}`, { config })
  }
  
  /**
   * Transition agent to new phase
   */
  private async transitionPhase(
    agentId: string,
    newPhase: AgentPhase
  ): Promise<void> {
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    const previousPhase = state.phase
    state.phase = newPhase
    state.phaseStartTime = epochDateNow()
    
    this.logger?.debug(`Agent ${agentId} transitioned`, {
      from: previousPhase,
      to: newPhase
    })
    
    // Emit phase change event
    this.eventBus.emit(EventTypes.AGENT_LIFECYCLE_CHANGED, {
      timestamp: epochDateNow(),
      agentId,
      phase: newPhase,
      previousPhase
    })
  }
  
  /**
   * Perform agent warm-up
   */
  private async performWarmUp(agent: ITradeAgent): Promise<void> {
    const agentId = agent.metadata.id
    
    this.logger?.debug(`Warming up agent ${agentId}`)
    
    // Simulate warm-up delay
    await new Promise(resolve => setTimeout(resolve, this.config.warmUpDuration))
    
    // Agent-specific warm-up logic could go here
    // For example: loading models, calibrating parameters, etc.
    
    this.logger?.debug(`Agent ${agentId} warm-up complete`)
  }
  
  /**
   * Perform agent cool-down
   */
  private async performCoolDown(agent: ITradeAgent): Promise<void> {
    const agentId = agent.metadata.id
    
    this.logger?.debug(`Cooling down agent ${agentId}`)
    
    // Simulate cool-down delay
    await new Promise(resolve => setTimeout(resolve, this.config.coolDownDuration))
    
    // Agent-specific cool-down logic could go here
    // For example: saving state, flushing buffers, etc.
    
    this.logger?.debug(`Agent ${agentId} cool-down complete`)
  }
  
  /**
   * Start health checks for agent
   */
  private startHealthChecks(agent: ITradeAgent): void {
    const agentId = agent.metadata.id
    
    // Clear any existing timer
    this.stopHealthChecks(agentId)
    
    // Start new health check timer
    const timer = setInterval(async () => {
      try {
        const health = await this.performHealthCheck(agent)
        
        if (!health.healthy && this.config.enableAutoRecovery) {
          const state = this.agentStates.get(agentId)
          if (state && state.phase !== AgentPhase.ERROR) {
            await this.handleAgentError(
              agentId,
              new Error(`Health check failed: ${health.issues.join(', ')}`)
            )
          }
        }
        
      } catch (error) {
        this.logger?.error(`Health check error for agent ${agentId}`, { error })
      }
    }, this.config.healthCheckInterval)
    
    this.healthCheckTimers.set(agentId, timer)
  }
  
  /**
   * Stop health checks for agent
   */
  private stopHealthChecks(agentId: string): void {
    const timer = this.healthCheckTimers.get(agentId)
    if (timer) {
      clearInterval(timer)
      this.healthCheckTimers.delete(agentId)
    }
  }
  
  /**
   * Handle agent error
   */
  private async handleAgentError(agentId: string, error: Error): Promise<void> {
    const state = this.agentStates.get(agentId)
    
    if (!state) {
      return
    }
    
    state.errorCount++
    
    this.logger?.error(`Agent ${agentId} encountered error`, {
      error: error.message,
      errorCount: state.errorCount
    })
    
    if (state.errorCount >= this.config.maxErrorCount) {
      await this.transitionPhase(agentId, AgentPhase.ERROR)
      
      // Emit critical error event
      this.eventBus.emit(EventTypes.AGENT_ERROR, {
        timestamp: epochDateNow(),
        agentId,
        error: error.message,
        errorCount: state.errorCount,
        critical: true
      })
    }
  }
  
  /**
   * Subscribe to relevant events
   */
  private subscribeToEvents(): void {
    // Update last activity time on agent signals
    this.eventBus.subscribe(EventTypes.CONSENSUS_REACHED, (data) => {
      const consensusData = data as unknown as { consensus: { agentSignals: Record<string, unknown> } }
      
      // Update activity time for all participating agents
      if (consensusData.consensus?.agentSignals) {
        for (const agentId of Object.keys(consensusData.consensus.agentSignals)) {
          const state = this.agentStates.get(agentId)
          if (state) {
            state.lastActivityTime = epochDateNow()
          }
        }
      }
    })
  }
  
  /**
   * Shutdown lifecycle manager
   */
  async shutdown(): Promise<void> {
    // Stop all health checks
    for (const agentId of this.healthCheckTimers.keys()) {
      this.stopHealthChecks(agentId)
    }
    
    // Clear all data
    this.agentStates.clear()
    this.configStore.clear()
    
    this.logger?.info('Agent lifecycle manager shutdown complete')
  }
}