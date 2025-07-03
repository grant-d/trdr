import type { 
  ITradeAgent, 
  AgentMetadata, 
  AgentSignal, 
  MarketContext,
  AgentPerformance,
  Trade
} from './types'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'

/**
 * Abstract base class for trading agents
 * Provides common functionality and default implementations
 */
export abstract class BaseAgent implements ITradeAgent {
  protected config: Record<string, unknown> = {}
  protected isInitialized = false
  
  // Use mutable performance tracking
  protected performanceData = {
    totalSignals: 0,
    profitableSignals: 0,
    winRate: 0,
    averageReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    avgExecutionTime: 0,
    timeouts: 0,
    lastUpdated: epochDateNow()
  }
  
  // Expose readonly performance
  protected get performance(): AgentPerformance {
    return { ...this.performanceData }
  }
  
  /** Track execution times for performance metrics */
  private executionTimes: number[] = []
  private readonly maxExecutionHistory = 100
  
  /** Track returns for performance calculation */
  private returns: number[] = []
  private readonly maxReturnsHistory = 1000
  
  constructor(
    public readonly metadata: AgentMetadata,
    protected readonly logger?: Logger
  ) {}
  
  /**
   * Initialize the agent with configuration
   */
  async initialize(config?: Record<string, unknown>): Promise<void> {
    if (this.isInitialized) {
      this.logger?.warn(`Agent ${this.metadata.id} already initialized`)
      return
    }
    
    this.config = { ...this.metadata.defaultConfig, ...config }
    
    // Perform agent-specific initialization
    await this.onInitialize()
    
    this.isInitialized = true
    this.logger?.info(`Agent ${this.metadata.id} initialized`, { config: this.config })
  }
  
  /**
   * Agent-specific initialization logic
   * Override in subclasses
   */
  protected abstract onInitialize(): Promise<void>
  
  /**
   * Analyze market conditions and generate trading signal
   */
  async analyze(context: MarketContext): Promise<AgentSignal> {
    if (!this.isInitialized) {
      throw new Error(`Agent ${this.metadata.id} not initialized`)
    }
    
    const startTime = Date.now()
    
    try {
      // Validate context
      this.validateContext(context)
      
      // Perform agent-specific analysis
      const signal = await this.performAnalysis(context)
      
      // Track execution time
      const executionTime = Date.now() - startTime
      this.trackExecutionTime(executionTime)
      
      // Update performance metrics
      this.performanceData.totalSignals++
      this.performanceData.lastUpdated = epochDateNow()
      
      this.logger?.debug(`Agent ${this.metadata.id} generated signal`, {
        action: signal.action,
        confidence: signal.confidence,
        executionTime
      })
      
      return signal
      
    } catch (error) {
      this.logger?.error(`Agent ${this.metadata.id} analysis failed`, { error })
      throw error
    }
  }
  
  /**
   * Agent-specific analysis logic
   * Must be implemented by subclasses
   */
  protected abstract performAnalysis(context: MarketContext): Promise<AgentSignal>
  
  /**
   * Validate market context before analysis
   */
  protected validateContext(context: MarketContext): void {
    if (!context.symbol) {
      throw new Error('Missing symbol in market context')
    }
    
    if (!context.currentPrice || context.currentPrice <= 0) {
      throw new Error('Invalid current price in market context')
    }
    
    if (!context.candles || context.candles.length === 0) {
      throw new Error('No candles provided in market context')
    }
    
    // Check for required indicators
    if (this.metadata.requiredIndicators) {
      for (const indicator of this.metadata.requiredIndicators) {
        if (!context.indicators?.[indicator]) {
          throw new Error(`Required indicator ${indicator} not provided`)
        }
      }
    }
  }
  
  /**
   * Update agent state based on trade execution result
   */
  async updateOnTrade(trade: Trade, _signal: AgentSignal): Promise<void> {
    // Calculate return if we have PnL
    if (trade.pnl !== undefined) {
      const returnPct = trade.pnl / (trade.price * trade.size)
      this.trackReturn(returnPct)
      
      if (trade.pnl > 0) {
        this.performanceData.profitableSignals++
      }
      
      // Update performance metrics
      this.updatePerformanceMetrics()
    }
    
    this.logger?.debug(`Agent ${this.metadata.id} updated on trade`, {
      tradeId: trade.id,
      pnl: trade.pnl
    })
  }
  
  /**
   * Get current agent performance metrics
   */
  getPerformance(): AgentPerformance {
    return { ...this.performanceData }
  }
  
  /**
   * Reset agent state
   */
  async reset(): Promise<void> {
    this.performanceData = {
      totalSignals: 0,
      profitableSignals: 0,
      winRate: 0,
      averageReturn: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      avgExecutionTime: 0,
      timeouts: 0,
      lastUpdated: epochDateNow()
    }
    
    this.executionTimes = []
    this.returns = []
    
    // Call agent-specific reset
    await this.onReset()
    
    this.logger?.info(`Agent ${this.metadata.id} reset`)
  }
  
  /**
   * Agent-specific reset logic
   * Override in subclasses if needed
   */
  protected async onReset(): Promise<void> {
    // Default: no-op
  }
  
  /**
   * Cleanup resources when agent is stopped
   */
  async shutdown(): Promise<void> {
    // Call agent-specific shutdown
    await this.onShutdown()
    
    this.isInitialized = false
    this.logger?.info(`Agent ${this.metadata.id} shutdown`)
  }
  
  /**
   * Agent-specific shutdown logic
   * Override in subclasses if needed
   */
  protected async onShutdown(): Promise<void> {
    // Default: no-op
  }
  
  /**
   * Track execution time for performance metrics
   */
  private trackExecutionTime(time: number): void {
    this.executionTimes.push(time)
    
    // Keep only recent history
    if (this.executionTimes.length > this.maxExecutionHistory) {
      this.executionTimes.shift()
    }
    
    // Update average
    this.performanceData.avgExecutionTime = 
      this.executionTimes.reduce((sum, t) => sum + t, 0) / this.executionTimes.length
  }
  
  /**
   * Track return for performance calculation
   */
  private trackReturn(returnPct: number): void {
    this.returns.push(returnPct)
    
    // Keep only recent history
    if (this.returns.length > this.maxReturnsHistory) {
      this.returns.shift()
    }
  }
  
  /**
   * Update performance metrics based on tracked data
   */
  private updatePerformanceMetrics(): void {
    // Win rate
    this.performanceData.winRate = this.performanceData.totalSignals > 0
      ? this.performanceData.profitableSignals / this.performanceData.totalSignals
      : 0
    
    // Average return
    this.performanceData.averageReturn = this.returns.length > 0
      ? this.returns.reduce((sum, r) => sum + r, 0) / this.returns.length
      : 0
    
    // Sharpe ratio (simplified - assumes risk-free rate of 0)
    if (this.returns.length > 1) {
      const mean = this.performanceData.averageReturn
      const variance = this.returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / this.returns.length
      const stdDev = Math.sqrt(variance)
      this.performanceData.sharpeRatio = stdDev > 0 ? mean / stdDev : 0
    }
    
    // Maximum drawdown
    if (this.returns.length > 0) {
      let peak = 0
      let maxDrawdown = 0
      let cumReturn = 0
      
      for (const ret of this.returns) {
        cumReturn += ret
        if (cumReturn > peak) {
          peak = cumReturn
        }
        const drawdown = (peak - cumReturn) / (1 + peak)
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown
        }
      }
      
      this.performanceData.maxDrawdown = maxDrawdown
    }
  }
  
  /**
   * Helper method to create a signal
   */
  protected createSignal(
    action: AgentSignal['action'],
    confidence: number,
    reason: string,
    analysis?: string,
    priceTarget?: number,
    stopLoss?: number,
    positionSize?: number
  ): AgentSignal {
    return {
      action,
      confidence: Math.max(0, Math.min(1, confidence)), // Clamp to [0, 1]
      reason,
      analysis,
      priceTarget,
      stopLoss,
      positionSize,
      timestamp: epochDateNow()
    }
  }
}