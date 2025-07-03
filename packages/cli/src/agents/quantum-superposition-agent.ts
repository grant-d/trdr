import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface QuantumConfig {
  /** Time window for volume observations (in minutes) */
  observationWindow?: number
  /** Minimum volume threshold to "collapse" the wave function */
  collapseVolumeThreshold?: number
  /** Number of probability states to maintain */
  superpositionStates?: number
  /** Decay rate for probability amplitude over time */
  amplitudeDecay?: number
  /** Sensitivity to volume spikes (0-1) */
  volumeSensitivity?: number
}

interface QuantumState {
  price: number
  amplitude: number // Probability amplitude
  phase: number // Wave phase (0 to 2π)
  coherence: number // State coherence (0-1)
  timestamp: number
}

interface WaveFunction {
  states: QuantumState[]
  collapsed: boolean
  collapsePrice?: number
  collapseVolume?: number
  observationTime?: number
}

/**
 * Quantum Superposition Agent
 * 
 * Models price as existing in multiple probability states simultaneously until
 * "observed" through significant volume. Based on quantum mechanics concepts where:
 * - Price exists in superposition of multiple possible states
 * - High volume acts as "measurement" that collapses the wave function
 * - Probability amplitudes decay over time without observation
 * - Coherence decreases with market noise
 * 
 * Key Concepts:
 * - **Superposition**: Price can be in multiple states with different probabilities
 * - **Wave Collapse**: Large volume forces price to "choose" a specific state
 * - **Quantum Coherence**: How well the probability states maintain relationships
 * - **Amplitude Decay**: Unobserved states lose probability over time
 * 
 * Trading Signals:
 * - Strong buy/sell when wave function collapses with high coherence
 * - Weak signals when in superposition (uncertainty)
 * - Trail distance based on probability distribution width
 * 
 * @todo Implement wave function calculation from price action
 * @todo Implement volume-triggered collapse detection
 * @todo Calculate probability amplitudes for each price state
 * @todo Model quantum interference between price states
 * @todo Implement decoherence from market noise
 * @todo Calculate expected value from superposition states
 */
export class QuantumSuperpositionAgent extends BaseAgent {
  protected readonly config: Required<QuantumConfig>
  private waveFunction: WaveFunction = { states: [], collapsed: false }
  private priceHistory: { price: number, volume: number, timestamp: number }[] = []
  private coherenceHistory: number[] = []
  
  constructor(metadata: any, logger?: any, config?: QuantumConfig) {
    super(metadata, logger)
    
    this.config = {
      observationWindow: config?.observationWindow ?? 60, // 1 hour
      collapseVolumeThreshold: config?.collapseVolumeThreshold ?? 1.2, // 1.2x average (more sensitive)
      superpositionStates: config?.superpositionStates ?? 5,
      amplitudeDecay: config?.amplitudeDecay ?? 0.95, // 5% decay per period
      volumeSensitivity: config?.volumeSensitivity ?? 0.7
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Quantum Superposition Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    // Update price/volume history
    const currentVolume = candles[candles.length - 1]?.volume ?? 0
    this.priceHistory.push({ 
      price: currentPrice, 
      volume: currentVolume, 
      timestamp: Date.now() 
    })
    
    // Keep history within window
    const cutoffTime = Date.now() - this.config.observationWindow * 60 * 1000
    this.priceHistory = this.priceHistory.filter(h => h.timestamp > cutoffTime)
    
    // Calculate average volume for collapse detection
    const avgVolume = this.priceHistory.length > 0
      ? this.priceHistory.reduce((sum, h) => sum + h.volume, 0) / this.priceHistory.length
      : currentVolume
    
    // Check for volume-triggered collapse
    const volumeRatio = currentVolume / avgVolume
    const isCollapsing = volumeRatio >= this.config.collapseVolumeThreshold
    
    // Update wave function
    if (isCollapsing && !this.waveFunction.collapsed) {
      // Collapse the wave function
      this.waveFunction.collapsed = true
      this.waveFunction.collapsePrice = currentPrice
      this.waveFunction.collapseVolume = currentVolume
      this.waveFunction.observationTime = Date.now()
      
      // Determine collapse direction from recent price movement
      const priceChange = this.priceHistory.length > 1
        ? (currentPrice - this.priceHistory[this.priceHistory.length - 2]!.price) / this.priceHistory[this.priceHistory.length - 2]!.price
        : 0
      
      const action = priceChange > 0.0005 ? 'buy' : priceChange < -0.0005 ? 'sell' : 'hold'
      const confidence = Math.min(0.95, 0.7 + volumeRatio * 0.15 + Math.abs(priceChange) * 20)
      
      const signal = this.createSignal(
        action,
        confidence,
        `Wave function collapsed! Volume spike ${volumeRatio.toFixed(1)}x average, price ${priceChange > 0 ? 'rising' : 'falling'}`
      )
      
      return enforceNoShorting(signal, context)
    }
    
    // If already collapsed, check if we should reset
    if (this.waveFunction.collapsed) {
      const timeSinceCollapse = Date.now() - this.waveFunction.observationTime!
      if (timeSinceCollapse > this.config.observationWindow * 60 * 1000 / 2) {
        // Reset to superposition after half the observation window
        this.waveFunction.collapsed = false
        this.waveFunction.states = this.calculateWaveFunction(
          this.priceHistory.map(h => h.price),
          this.priceHistory.map(h => h.volume)
        ).states
      }
    }
    
    // In superposition - calculate probability states
    if (!this.waveFunction.collapsed) {
      this.waveFunction = this.calculateWaveFunction(
        this.priceHistory.map(h => h.price),
        this.priceHistory.map(h => h.volume)
      )
      
      // Calculate expected value from superposition
      const expectedPrice = this.getExpectedValue(this.waveFunction)
      const priceDeviation = (currentPrice - expectedPrice) / expectedPrice
      
      // Generate signals during superposition based on deviation
      if (Math.abs(priceDeviation) > 0.005) { // More sensitive threshold
        const action = priceDeviation < 0 ? 'buy' : 'sell'
        const confidence = Math.min(0.8, 0.5 + Math.abs(priceDeviation) * 15)
        
        const signal = this.createSignal(
          action,
          confidence,
          `In superposition. Price deviates ${(priceDeviation * 100).toFixed(1)}% from expected value`
        )
        
        return enforceNoShorting(signal, context)
      }
    }
    
    // Default hold signal
    return this.createSignal(
      'hold',
      0.4,
      `Quantum state: ${this.waveFunction.collapsed ? 'collapsed' : 'superposition'}, coherence: ${this.calculateCoherence().toFixed(2)}`
    )
  }
  
  /**
   * Calculate wave function from recent price action
   */
  private calculateWaveFunction(prices: number[], volumes: number[]): WaveFunction {
    if (prices.length === 0) {
      return { states: [], collapsed: false }
    }
    
    const states: QuantumState[] = []
    const currentPrice = prices[prices.length - 1]!
    const priceRange = Math.max(...prices) - Math.min(...prices)
    
    // Create superposition states around current price
    for (let i = 0; i < this.config.superpositionStates; i++) {
      const offset = (i - Math.floor(this.config.superpositionStates / 2)) * priceRange / this.config.superpositionStates
      const statePrice = currentPrice + offset
      
      // Calculate amplitude based on distance from current price
      const distance = Math.abs(offset) / priceRange
      const amplitude = Math.exp(-distance * distance * 2) // Gaussian distribution
      
      // Phase based on price momentum
      const momentum = prices.length > 1 ? prices[prices.length - 1]! - prices[prices.length - 2]! : 0
      const phase = Math.atan2(momentum, statePrice - currentPrice) + Math.PI
      
      // Coherence based on volume consistency
      const volumeStdDev = this.calculateStdDev(volumes)
      const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length
      const coherence = Math.exp(-volumeStdDev / avgVolume)
      
      states.push({
        price: statePrice,
        amplitude,
        phase,
        coherence,
        timestamp: Date.now()
      })
    }
    
    return { states, collapsed: false }
  }
  
  
  /**
   * Get expected price from superposition states
   */
  private getExpectedValue(waveFunction: WaveFunction): number {
    if (waveFunction.states.length === 0) {
      return this.priceHistory.length > 0 ? this.priceHistory[this.priceHistory.length - 1]!.price : 0
    }
    
    // Quantum expectation value: Σ(amplitude² × price)
    let totalProbability = 0
    let expectedValue = 0
    
    for (const state of waveFunction.states) {
      const probability = state.amplitude * state.amplitude
      totalProbability += probability
      expectedValue += probability * state.price
    }
    
    return totalProbability > 0 ? expectedValue / totalProbability : waveFunction.states[0]!.price
  }
  
  /**
   * Calculate coherence of the current quantum state
   */
  private calculateCoherence(): number {
    if (this.waveFunction.states.length === 0) {
      return 0
    }
    
    // Average coherence of all states
    const avgCoherence = this.waveFunction.states.reduce((sum, state) => sum + state.coherence, 0) / this.waveFunction.states.length
    
    // Decay coherence over time
    const timeFactor = this.waveFunction.collapsed && this.waveFunction.observationTime
      ? Math.exp(-(Date.now() - this.waveFunction.observationTime) / (this.config.observationWindow * 60 * 1000))
      : 1
    
    return avgCoherence * timeFactor
  }
  
  /**
   * Calculate standard deviation
   */
  private calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    return Math.sqrt(variance)
  }
  
  protected async onReset(): Promise<void> {
    this.waveFunction = { states: [], collapsed: false }
    this.priceHistory = []
    this.coherenceHistory = []
  }
}