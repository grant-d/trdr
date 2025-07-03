import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface LorentzianConfig {
  /** Number of historical points to compare */
  lookbackPeriod?: number
  /** Threshold for pattern similarity (lower = more similar) */
  similarityThreshold?: number
  /** Weight for imaginary component (momentum) */
  imaginaryWeight?: number
  /** Minimum pattern matches for signal */
  minMatches?: number
  /** Use relativistic time dilation effects */
  useTimeDilation?: boolean
  /** Speed of light analog for market (max price velocity) */
  marketSpeedLimit?: number
}

interface ComplexPrice {
  real: number // Actual price
  imaginary: number // Momentum/velocity component
  magnitude: number // |z| = sqrt(real² + imaginary²)
  phase: number // arg(z) = atan2(imaginary, real)
  timestamp: number
}

interface LorentzianPattern {
  centerPoint: ComplexPrice
  radius: number // Pattern boundary in Lorentzian space
  matches: ComplexPrice[] // Similar historical points
  averageDistance: number
  futureReturn: number // Average return after pattern
  confidence: number
}

interface SpacetimeEvent {
  price: ComplexPrice
  timeCoordinate: number
  spaceCoordinate: number // Price level mapped to space
  fourVelocity: { t: number, x: number, y: number, z: number }
  properTime: number // Invariant time in price rest frame
}

/**
 * Lorentzian Distance Agent
 * 
 * Uses Lorentzian (hyperbolic) distance metrics and complex number representations
 * to find patterns in price-momentum space. Based on special relativity concepts
 * where price and momentum form a spacetime-like manifold.
 * 
 * The Lorentzian distance formula:
 * d = log(1 + (x - y)²)
 * 
 * Extended to complex numbers:
 * d = log(1 + |z₁ - z₂|²) where z = price + i*momentum
 * 
 * Key Concepts:
 * - **Complex Price**: price + i*momentum creates phase space representation
 * - **Lorentzian Metric**: Hyperbolic distance preserves causal relationships
 * - **Light Cones**: Future/past regions based on maximum price velocity
 * - **Invariant Intervals**: Pattern matching in relativistic price-time
 * - **Time Dilation**: High momentum periods experience "slower" time
 * 
 * Trading Signals:
 * - Buy when current pattern matches historical profitable patterns
 * - Sell when entering "forbidden" regions (outside light cone)
 * - Strong signals at phase transitions (real/imaginary crossover)
 * - Trail distance based on Lorentzian neighborhood radius
 * - Exit when pattern coherence breaks down
 * 
 * @todo Implement complex price representation with momentum
 * @todo Calculate Lorentzian distances between price points
 * @todo Build pattern library using Lorentzian neighborhoods
 * @todo Implement light cone constraints for causality
 * @todo Add relativistic effects for high-momentum periods
 * @todo Find nearest neighbors in Lorentzian space
 */
export class LorentzianDistanceAgent extends BaseAgent {
  protected readonly config: Required<LorentzianConfig>
  private priceHistory: ComplexPrice[] = []
  // @ts-ignore - unused variable (reserved for future use)
  private _patternLibrary: LorentzianPattern[] = []
  private spacetimeEvents: SpacetimeEvent[] = []
  // @ts-ignore - unused variable (reserved for future use)
  private _currentLightCone: { future: ComplexPrice[], past: ComplexPrice[] } = { future: [], past: [] }
  
  constructor(metadata: any, logger?: any, config?: LorentzianConfig) {
    super(metadata, logger)
    
    this.config = {
      lookbackPeriod: config?.lookbackPeriod ?? 100,
      similarityThreshold: config?.similarityThreshold ?? 0.5, // Increased significantly for more matches
      imaginaryWeight: config?.imaginaryWeight ?? 1.0, // Increased to make momentum more significant
      minMatches: config?.minMatches ?? 2, // Reduced to allow more signals
      useTimeDilation: config?.useTimeDilation ?? true,
      marketSpeedLimit: config?.marketSpeedLimit ?? 0.05 // 5% max price change per period
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Lorentzian Distance Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < 10) {
      return this.createSignal('hold', 0.3, 'Insufficient data for Lorentzian analysis')
    }
    
    // 1. Calculate momentum from recent price changes
    const prices = candles.slice(-20).map(c => c.close)
    const momentum = this.calculateMomentum(prices, 5)
    
    // Debug logging
    if (Math.random() < 0.05) { // Log 5% of the time to avoid spam
      this.logger?.debug('Lorentzian analysis', {
        currentPrice,
        momentum: momentum.toFixed(6),
        priceHistoryLength: this.priceHistory.length
      })
    }
    
    // 2. Create complex price representation
    const currentComplex = this.createComplexPrice(currentPrice, momentum)
    
    // 3. Update price history
    this.priceHistory.push(currentComplex)
    if (this.priceHistory.length > this.config.lookbackPeriod) {
      this.priceHistory.shift()
    }
    
    // Need minimum history
    if (this.priceHistory.length < 20) {
      return this.createSignal('hold', 0.4, 'Building Lorentzian pattern library')
    }
    
    // 4. Find similar patterns using Lorentzian distance
    const neighbors = this.findLorentzianNeighbors(
      currentComplex,
      this.priceHistory.slice(0, -1), // Exclude current
      this.config.similarityThreshold
    )
    
    // Debug pattern matching
    if (this.priceHistory.length > 50 && neighbors.length === 0) {
      // Try with a more lenient threshold to see if patterns exist
      const lenientNeighbors = this.findLorentzianNeighbors(
        currentComplex,
        this.priceHistory.slice(0, -1),
        this.config.similarityThreshold * 2 // Double the threshold
      )
      if (lenientNeighbors.length > 0) {
        this.logger?.debug('Found patterns with lenient threshold', {
          strictThreshold: this.config.similarityThreshold,
          lenientThreshold: this.config.similarityThreshold * 2,
          lenientMatches: lenientNeighbors.length
        })
      }
    }
    
    if (neighbors.length < this.config.minMatches) {
      return this.createSignal('hold', 0.5, `Found ${neighbors.length} patterns, need ${this.config.minMatches}`)
    }
    
    // 5. Create spacetime event for relativistic analysis
    const currentEvent = this.createSpacetimeEvent(currentComplex, this.priceHistory)
    this.spacetimeEvents.push(currentEvent)
    
    // 6. Check causality constraints
    if (this.config.useTimeDilation) {
      this._currentLightCone = this.constructLightCone(currentEvent, this.config.marketSpeedLimit)
    }
    
    // 7. Analyze pattern outcomes
    const patternAnalysis = this.analyzePatternOutcomes(currentComplex, neighbors)
    
    // 8. Generate signal based on pattern predictions
    // Adjusted thresholds to be more balanced
    if (patternAnalysis.averageReturn > 0.0005 && patternAnalysis.confidence > 0.55) {
      return this.createSignal(
        'buy',
        patternAnalysis.confidence,
        `Lorentzian pattern match: ${neighbors.length} similar patterns, avg return: ${(patternAnalysis.averageReturn * 100).toFixed(2)}%`
      )
    } else if (patternAnalysis.averageReturn < -0.0005 && patternAnalysis.confidence > 0.55) {
      const signal = this.createSignal(
        'sell',
        patternAnalysis.confidence,
        `Lorentzian pattern match: ${neighbors.length} similar patterns, avg return: ${(patternAnalysis.averageReturn * 100).toFixed(2)}%`
      )
      return enforceNoShorting(signal, context)
    }
    
    // Check phase transitions
    const phaseTransition = this.detectPhaseTransition(currentComplex, this.priceHistory)
    if (phaseTransition.detected && Math.abs(currentComplex.imaginary) > 0.0001) {
      // Only act on phase transitions with significant momentum
      const signal = this.createSignal(
        phaseTransition.direction === 'positive' ? 'buy' : 'sell',
        0.65,
        `Phase transition detected: ${phaseTransition.type}, momentum: ${currentComplex.imaginary.toFixed(4)}`
      )
      return enforceNoShorting(signal, context)
    }
    
    // Check for momentum reversal patterns
    const reversalSignal = this.detectMomentumReversal(currentComplex, this.priceHistory)
    if (reversalSignal.action !== 'hold') {
      return enforceNoShorting(reversalSignal, context)
    }
    
    const phaseDegrees = isFinite(currentComplex.phase) ? 
      (currentComplex.phase * 180 / Math.PI).toFixed(0) : '0'
    
    return this.createSignal(
      'hold',
      0.5,
      `Lorentzian analysis: ${neighbors.length} patterns, phase: ${phaseDegrees}°, momentum: ${currentComplex.imaginary.toFixed(4)}`
    )
  }
  
  /**
   * Create spacetime event from complex price
   */
  private createSpacetimeEvent(price: ComplexPrice, history: ComplexPrice[]): SpacetimeEvent {
    const previousPrice = history[history.length - 1] || price
    const fourVelocity = this.calculateFourVelocity(price, previousPrice)
    
    return {
      price,
      timeCoordinate: price.timestamp,
      spaceCoordinate: price.real,
      fourVelocity,
      properTime: this.config.useTimeDilation ? 
        this.applyTimeDilation(Math.abs(price.imaginary), price.timestamp - previousPrice.timestamp) :
        price.timestamp - previousPrice.timestamp
    }
  }
  
  /**
   * Analyze outcomes of similar patterns
   */
  private analyzePatternOutcomes(
    _current: ComplexPrice,
    neighbors: ComplexPrice[]
  ): { averageReturn: number, confidence: number } {
    if (neighbors.length === 0) return { averageReturn: 0, confidence: 0 }
    
    let totalReturn = 0
    let validOutcomes = 0
    
    // Look at what happened after each neighbor pattern
    for (const neighbor of neighbors) {
      const neighborIndex = this.priceHistory.findIndex(p => 
        p.timestamp === neighbor.timestamp
      )
      
      if (neighborIndex >= 0 && neighborIndex < this.priceHistory.length - 5) {
        // Look 5 periods ahead
        const futurePrice = this.priceHistory[neighborIndex + 5]!
        const return_ = (futurePrice.real - neighbor.real) / neighbor.real
        totalReturn += return_
        validOutcomes++
      } else if (neighborIndex >= 0 && neighborIndex < this.priceHistory.length - 2) {
        // If we can't look 5 ahead, try 2 periods
        const futurePrice = this.priceHistory[neighborIndex + 2]!
        const return_ = (futurePrice.real - neighbor.real) / neighbor.real
        totalReturn += return_ * 0.5 // Weight less since shorter timeframe
        validOutcomes += 0.5
      }
    }
    
    if (validOutcomes === 0) return { averageReturn: 0, confidence: 0 }
    
    const averageReturn = totalReturn / validOutcomes
    const confidence = Math.min(0.9, 0.5 + (validOutcomes / neighbors.length) * 0.4)
    
    return { averageReturn, confidence }
  }
  
  /**
   * Detect phase transitions in complex price space
   */
  private detectPhaseTransition(
    current: ComplexPrice,
    history: ComplexPrice[]
  ): { detected: boolean, type: string, direction: 'positive' | 'negative' } {
    if (history.length < 5) return { detected: false, type: '', direction: 'positive' }
    
    const recentPhases = history.slice(-5).map(p => p.phase)
    const currentPhase = current.phase
    
    // Check for phase wrap-around (crossing ±π boundary)
    const phaseJump = Math.abs(currentPhase - recentPhases[recentPhases.length - 1]!)
    if (phaseJump > Math.PI) {
      return {
        detected: true,
        type: 'Phase wrap-around',
        direction: currentPhase > 0 ? 'positive' : 'negative'
      }
    }
    
    // Check for quadrant changes
    const previousQuadrant = Math.floor((recentPhases[recentPhases.length - 1]! + Math.PI) / (Math.PI / 2))
    const currentQuadrant = Math.floor((currentPhase + Math.PI) / (Math.PI / 2))
    
    if (previousQuadrant !== currentQuadrant) {
      // Determine direction based on quadrant transition, not just momentum
      const quadrantTransition = currentQuadrant - previousQuadrant
      const isPositiveTransition = (quadrantTransition === 1 || quadrantTransition === -3) || 
                                   (currentQuadrant === 0 || currentQuadrant === 3)
      
      return {
        detected: true,
        type: `Quadrant ${previousQuadrant} → ${currentQuadrant}`,
        direction: isPositiveTransition ? 'positive' : 'negative'
      }
    }
    
    return { detected: false, type: '', direction: 'positive' }
  }
  
  /**
   * Convert price and momentum to complex number
   */
  private createComplexPrice(price: number, momentum: number): ComplexPrice {
    const real = price
    const imaginary = momentum * this.config.imaginaryWeight
    const magnitude = Math.sqrt(real * real + imaginary * imaginary)
    const phase = Math.atan2(imaginary, real)
    
    return {
      real,
      imaginary,
      magnitude,
      phase,
      timestamp: Date.now()
    }
  }
  
  /**
   * Calculate Lorentzian distance between complex prices
   * d = log(1 + |z₁ - z₂|²)
   */
  private lorentzianDistance(z1: ComplexPrice, z2: ComplexPrice): number {
    const deltaReal = z1.real - z2.real
    const deltaImag = z1.imaginary - z2.imaginary
    const deltaMagnitudeSquared = deltaReal * deltaReal + deltaImag * deltaImag
    return Math.log(1 + deltaMagnitudeSquared)
  }
  
  /**
   * Find patterns in Lorentzian neighborhoods
   */
  private findLorentzianNeighbors(
    target: ComplexPrice, 
    history: ComplexPrice[], 
    radius: number
  ): ComplexPrice[] {
    const neighbors: ComplexPrice[] = []
    
    for (const historical of history) {
      const distance = this.lorentzianDistance(target, historical)
      
      // Check if within Lorentzian radius
      if (distance <= radius) {
        neighbors.push(historical)
      }
    }
    
    // Sort by distance (closest first)
    neighbors.sort((a, b) => {
      const distA = this.lorentzianDistance(target, a)
      const distB = this.lorentzianDistance(target, b)
      return distA - distB
    })
    
    return neighbors
  }
  
  /**
   * Build light cone for causality constraints
   */
  private constructLightCone(
    event: SpacetimeEvent, 
    speedLimit: number
  ): { future: ComplexPrice[], past: ComplexPrice[] } {
    const future: ComplexPrice[] = []
    const past: ComplexPrice[] = []
    
    // Check all price points in history
    for (const price of this.priceHistory) {
      const timeDiff = price.timestamp - event.timeCoordinate
      const spaceDiff = Math.abs(price.real - event.spaceCoordinate)
      
      // Maximum distance reachable at speed limit
      const maxDistance = Math.abs(timeDiff) * speedLimit / 1000 // Convert ms to s
      
      // Check if within light cone
      if (spaceDiff <= maxDistance) {
        if (timeDiff > 0) {
          future.push(price) // Future light cone
        } else if (timeDiff < 0) {
          past.push(price) // Past light cone
        }
      }
    }
    
    return { future, past }
  }
  
  /**
   * Apply relativistic time dilation
   */
  private applyTimeDilation(momentum: number, deltaTime: number): number {
    // Treat momentum as velocity analog
    const v = Math.min(Math.abs(momentum), this.config.marketSpeedLimit * 0.99) // Cap at 99% of c
    const c = this.config.marketSpeedLimit
    
    // Lorentz factor
    const gamma = 1 / Math.sqrt(1 - (v / c) ** 2)
    
    // Proper time = coordinate time / gamma
    // High momentum = time dilation = slower proper time
    return deltaTime / gamma
  }
  
  /**
   * Calculate four-velocity in price spacetime
   */
  private calculateFourVelocity(
    current: ComplexPrice, 
    previous: ComplexPrice
  ): { t: number, x: number, y: number, z: number } {
    const dt = (current.timestamp - previous.timestamp) / 1000 // Convert to seconds
    if (dt === 0) return { t: 1, x: 0, y: 0, z: 0 }
    
    // Spatial components from price changes
    const dx = current.real - previous.real
    const dy = current.imaginary - previous.imaginary
    const dz = current.magnitude - previous.magnitude // Use magnitude as z-dimension
    
    // Calculate 3-velocity
    const v_x = dx / dt
    const v_y = dy / dt
    const v_z = dz / dt
    
    // Calculate velocity magnitude
    const v_squared = v_x * v_x + v_y * v_y + v_z * v_z
    // @ts-ignore - unused variable (reserved for future use)
    const _v = Math.sqrt(v_squared)
    
    // Lorentz factor (gamma)
    const c = this.config.marketSpeedLimit // Speed limit
    const gamma = 1 / Math.sqrt(1 - Math.min(0.99, v_squared / (c * c)))
    
    // Four-velocity components
    return {
      t: gamma * c, // Time component
      x: gamma * v_x,
      y: gamma * v_y,
      z: gamma * v_z
    }
  }
  
  /**
   * Check if pattern is within light cone (causal)
   * @todo Verify causality constraints
   */
  // @ts-ignore - unused variable (reserved for future use)
  private _isWithinLightCone(
    _event1: SpacetimeEvent, 
    _event2: SpacetimeEvent
  ): boolean {
    // Stub implementation
    return true
  }
  
  /**
   * Transform to price rest frame
   * @todo Lorentz transformation to remove price trend
   */
  // @ts-ignore - unused variable (reserved for future use)
  private _lorentzTransform(
    event: SpacetimeEvent, 
    _velocity: number
  ): SpacetimeEvent {
    // Stub implementation
    return event
  }
  
  /**
   * Calculate pattern quality using Lorentzian metric
   * @todo Assess pattern strength in hyperbolic space
   */
  // @ts-ignore - unused variable (reserved for future use)
  private _calculatePatternQuality(_pattern: LorentzianPattern): number {
    // Stub implementation
    return 0.5
  }
  
  /**
   * Predict future price using matched patterns
   * @todo Extrapolate along geodesics in Lorentzian space
   */
  // @ts-ignore - unused variable (reserved for future use)
  private _predictFromPatterns(
    _matches: LorentzianPattern[], 
    _currentPhase: number
  ): { price: number, confidence: number } {
    // Stub implementation
    return { price: 0, confidence: 0 }
  }
  
  /**
   * Calculate momentum from price changes
   */
  private calculateMomentum(prices: number[], period: number): number {
    if (prices.length < period + 1) return 0
    
    // Calculate rate of change over period
    const recentPrices = prices.slice(-period - 1)
    const oldPrice = recentPrices[0]!
    const newPrice = recentPrices[recentPrices.length - 1]!
    
    // Basic momentum as price rate of change
    const momentum = (newPrice - oldPrice) / oldPrice / period
    
    // Apply smoothing using weighted average of recent changes
    let smoothedMomentum = 0
    let weightSum = 0
    
    for (let i = 1; i < recentPrices.length; i++) {
      const change = (recentPrices[i]! - recentPrices[i-1]!) / recentPrices[i-1]!
      const weight = i / recentPrices.length // More recent = higher weight
      smoothedMomentum += change * weight
      weightSum += weight
    }
    
    if (weightSum > 0) {
      smoothedMomentum /= weightSum
    }
    
    // Blend basic and smoothed momentum with normalization
    const blendedMomentum = momentum * 0.7 + smoothedMomentum * 0.3
    
    // Add detrending to avoid directional bias
    const avgPrice = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length
    const priceStdDev = Math.sqrt(
      recentPrices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / recentPrices.length
    )
    const normalizedMomentum = priceStdDev > 0 ? blendedMomentum * avgPrice / priceStdDev : blendedMomentum
    
    return normalizedMomentum
  }
  
  /**
   * Detect momentum reversal patterns
   */
  private detectMomentumReversal(
    current: ComplexPrice,
    history: ComplexPrice[]
  ): AgentSignal {
    if (history.length < 10) {
      return this.createSignal('hold', 0.4, 'Insufficient history for reversal detection')
    }
    
    const recentHistory = history.slice(-10)
    const momentums = recentHistory.map(p => p.imaginary)
    momentums.push(current.imaginary)
    
    // Calculate momentum trend
    let trendSum = 0
    for (let i = 1; i < momentums.length; i++) {
      trendSum += momentums[i]! - momentums[i-1]!
    }
    const momentumTrend = trendSum / (momentums.length - 1)
    
    // Find local extremes
    const recentMomentum = momentums.slice(-5)
    const minMomentum = Math.min(...recentMomentum)
    const maxMomentum = Math.max(...recentMomentum)
    const currentMomentum = current.imaginary
    
    // Detect bullish reversal: momentum was very negative, now turning positive
    if (minMomentum < -0.001 && currentMomentum > minMomentum * 0.5 && momentumTrend > 0.00001) {
      const confidence = Math.min(0.7, 0.5 + Math.abs(momentumTrend) * 100)
      return this.createSignal(
        'buy',
        confidence,
        `Bullish momentum reversal: min ${minMomentum.toFixed(4)} → current ${currentMomentum.toFixed(4)}`
      )
    }
    
    // Detect bearish reversal: momentum was very positive, now turning negative
    if (maxMomentum > 0.001 && currentMomentum < maxMomentum * 0.5 && momentumTrend < -0.00001) {
      const confidence = Math.min(0.7, 0.5 + Math.abs(momentumTrend) * 100)
      return this.createSignal(
        'sell',
        confidence,
        `Bearish momentum reversal: max ${maxMomentum.toFixed(4)} → current ${currentMomentum.toFixed(4)}`
      )
    }
    
    return this.createSignal('hold', 0.5, 'No momentum reversal detected')
  }
  
  protected async onReset(): Promise<void> {
    this.priceHistory = []
    this._patternLibrary = []
    this.spacetimeEvents = []
    this._currentLightCone = { future: [], past: [] }
  }
}