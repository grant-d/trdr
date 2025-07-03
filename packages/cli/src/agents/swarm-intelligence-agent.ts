import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface SwarmConfig {
  /** Number of market participants to simulate */
  swarmSize?: number
  /** Communication radius between participants */
  communicationRadius?: number
  /** Influence decay over distance */
  influenceDecay?: number
  /** Threshold for herd behavior detection */
  herdThreshold?: number
  /** Weight of social vs individual signals */
  socialWeight?: number
  /** Memory length for participant behavior */
  memoryLength?: number
}

interface MarketParticipant {
  id: string
  position: { price: number, size: number }
  velocity: number // Rate of position change
  bias: 'bullish' | 'bearish' | 'neutral'
  confidence: number
  influence: number // How much this participant influences others
  neighbors: string[] // IDs of nearby participants
  history: { action: 'buy' | 'sell' | 'hold', timestamp: number }[]
}

interface SwarmState {
  participants: Map<string, MarketParticipant>
  centerOfMass: number // Average price position
  momentum: number // Collective movement direction
  cohesion: number // How tightly grouped the swarm is
  alignment: number // How aligned participant movements are
  separation: number // Tendency to avoid crowding
}

interface EmergentPattern {
  type: 'convergence' | 'divergence' | 'migration' | 'fragmentation' | 'flocking'
  strength: number
  direction: 'up' | 'down' | 'sideways'
  participants: string[] // IDs involved in pattern
  duration: number
}

/**
 * Swarm Intelligence Agent
 * 
 * Models market participants as a swarm with emergent collective behavior.
 * Based on swarm intelligence principles where simple individual rules lead to
 * complex group patterns (like bird flocking or fish schooling).
 * 
 * Key Concepts:
 * - **Separation**: Avoid crowding neighbors (contrarian behavior)
 * - **Alignment**: Steer toward average heading of neighbors (trend following)
 * - **Cohesion**: Move toward average position of neighbors (herd behavior)
 * - **Emergence**: Complex patterns from simple rules (market phenomena)
 * - **Information Cascade**: How signals propagate through the swarm
 * 
 * Trading Signals:
 * - Buy when swarm converges upward (collective bullishness)
 * - Sell when swarm fragments (uncertainty/divergence)
 * - Strong signals during coordinated migration patterns
 * - Fade moves when swarm is too cohesive (overcrowded)
 * - Trail distance based on swarm dispersion
 * 
 * @todo Implement participant simulation with position/velocity
 * @todo Calculate swarm metrics (center of mass, cohesion, etc.)
 * @todo Detect emergent patterns from collective behavior
 * @todo Model information propagation through swarm
 * @todo Identify leader/follower dynamics
 * @todo Generate signals from swarm behavior
 */
export class SwarmIntelligenceAgent extends BaseAgent {
  protected readonly config: Required<SwarmConfig>
  private swarmState: SwarmState = {
    participants: new Map(),
    centerOfMass: 0,
    momentum: 0,
    cohesion: 0,
    alignment: 0,
    separation: 0
  }
  private emergentPatterns: EmergentPattern[] = []
  
  constructor(metadata: any, logger?: any, config?: SwarmConfig) {
    super(metadata, logger)
    
    this.config = {
      swarmSize: config?.swarmSize ?? 100, // 100 simulated participants
      communicationRadius: config?.communicationRadius ?? 0.02, // 2% price distance
      influenceDecay: config?.influenceDecay ?? 0.5, // 50% decay per hop
      herdThreshold: config?.herdThreshold ?? 0.7, // 70% alignment = herd
      socialWeight: config?.socialWeight ?? 0.6, // 60% social, 40% individual
      memoryLength: config?.memoryLength ?? 50 // Remember 50 actions
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Swarm Intelligence Agent initialized', this.config)
    this.initializeSwarm()
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    const currentVolume = candles[candles.length - 1]?.volume ?? 0
    
    // Update participant positions based on market data
    this.updateParticipantPositions(currentPrice, currentVolume)
    
    // Calculate neighbor relationships for all participants
    this.updateNeighborRelationships()
    
    // Apply swarm behavior rules to each participant
    const participants = Array.from(this.swarmState.participants.values())
    for (const participant of participants) {
      const neighbors = this.getNeighbors(participant)
      
      // Calculate three forces
      const separation = this.calculateSeparation(participant, neighbors)
      const alignment = this.calculateAlignment(participant, neighbors)  
      const cohesion = this.calculateCohesion(participant, neighbors)
      
      // Update velocity based on forces
      const socialForce = separation + alignment + cohesion
      const individualForce = this.calculateIndividualForce(participant, currentPrice)
      
      participant.velocity = participant.velocity * 0.9 + // Momentum
        socialForce * this.config.socialWeight +
        individualForce * (1 - this.config.socialWeight)
      
      // Clamp velocity
      participant.velocity = Math.max(-0.05, Math.min(0.05, participant.velocity))
      
      // Update position
      participant.position.price += participant.velocity
      
      // Update action history
      const action = participant.velocity > 0.001 ? 'buy' : 
                     participant.velocity < -0.001 ? 'sell' : 'hold'
      participant.history.push({ action, timestamp: Date.now() })
      if (participant.history.length > this.config.memoryLength) {
        participant.history.shift()
      }
    }
    
    // Update swarm state metrics
    this.updateSwarmMetrics()
    
    // Detect emergent patterns
    this.emergentPatterns = this.detectEmergentPatterns(this.swarmState)
    
    // Calculate consensus metrics
    const consensus = this.calculateSwarmConsensus()
    
    // Generate signal from collective behavior
    const signal = this.generateSwarmSignal(this.emergentPatterns, consensus, this.swarmState)
    
    return enforceNoShorting(signal, context)
  }
  
  /**
   * Update neighbor relationships based on communication radius
   */
  private updateNeighborRelationships(): void {
    const participants = Array.from(this.swarmState.participants.values())
    
    for (const p1 of participants) {
      p1.neighbors = []
      
      for (const p2 of participants) {
        if (p1.id === p2.id) continue
        
        const distance = Math.abs(p1.position.price - p2.position.price) / p1.position.price
        if (distance <= this.config.communicationRadius) {
          p1.neighbors.push(p2.id)
        }
      }
    }
  }
  
  /**
   * Get neighbor participants
   */
  private getNeighbors(participant: MarketParticipant): MarketParticipant[] {
    return participant.neighbors
      .map(id => this.swarmState.participants.get(id))
      .filter((p): p is MarketParticipant => p !== undefined)
  }
  
  /**
   * Calculate individual force based on market conditions
   */
  private calculateIndividualForce(participant: MarketParticipant, marketPrice: number): number {
    // Individual decision based on bias and market position
    const priceDeviation = (marketPrice - participant.position.price) / participant.position.price
    
    let force = 0
    if (participant.bias === 'bullish') {
      // Bullish participants buy dips
      force = priceDeviation < 0 ? Math.abs(priceDeviation) * 0.1 : -priceDeviation * 0.05
    } else if (participant.bias === 'bearish') {
      // Bearish participants sell rallies
      force = priceDeviation > 0 ? -Math.abs(priceDeviation) * 0.1 : priceDeviation * 0.05
    } else {
      // Neutral participants mean revert
      force = -priceDeviation * 0.05
    }
    
    // Apply confidence scaling
    return force * participant.confidence
  }
  
  /**
   * Initialize swarm with random participants
   */
  private initializeSwarm(): void {
    this.swarmState.participants.clear()
    
    for (let i = 0; i < this.config.swarmSize; i++) {
      const id = `p${i}`
      
      // Create diverse participant with random characteristics
      const participant: MarketParticipant = {
        id,
        position: {
          price: 1 + (Math.random() - 0.5) * 0.1, // ±5% from baseline
          size: Math.random() * 1000 // Random position size
        },
        velocity: (Math.random() - 0.5) * 0.01, // Random initial velocity
        bias: Math.random() < 0.33 ? 'bullish' : Math.random() < 0.66 ? 'bearish' : 'neutral',
        confidence: 0.5 + Math.random() * 0.5, // 0.5-1.0 confidence
        influence: Math.random(), // Random influence level
        neighbors: [],
        history: []
      }
      
      this.swarmState.participants.set(id, participant)
    }
    
    // Calculate initial swarm metrics
    this.updateSwarmMetrics()
  }
  
  /**
   * Update swarm-wide metrics
   */
  private updateSwarmMetrics(): void {
    const participants = Array.from(this.swarmState.participants.values())
    if (participants.length === 0) return
    
    // Center of mass (average price position)
    this.swarmState.centerOfMass = participants.reduce((sum, p) => sum + p.position.price, 0) / participants.length
    
    // Momentum (average velocity)
    this.swarmState.momentum = participants.reduce((sum, p) => sum + p.velocity, 0) / participants.length
    
    // Cohesion (inverse of price variance)
    const priceVariance = participants.reduce((sum, p) => 
      sum + Math.pow(p.position.price - this.swarmState.centerOfMass, 2), 0
    ) / participants.length
    this.swarmState.cohesion = 1 / (1 + Math.sqrt(priceVariance) * 10)
    
    // Alignment (velocity correlation)
    const velocityVariance = participants.reduce((sum, p) => 
      sum + Math.pow(p.velocity - this.swarmState.momentum, 2), 0
    ) / participants.length
    this.swarmState.alignment = 1 / (1 + Math.sqrt(velocityVariance) * 100)
    
    // Separation (minimum distance tendency)
    let totalSeparation = 0
    for (let i = 0; i < participants.length; i++) {
      const p1 = participants[i]!
      let minDistance = Infinity
      for (let j = 0; j < participants.length; j++) {
        if (i !== j) {
          const p2 = participants[j]!
          const distance = Math.abs(p1.position.price - p2.position.price)
          minDistance = Math.min(minDistance, distance)
        }
      }
      totalSeparation += minDistance
    }
    this.swarmState.separation = totalSeparation / participants.length
  }
  
  /**
   * Update participant positions based on swarm rules
   */
  private updateParticipantPositions(marketPrice: number, marketVolume: number): void {
    // Update market anchor point for participants
    const participants = Array.from(this.swarmState.participants.values())
    
    for (const participant of participants) {
      // Adjust participant's reference frame to current market
      const marketInfluence = 0.1 // How much market price pulls participants
      const targetPrice = participant.position.price * (1 - marketInfluence) + marketPrice * marketInfluence
      
      // Update position toward market
      participant.position.price = targetPrice
      
      // Volume affects confidence
      const avgVolume = 1000000 // Baseline volume
      const volumeRatio = marketVolume / avgVolume
      participant.confidence = Math.min(1, participant.confidence * (0.9 + volumeRatio * 0.1))
    }
  }
  
  /**
   * Calculate separation force (avoid crowding)
   */
  private calculateSeparation(participant: MarketParticipant, neighbors: MarketParticipant[]): number {
    if (neighbors.length === 0) return 0
    
    let separationForce = 0
    let tooClose = 0
    
    for (const neighbor of neighbors) {
      const distance = Math.abs(participant.position.price - neighbor.position.price) / participant.position.price
      
      // If too close, apply repulsion
      const minDistance = this.config.communicationRadius * 0.2 // 20% of communication radius
      if (distance < minDistance) {
        const repulsion = (minDistance - distance) / minDistance
        const direction = participant.position.price > neighbor.position.price ? 1 : -1
        separationForce += direction * repulsion * 0.01
        tooClose++
      }
    }
    
    // Normalize by number of close neighbors
    return tooClose > 0 ? separationForce / tooClose : 0
  }
  
  /**
   * Calculate alignment force (follow group direction)
   */
  private calculateAlignment(participant: MarketParticipant, neighbors: MarketParticipant[]): number {
    if (neighbors.length === 0) return 0
    
    // Calculate average velocity of neighbors
    const avgVelocity = neighbors.reduce((sum, n) => sum + n.velocity, 0) / neighbors.length
    
    // Force to align with group velocity
    const velocityDiff = avgVelocity - participant.velocity
    
    // Weight by neighbor influence
    const avgInfluence = neighbors.reduce((sum, n) => sum + n.influence, 0) / neighbors.length
    
    return velocityDiff * 0.1 * avgInfluence
  }
  
  /**
   * Calculate cohesion force (move toward group center)
   */
  private calculateCohesion(participant: MarketParticipant, neighbors: MarketParticipant[]): number {
    if (neighbors.length === 0) return 0
    
    // Calculate center of mass of neighbors
    const avgPrice = neighbors.reduce((sum, n) => sum + n.position.price, 0) / neighbors.length
    
    // Force toward center
    const priceDeviation = (avgPrice - participant.position.price) / participant.position.price
    
    // Stronger cohesion when confidence is low (seek safety in numbers)
    const cohesionStrength = 1 - participant.confidence
    
    return priceDeviation * 0.05 * cohesionStrength
  }
  
  /**
   * Detect emergent patterns in swarm behavior
   */
  private detectEmergentPatterns(swarmState: SwarmState): EmergentPattern[] {
    const patterns: EmergentPattern[] = []
    const participants = Array.from(this.swarmState.participants.values())
    
    // Pattern 1: Convergence/Divergence
    if (swarmState.cohesion > 0.8 && swarmState.alignment > 0.7) {
      patterns.push({
        type: 'convergence',
        strength: (swarmState.cohesion + swarmState.alignment) / 2,
        direction: swarmState.momentum > 0 ? 'up' : swarmState.momentum < 0 ? 'down' : 'sideways',
        participants: participants.filter(p => Math.abs(p.velocity - swarmState.momentum) < 0.01).map(p => p.id),
        duration: Date.now() // Track when pattern started
      })
    } else if (swarmState.cohesion < 0.3 && swarmState.alignment < 0.3) {
      patterns.push({
        type: 'divergence',
        strength: 1 - (swarmState.cohesion + swarmState.alignment) / 2,
        direction: 'sideways',
        participants: participants.map(p => p.id),
        duration: Date.now()
      })
    }
    
    // Pattern 2: Migration (collective movement)
    if (Math.abs(swarmState.momentum) > 0.02 && swarmState.alignment > 0.6) {
      patterns.push({
        type: 'migration',
        strength: Math.abs(swarmState.momentum) * 10 * swarmState.alignment,
        direction: swarmState.momentum > 0 ? 'up' : 'down',
        participants: participants.filter(p => Math.sign(p.velocity) === Math.sign(swarmState.momentum)).map(p => p.id),
        duration: Date.now()
      })
    }
    
    // Pattern 3: Fragmentation (multiple clusters)
    const clusters = this.identifyClusters(participants)
    if (clusters.length > 1) {
      patterns.push({
        type: 'fragmentation',
        strength: Math.min(1, clusters.length / 5), // More clusters = stronger fragmentation
        direction: 'sideways',
        participants: participants.map(p => p.id),
        duration: Date.now()
      })
    }
    
    // Pattern 4: Flocking (tight group movement)
    if (swarmState.cohesion > 0.7 && swarmState.alignment > 0.8 && swarmState.separation > 0.001) {
      patterns.push({
        type: 'flocking',
        strength: (swarmState.cohesion + swarmState.alignment) / 2,
        direction: swarmState.momentum > 0 ? 'up' : swarmState.momentum < 0 ? 'down' : 'sideways',
        participants: participants.map(p => p.id),
        duration: Date.now()
      })
    }
    
    return patterns
  }
  
  /**
   * Identify distinct clusters in the swarm
   */
  private identifyClusters(participants: MarketParticipant[]): MarketParticipant[][] {
    const clusters: MarketParticipant[][] = []
    const visited = new Set<string>()
    
    for (const participant of participants) {
      if (visited.has(participant.id)) continue
      
      const cluster: MarketParticipant[] = []
      const queue = [participant]
      
      while (queue.length > 0) {
        const current = queue.shift()!
        if (visited.has(current.id)) continue
        
        visited.add(current.id)
        cluster.push(current)
        
        // Add neighbors to queue
        for (const neighborId of current.neighbors) {
          const neighbor = this.swarmState.participants.get(neighborId)
          if (neighbor && !visited.has(neighbor.id)) {
            queue.push(neighbor)
          }
        }
      }
      
      if (cluster.length > 0) {
        clusters.push(cluster)
      }
    }
    
    return clusters
  }
  
  
  /**
   * Calculate swarm diversity/consensus
   */
  private calculateSwarmConsensus(): {
    consensus: number // 0-1
    diversity: number // 0-1
    polarization: number // 0-1
  } {
    const participants = Array.from(this.swarmState.participants.values())
    
    // Calculate bias distribution
    const bullishCount = participants.filter(p => p.bias === 'bullish').length
    const bearishCount = participants.filter(p => p.bias === 'bearish').length
    const neutralCount = participants.filter(p => p.bias === 'neutral').length
    
    // Calculate velocity consensus
    const buyingCount = participants.filter(p => p.velocity > 0.001).length
    const sellingCount = participants.filter(p => p.velocity < -0.001).length
    const holdingCount = participants.filter(p => Math.abs(p.velocity) <= 0.001).length
    
    // Consensus based on velocity alignment
    const maxVelocityGroup = Math.max(buyingCount, sellingCount, holdingCount)
    const velocityConsensus = maxVelocityGroup / participants.length
    
    // Diversity based on bias distribution
    const biasEntropy = this.calculateEntropy([bullishCount, bearishCount, neutralCount])
    const diversity = biasEntropy / Math.log(3) // Normalize to 0-1
    
    // Polarization (how much bulls vs bears)
    const polarization = Math.abs(bullishCount - bearishCount) / participants.length
    
    // Overall consensus combines velocity and bias alignment
    const consensus = velocityConsensus * 0.7 + (1 - diversity) * 0.3
    
    return { consensus, diversity, polarization }
  }
  
  /**
   * Calculate entropy for diversity measurement
   */
  private calculateEntropy(counts: number[]): number {
    const total = counts.reduce((a, b) => a + b, 0)
    if (total === 0) return 0
    
    let entropy = 0
    for (const count of counts) {
      if (count > 0) {
        const p = count / total
        entropy -= p * Math.log(p)
      }
    }
    return entropy
  }
  
  /**
   * Generate trading signal from swarm behavior
   */
  private generateSwarmSignal(
    patterns: EmergentPattern[],
    consensus: { consensus: number, diversity: number, polarization: number },
    swarmMetrics: SwarmState
  ): AgentSignal {
    let action: 'buy' | 'sell' | 'hold' = 'hold'
    let confidence = 0.5
    let reasoning = ''
    
    // Find strongest pattern
    const strongestPattern = patterns.reduce((max, p) => 
      p.strength > (max?.strength ?? 0) ? p : max, patterns[0]
    )
    
    if (strongestPattern) {
      // Map patterns to actions
      switch (strongestPattern.type) {
        case 'convergence':
          if (strongestPattern.direction === 'up') {
            action = 'buy'
            confidence = 0.6 + strongestPattern.strength * 0.3
            reasoning = `Swarm converging upward (${(strongestPattern.strength * 100).toFixed(0)}% strength)`
          } else if (strongestPattern.direction === 'down') {
            action = 'sell'
            confidence = 0.6 + strongestPattern.strength * 0.3
            reasoning = `Swarm converging downward (${(strongestPattern.strength * 100).toFixed(0)}% strength)`
          }
          break
          
        case 'migration':
          action = strongestPattern.direction === 'up' ? 'buy' : 'sell'
          confidence = 0.65 + strongestPattern.strength * 0.25
          reasoning = `Swarm migrating ${strongestPattern.direction} (${(consensus.consensus * 100).toFixed(0)}% consensus)`
          break
          
        case 'divergence':
        case 'fragmentation':
          action = 'hold'
          confidence = 0.7 + strongestPattern.strength * 0.2
          reasoning = `Market fragmented (${(consensus.diversity * 100).toFixed(0)}% diversity), avoid trading`
          break
          
        case 'flocking':
          // Flocking suggests strong trend but be cautious of reversal
          if (Math.abs(swarmMetrics.momentum) > 0.03) {
            // Strong momentum - fade it
            action = swarmMetrics.momentum > 0 ? 'sell' : 'buy'
            confidence = 0.55 + strongestPattern.strength * 0.2
            reasoning = `Swarm overcrowded, fading the move`
          } else {
            // Moderate momentum - follow it
            action = swarmMetrics.momentum > 0 ? 'buy' : 'sell'
            confidence = 0.6 + strongestPattern.strength * 0.2
            reasoning = `Following swarm flock (cohesion: ${(swarmMetrics.cohesion * 100).toFixed(0)}%)`
          }
          break
      }
    }
    
    // Adjust confidence based on consensus
    if (consensus.consensus > 0.8 && action !== 'hold') {
      // Very high consensus might mean overcrowded trade
      confidence *= 0.8
      reasoning += `. High consensus (${(consensus.consensus * 100).toFixed(0)}%) - be cautious`
    } else if (consensus.consensus < 0.3) {
      // Low consensus - reduce confidence
      confidence *= 0.7
      reasoning += `. Low consensus (${(consensus.consensus * 100).toFixed(0)}%)`
    }
    
    return this.createSignal(
      action,
      Math.min(0.9, confidence),
      reasoning + `. Momentum: ${(swarmMetrics.momentum * 100).toFixed(1)}%`
    )
  }
  
  protected async onReset(): Promise<void> {
    this.swarmState = {
      participants: new Map(),
      centerOfMass: 0,
      momentum: 0,
      cohesion: 0,
      alignment: 0,
      separation: 0
    }
    this.emergentPatterns = []
    this.initializeSwarm()
  }
}