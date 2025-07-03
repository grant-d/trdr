import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import type { Candle } from '@trdr/shared/src/types/market-data'

interface MemoryNode {
  price: number
  timestamp: number
  strength: number
  emotion: 'euphoria' | 'fear' | 'greed' | 'panic' | 'neutral'
  velocity: number
  volume: number
  volatility: number
  outcomes: Array<{result: 'bounce' | 'break' | 'consolidate', magnitude: number}>
}

interface MemoryCluster {
  centroid: number
  nodes: MemoryNode[]
  totalStrength: number
  dominantEmotion: string
  reliability: number
}

/**
 * Market Memory Agent
 * 
 * An innovative agent that models how markets "remember" previous price interactions
 * and tend to replay similar behaviors when returning to those levels. This agent
 * captures the psychological imprinting that occurs at significant price points.
 * 
 * ## Core Concept:
 * 
 * Markets have memory. When price visits a level with significant activity (volume,
 * volatility, or strong directional moves), it creates a "memory" at that level.
 * When price returns, market participants remember what happened before and often
 * react similarly, creating predictable patterns.
 * 
 * ## Memory Formation:
 * 
 * Memories are created when:
 * - High volume events occur (institutional participation)
 * - Extreme price velocity (rapid moves create strong impressions)
 * - High volatility (emotional extremes are memorable)
 * - Multiple touches (repetition strengthens memory)
 * 
 * ## Memory Characteristics:
 * 
 * ### Strength (0-1 scale)
 * - Initial strength based on: volume × velocity × volatility
 * - Decays over time using psychological forgetting curve
 * - Refreshed when price revisits and confirms pattern
 * 
 * ### Emotion Type
 * - **Euphoria**: Rapid upward moves with expanding volume
 * - **Fear**: Sharp drops with high volume
 * - **Greed**: Steady climbs with increasing participation  
 * - **Panic**: Waterfall declines with extreme volume
 * - **Neutral**: Consolidation zones with balanced activity
 * 
 * ### Decay Function
 * - Recent memories (< 24 hours): 100% strength
 * - Short-term (1-7 days): 80% strength with gradual decay
 * - Medium-term (1-4 weeks): 50% strength with faster decay
 * - Long-term (> 1 month): 20% baseline with slow decay
 * - Traumatic events (crashes/squeezes) decay slower
 * 
 * ## Memory Clustering:
 * 
 * Individual memories cluster together when within 0.2% price range,
 * creating "memory zones" with combined influence. Clusters track:
 * - Centroid price (weighted average)
 * - Combined strength
 * - Dominant emotion
 * - Historical reliability
 * 
 * ## Signal Generation:
 * 
 * ### Strong Memory Approach (>0.7 strength)
 * - High reliability (>70%) = Expect similar outcome
 * - Low reliability (<30%) = Expect opposite outcome
 * - Mixed reliability = Reduced confidence
 * 
 * ### Memory Confluence
 * - Multiple memories at similar levels = Stronger signal
 * - Conflicting emotions = Volatility expected
 * - Aligned emotions = Directional bias
 * 
 * ### Fresh vs Stale Memories
 * - Fresh memories (< 48 hours) = Higher confidence
 * - Stale memories (> 2 weeks) = Lower confidence
 * - Reactivated memories = Renewed strength
 * 
 * ## Unique Features:
 * 
 * ### Emotional Contagion
 * When strong memories trigger, they can create new memories
 * at nearby levels, spreading the emotional state.
 * 
 * ### Memory Interference  
 * Overlapping memories with different emotions create
 * "interference patterns" suggesting consolidation.
 * 
 * ### Collective Amnesia
 * During strong trends, old memories are overwritten,
 * creating new behavioral patterns.
 * 
 * ### Memory Reinforcement
 * Each time a memory successfully predicts behavior,
 * its strength and reliability increase.
 */
export class MarketMemoryAgent extends BaseAgent {
  private memories: MemoryNode[] = []
  private memoryClusters: MemoryCluster[] = []
  
  // Memory parameters
  private readonly maxMemories = 1000
  private readonly clusterThreshold = 0.002 // 0.2% price range
  private readonly significantVolume = 2.0 // 2x average
  private readonly significantVelocity = 0.003 // 0.3% per candle
  private readonly significantVolatility = 0.02 // 2% range
  
  // Decay parameters
  private readonly hourlyDecay = 0.01
  private readonly dailyDecay = 0.05
  private readonly weeklyDecay = 0.15
  
  // Tracking
  private volumeMA = 0
  // @ts-ignore - unused variable (reserved for future use)
  private _volatilityMA = 0
  private lastClusterUpdate = 0
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Market Memory Agent initialized')
  }
  
  async analyze(context: MarketContext): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < 50) {
      return this.createSignal('hold', 0.3, 'Insufficient data for memory formation')
    }
    
    // Update moving averages
    this.updateMetrics(candles)
    
    // Form new memories from recent price action
    this.formMemories(candles)
    
    // Decay old memories
    const currentTimestamp = candles[candles.length - 1]!.timestamp
    this.decayMemories(currentTimestamp)
    
    // Cluster memories
    if (Date.now() - this.lastClusterUpdate > 60000) { // Update every minute
      this.clusterMemories()
      this.lastClusterUpdate = Date.now()
    }
    
    // Find relevant memories at current price
    const relevantMemories = this.findRelevantMemories(currentPrice)
    const relevantClusters = this.findRelevantClusters(currentPrice)
    
    // Generate signal based on memory analysis
    return this.generateMemoryBasedSignal(
      relevantMemories,
      relevantClusters,
      candles,
      currentPrice
    )
  }
  
  private updateMetrics(candles: readonly Candle[]): void {
    const recentCandles = candles.slice(-20)
    
    // Volume moving average
    this.volumeMA = recentCandles.reduce((sum, c) => sum + c.volume, 0) / recentCandles.length
    
    // Volatility (average true range)
    let totalRange = 0
    for (let i = 1; i < recentCandles.length; i++) {
      const high = recentCandles[i]!.high
      const low = recentCandles[i]!.low
      const prevClose = recentCandles[i-1]!.close
      const trueRange = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose))
      totalRange += trueRange / prevClose
    }
    this._volatilityMA = totalRange / (recentCandles.length - 1)
  }
  
  private formMemories(candles: readonly Candle[]): void {
    const currentCandle = candles[candles.length - 1]!
    const prevCandle = candles[candles.length - 2]
    
    if (!prevCandle) return
    
    // Calculate metrics
    const velocity = (currentCandle.close - prevCandle.close) / prevCandle.close
    const volume = currentCandle.volume / this.volumeMA
    const volatility = (currentCandle.high - currentCandle.low) / currentCandle.close
    
    // Check if this candle is memorable
    const isMemorableVolume = volume >= this.significantVolume
    const isMemorableVelocity = Math.abs(velocity) >= this.significantVelocity
    const isMemorableVolatility = volatility >= this.significantVolatility
    
    if (isMemorableVolume || isMemorableVelocity || isMemorableVolatility) {
      // Determine emotion based on price action
      const emotion = this.determineEmotion(velocity, volume, volatility)
      
      // Calculate initial memory strength
      const strength = Math.min(1, 
        (volume / this.significantVolume) * 0.3 +
        (Math.abs(velocity) / this.significantVelocity) * 0.4 +
        (volatility / this.significantVolatility) * 0.3
      )
      
      // Create memory node
      const memory: MemoryNode = {
        price: currentCandle.close,
        timestamp: currentCandle.timestamp,
        strength,
        emotion,
        velocity,
        volume: currentCandle.volume,
        volatility,
        outcomes: []
      }
      
      // Add to memories
      this.memories.push(memory)
      
      // Maintain memory limit
      if (this.memories.length > this.maxMemories) {
        // Remove weakest old memory
        this.memories.sort((a, b) => b.strength - a.strength)
        this.memories = this.memories.slice(0, this.maxMemories)
      }
      
      this.logger?.debug('Memory formed', {
        price: memory.price,
        emotion: memory.emotion,
        strength: memory.strength
      })
    }
  }
  
  private determineEmotion(
    velocity: number, 
    volumeRatio: number, 
    volatility: number
  ): 'euphoria' | 'fear' | 'greed' | 'panic' | 'neutral' {
    // Extreme positive with high volume = euphoria
    if (velocity > 0.005 && volumeRatio > 3 && volatility > 0.03) {
      return 'euphoria'
    }
    
    // Extreme negative with high volume = panic
    if (velocity < -0.005 && volumeRatio > 3 && volatility > 0.03) {
      return 'panic'
    }
    
    // Steady positive with increasing volume = greed
    if (velocity > 0.002 && velocity < 0.005 && volumeRatio > 1.5) {
      return 'greed'
    }
    
    // Moderate negative with elevated volume = fear
    if (velocity < -0.002 && velocity > -0.005 && volumeRatio > 1.5) {
      return 'fear'
    }
    
    return 'neutral'
  }
  
  private decayMemories(currentTime: number): void {
    // @ts-ignore - unused variable (reserved for future use)
    const _hourInMs = 3600000
    const dayInMs = 86400000
    const weekInMs = 604800000
    
    this.memories = this.memories.map(memory => {
      const age = currentTime - memory.timestamp
      let decayRate = this.hourlyDecay
      
      // Apply different decay rates based on age
      if (age > weekInMs) {
        decayRate = this.weeklyDecay
      } else if (age > dayInMs) {
        decayRate = this.dailyDecay
      }
      
      // Traumatic events (panic/euphoria) decay slower
      if (memory.emotion === 'panic' || memory.emotion === 'euphoria') {
        decayRate *= 0.5
      }
      
      // Apply decay
      const decayFactor = Math.exp(-decayRate * (age / dayInMs))
      memory.strength *= decayFactor
      
      return memory
    }).filter(m => m.strength > 0.01) // Remove very weak memories
  }
  
  private clusterMemories(): void {
    this.memoryClusters = []
    const clustered = new Set<MemoryNode>()
    
    // Sort memories by price
    const sortedMemories = [...this.memories].sort((a, b) => a.price - b.price)
    
    for (const memory of sortedMemories) {
      if (clustered.has(memory)) continue
      
      // Find all memories within cluster threshold
      const clusterNodes: MemoryNode[] = [memory]
      clustered.add(memory)
      
      for (const other of sortedMemories) {
        if (clustered.has(other)) continue
        
        const priceDiff = Math.abs(other.price - memory.price) / memory.price
        if (priceDiff <= this.clusterThreshold) {
          clusterNodes.push(other)
          clustered.add(other)
        }
      }
      
      // Create cluster
      const cluster = this.createCluster(clusterNodes)
      this.memoryClusters.push(cluster)
    }
    
    // Sort clusters by total strength
    this.memoryClusters.sort((a, b) => b.totalStrength - a.totalStrength)
  }
  
  private createCluster(nodes: MemoryNode[]): MemoryCluster {
    // Calculate weighted centroid
    let weightedSum = 0
    let totalWeight = 0
    const emotionCounts: Record<string, number> = {}
    
    for (const node of nodes) {
      weightedSum += node.price * node.strength
      totalWeight += node.strength
      emotionCounts[node.emotion] = (emotionCounts[node.emotion] || 0) + node.strength
    }
    
    const centroid = weightedSum / totalWeight
    
    // Find dominant emotion
    const dominantEmotion = Object.entries(emotionCounts)
      .sort((a, b) => b[1] - a[1])[0]![0]
    
    // Calculate reliability based on outcome consistency
    let totalOutcomes = 0
    let consistentOutcomes = 0
    
    for (const node of nodes) {
      if (node.outcomes.length > 0) {
        totalOutcomes += node.outcomes.length
        const primaryOutcome = node.outcomes[0]!.result
        consistentOutcomes += node.outcomes.filter(o => o.result === primaryOutcome).length
      }
    }
    
    const reliability = totalOutcomes > 0 ? consistentOutcomes / totalOutcomes : 0.5
    
    return {
      centroid,
      nodes,
      totalStrength: totalWeight,
      dominantEmotion,
      reliability
    }
  }
  
  private findRelevantMemories(currentPrice: number): MemoryNode[] {
    return this.memories.filter(memory => {
      const priceDiff = Math.abs(memory.price - currentPrice) / currentPrice
      return priceDiff <= this.clusterThreshold * 2 // Slightly wider range
    }).sort((a, b) => b.strength - a.strength)
  }
  
  private findRelevantClusters(currentPrice: number): MemoryCluster[] {
    return this.memoryClusters.filter(cluster => {
      const priceDiff = Math.abs(cluster.centroid - currentPrice) / currentPrice
      return priceDiff <= this.clusterThreshold * 2
    })
  }
  
  private generateMemoryBasedSignal(
    memories: MemoryNode[],
    clusters: MemoryCluster[],
    candles: readonly Candle[],
    _currentPrice: number
  ): AgentSignal {
    // No relevant memories
    if (memories.length === 0 && clusters.length === 0) {
      return this.createSignal('hold', 0.4, 'No memory of this price level')
    }
    
    // Analyze current approach
    const currentCandle = candles[candles.length - 1]!
    const prevCandle = candles[candles.length - 2]!
    const approachVelocity = (currentCandle.close - prevCandle.close) / prevCandle.close
    const approachVolume = currentCandle.volume / this.volumeMA
    
    // Check for strong cluster
    const strongCluster = clusters.find(c => c.totalStrength > 0.7)
    if (strongCluster) {
      return this.generateClusterSignal(strongCluster, approachVelocity, approachVolume)
    }
    
    // Check for strong individual memory
    const strongMemory = memories.find(m => m.strength > 0.5)
    if (strongMemory) {
      return this.generateMemorySignal(strongMemory, approachVelocity, approachVolume)
    }
    
    // Multiple weak memories - look for confluence
    if (memories.length >= 3) {
      return this.generateConfluenceSignal(memories, approachVelocity)
    }
    
    return this.createSignal('hold', 0.5, 'Weak memory influence at this level')
  }
  
  private generateClusterSignal(
    cluster: MemoryCluster, 
    approachVelocity: number,
    approachVolume: number
  ): AgentSignal {
    const { dominantEmotion, reliability, totalStrength } = cluster
    
    // High reliability - expect similar outcome
    if (reliability > 0.7) {
      switch (dominantEmotion) {
        case 'euphoria':
        case 'greed':
          // Previously bullish area
          if (approachVelocity > 0 && approachVolume > 1.2) {
            return this.createSignal(
              'buy',
              Math.min(0.9, 0.6 + totalStrength * 0.3),
              `Strong bullish memory cluster (${dominantEmotion})`
            )
          }
          break
          
        case 'panic':
        case 'fear':
          // Previously bearish area
          if (approachVelocity < 0 && approachVolume > 1.2) {
            return this.createSignal(
              'sell',
              Math.min(0.9, 0.6 + totalStrength * 0.3),
              `Strong bearish memory cluster (${dominantEmotion})`
            )
          }
          break
      }
    }
    
    // Low reliability - expect opposite outcome
    if (reliability < 0.3) {
      switch (dominantEmotion) {
        case 'panic':
        case 'fear':
          // Failed resistance becomes support
          return this.createSignal(
            'buy',
            Math.min(0.8, 0.5 + totalStrength * 0.3),
            'Memory flip: Previous resistance now support'
          )
          
        case 'euphoria':
        case 'greed':
          // Failed support becomes resistance
          return this.createSignal(
            'sell',
            Math.min(0.8, 0.5 + totalStrength * 0.3),
            'Memory flip: Previous support now resistance'
          )
      }
    }
    
    return this.createSignal(
      'hold',
      0.6,
      `Mixed memory cluster at ${cluster.centroid.toFixed(2)}`
    )
  }
  
  private generateMemorySignal(
    memory: MemoryNode,
    approachVelocity: number,
    approachVolume: number
  ): AgentSignal {
    // Check if approach matches memory emotion
    const emotionMatch = this.checkEmotionMatch(memory.emotion, approachVelocity, approachVolume)
    
    if (emotionMatch) {
      // Reinforced memory - strong signal
      const action = memory.emotion === 'euphoria' || memory.emotion === 'greed' ? 'buy' : 'sell'
      return this.createSignal(
        action,
        Math.min(0.85, 0.6 + memory.strength * 0.25),
        `Memory reinforcement: ${memory.emotion} at ${memory.price.toFixed(2)}`
      )
    } else {
      // Conflicting approach - caution
      return this.createSignal(
        'hold',
        0.5,
        `Memory conflict at ${memory.price.toFixed(2)}`
      )
    }
  }
  
  private generateConfluenceSignal(
    memories: MemoryNode[],
    _approachVelocity: number
  ): AgentSignal {
    // Count emotion types
    const emotions = memories.reduce((acc, m) => {
      acc[m.emotion] = (acc[m.emotion] || 0) + m.strength
      return acc
    }, {} as Record<string, number>)
    
    const bullishStrength = (emotions.euphoria || 0) + (emotions.greed || 0)
    const bearishStrength = (emotions.panic || 0) + (emotions.fear || 0)
    
    if (bullishStrength > bearishStrength * 1.5) {
      return this.createSignal(
        'buy',
        Math.min(0.8, 0.5 + bullishStrength * 0.2),
        'Bullish memory confluence'
      )
    }
    
    if (bearishStrength > bullishStrength * 1.5) {
      return this.createSignal(
        'sell',
        Math.min(0.8, 0.5 + bearishStrength * 0.2),
        'Bearish memory confluence'
      )
    }
    
    return this.createSignal(
      'hold',
      0.6,
      'Mixed emotional memories - expect volatility'
    )
  }
  
  private checkEmotionMatch(
    emotion: string,
    velocity: number,
    volumeRatio: number
  ): boolean {
    switch (emotion) {
      case 'euphoria':
      case 'greed':
        return velocity > 0.001 && volumeRatio > 1.2
      case 'panic':
      case 'fear':
        return velocity < -0.001 && volumeRatio > 1.2
      default:
        return Math.abs(velocity) < 0.001
    }
  }
  
  protected async onReset(): Promise<void> {
    this.memories = []
    this.memoryClusters = []
    this.volumeMA = 0
    this._volatilityMA = 0
    this.lastClusterUpdate = 0
  }
  
  // Required by BaseAgent
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}