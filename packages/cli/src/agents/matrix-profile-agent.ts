import { BaseAgent } from '@trdr/core'
import type { AgentMetadata, AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'
import type { Logger } from '@trdr/types'

interface MatrixProfileConfig {
  /** Window size for subsequence matching */
  windowSize?: number
  /** Minimum window size (adaptive) */
  minWindowSize?: number
  /** Maximum window size (adaptive) */
  maxWindowSize?: number
  /** Threshold for anomaly detection (in standard deviations) */
  anomalyThreshold?: number
  /** Threshold for motif detection (distance) */
  motifThreshold?: number
  /** Number of top motifs to track */
  topMotifsCount?: number
  /** Enable adaptive window sizing */
  adaptiveWindow?: boolean
  /** Minimum pattern matches for signal */
  minPatternMatches?: number
}

interface MatrixProfilePoint {
  index: number
  distance: number // Minimum distance to nearest neighbor
  nearestNeighborIndex: number
  isAnomaly: boolean
  isMotif: boolean
}

interface Motif {
  indices: number[] // Indices of subsequences in this motif
  distance: number // Average distance between members
  occurrences: number
  pattern: number[] // The actual pattern
  lastSeen: number // Last index where pattern was seen
}

interface Discord {
  index: number
  distance: number // How anomalous (larger = more anomalous)
  pattern: number[]
  timestamp: number
}

interface MatrixProfileAnalysis {
  profile: MatrixProfilePoint[]
  motifs: Motif[]
  discords: Discord[]
  windowSize: number
  meanDistance: number
  stdDistance: number
}

/**
 * Matrix Profile Agent
 * 
 * Implements a simplified version of the STUMPY algorithm for time series
 * pattern discovery. Matrix Profile is a data structure that annotates a 
 * time series with the distance to its nearest neighbor under z-normalized
 * Euclidean distance.
 * 
 * **Core Concepts:**
 * - **Matrix Profile**: For each subsequence, stores distance to nearest neighbor
 * - **Motifs**: Repeated patterns (subsequences with small distances)
 * - **Discords**: Anomalies (subsequences with large distances)
 * - **Z-normalization**: Makes patterns comparable regardless of scale/offset
 * - **Sliding Window**: Analyzes all possible subsequences of given length
 * 
 * **Key Features:**
 * - **Pattern Discovery**: Automatically finds repeated patterns (motifs)
 * - **Anomaly Detection**: Identifies unusual patterns (discords)
 * - **Self-Join**: Compares time series with itself to find patterns
 * - **Adaptive Windows**: Adjusts pattern length based on market conditions
 * - **Real-time Updates**: Efficiently updates profile with new data
 * 
 * **Trading Signals:**
 * - **Buy on Motifs**: When current pattern matches profitable historical patterns
 * - **Sell on Discords**: When detecting anomalous/unprecedented patterns
 * - **Trend Confirmation**: Multiple motif occurrences confirm trend
 * - **Reversal Detection**: Discord after stable motif suggests reversal
 * 
 * **Algorithm (Simplified STUMPY):**
 * 1. Extract all subsequences of length m from time series
 * 2. Z-normalize each subsequence (zero mean, unit variance)
 * 3. For each subsequence, find distance to nearest neighbor
 * 4. Identify motifs (low distances) and discords (high distances)
 * 5. Generate signals based on pattern matches and anomalies
 * 
 * @todo Implement efficient distance calculations using sliding dot product
 * @todo Add z-normalization for scale-invariant pattern matching
 * @todo Implement motif discovery (clustering similar patterns)
 * @todo Add discord detection for anomaly identification
 * @todo Create adaptive window sizing based on volatility
 * @todo Optimize with early abandoning for real-time performance
 */
export class MatrixProfileAgent extends BaseAgent {
  protected readonly config: Required<MatrixProfileConfig>
  private priceBuffer: number[] = []
  private matrixProfileHistory: MatrixProfileAnalysis[] = []
  private currentMotifs: Motif[] = []
  private recentDiscords: Discord[] = []
  private readonly distanceCache = new Map<string, number>()
  
  constructor(metadata: AgentMetadata, logger?: Logger, config?: MatrixProfileConfig) {
    super(metadata, logger)
    
    this.config = {
      windowSize: config?.windowSize ?? 10,
      minWindowSize: config?.minWindowSize ?? 5,
      maxWindowSize: config?.maxWindowSize ?? 20,
      anomalyThreshold: config?.anomalyThreshold ?? 2.0, // 2.0 standard deviations
      motifThreshold: config?.motifThreshold ?? 0.5, // More lenient for pattern matching
      topMotifsCount: config?.topMotifsCount ?? 5,
      adaptiveWindow: config?.adaptiveWindow ?? true,
      minPatternMatches: config?.minPatternMatches ?? 2
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Matrix Profile Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < this.config.windowSize * 2) {
      return this.createSignal('hold', 0.3, 'Insufficient data for matrix profile analysis')
    }
    
    // Update price buffer
    const prices = candles.map(c => c.close)
    this.priceBuffer = prices.slice(-100) // Keep last 100 prices
    
    // Determine adaptive window size if enabled
    const windowSize = this.config.adaptiveWindow ? 
      this.calculateAdaptiveWindowSize(prices) : 
      this.config.windowSize
    
    // Build matrix profile
    const analysis = this.buildMatrixProfile(this.priceBuffer, windowSize)
    this.matrixProfileHistory.push(analysis)
    
    // Keep history manageable
    if (this.matrixProfileHistory.length > 50) {
      this.matrixProfileHistory.shift()
    }
    
    // Update motifs and discords
    this.updateMotifs(analysis)
    this.updateDiscords(analysis)
    
    // Check for trading signals
    
    // 1. Check if current pattern is a known profitable motif
    const motifSignal = this.checkMotifSignal(currentPrice, analysis)
    if (motifSignal.action !== 'hold') {
      return motifSignal
    }
    
    // 2. Check for anomaly/discord signals
    const discordSignal = this.checkDiscordSignal(currentPrice, analysis)
    if (discordSignal.action !== 'hold') {
      return enforceNoShorting(discordSignal, context)
    }
    
    // 3. Check for pattern evolution signals
    const evolutionSignal = this.checkPatternEvolution(analysis)
    if (evolutionSignal.action !== 'hold') {
      return enforceNoShorting(evolutionSignal, context)
    }
    
    // Default signal with pattern information
    return this.createSignal(
      'hold',
      0.5,
      `Matrix Profile: ${analysis.motifs.length} motifs, ` +
      `${analysis.discords.length} discords, ` +
      `window: ${analysis.windowSize}`
    )
  }
  
  /**
   * Build matrix profile for time series
   */
  private buildMatrixProfile(prices: number[], windowSize: number): MatrixProfileAnalysis {
    const n = prices.length
    const profileLength = n - windowSize + 1
    const profile: MatrixProfilePoint[] = []
    
    // Calculate distances for each subsequence
    for (let i = 0; i < profileLength; i++) {
      const subsequence = prices.slice(i, i + windowSize)
      const normalized = this.zNormalize(subsequence)
      
      let minDistance = Infinity
      let nearestNeighborIndex = -1
      
      // Find nearest neighbor (excluding trivial matches)
      for (let j = 0; j < profileLength; j++) {
        // Skip trivial matches (overlapping windows)
        if (Math.abs(i - j) < windowSize / 2) continue
        
        const otherSubsequence = prices.slice(j, j + windowSize)
        const otherNormalized = this.zNormalize(otherSubsequence)
        
        const distance = this.euclideanDistance(normalized, otherNormalized)
        
        if (distance < minDistance) {
          minDistance = distance
          nearestNeighborIndex = j
        }
      }
      
      profile.push({
        index: i,
        distance: minDistance,
        nearestNeighborIndex,
        isAnomaly: false, // Set later
        isMotif: false // Set later
      })
    }
    
    // Calculate statistics for anomaly detection
    const distances = profile.map(p => p.distance)
    const meanDistance = distances.reduce((a, b) => a + b, 0) / distances.length
    const variance = distances.reduce((sum, d) => sum + Math.pow(d - meanDistance, 2), 0) / distances.length
    const stdDistance = Math.sqrt(variance)
    
    // Mark anomalies and motifs
    profile.forEach(point => {
      point.isAnomaly = point.distance > meanDistance + this.config.anomalyThreshold * stdDistance
      point.isMotif = point.distance < Math.max(this.config.motifThreshold, meanDistance * 0.8) // More lenient motif detection
    })
    
    // Extract motifs and discords
    const motifs = this.extractMotifs(prices, profile, windowSize)
    const discords = this.extractDiscords(prices, profile, windowSize)
    
    return {
      profile,
      motifs,
      discords,
      windowSize,
      meanDistance,
      stdDistance
    }
  }
  
  /**
   * Z-normalize a subsequence (zero mean, unit variance)
   */
  private zNormalize(subsequence: number[]): number[] {
    const mean = subsequence.reduce((a, b) => a + b, 0) / subsequence.length
    const variance = subsequence.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / subsequence.length
    const std = Math.sqrt(variance)
    
    // Handle zero variance
    if (std === 0) return subsequence.map(() => 0)
    
    return subsequence.map(x => (x - mean) / std)
  }
  
  /**
   * Calculate Euclidean distance between normalized subsequences
   */
  private euclideanDistance(a: number[], b: number[]): number {
    const cacheKey = `${a.join(',')}-${b.join(',')}`
    if (this.distanceCache.has(cacheKey)) {
      return this.distanceCache.get(cacheKey)!
    }
    
    let sum = 0
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i]! - b[i]!, 2)
    }
    const distance = Math.sqrt(sum)
    
    this.distanceCache.set(cacheKey, distance)
    return distance
  }
  
  /**
   * Extract motifs (repeated patterns) from matrix profile
   */
  private extractMotifs(
    prices: number[], 
    profile: MatrixProfilePoint[], 
    windowSize: number
  ): Motif[] {
    const motifCandidates = profile.filter(p => p.isMotif)
    const motifs: Motif[] = []
    const processed = new Set<number>()
    
    // Group similar patterns into motifs
    for (const candidate of motifCandidates) {
      if (processed.has(candidate.index)) continue
      
      const motifIndices = [candidate.index]
      const pattern = prices.slice(candidate.index, candidate.index + windowSize)
      processed.add(candidate.index)
      
      // Find all similar patterns
      for (const other of motifCandidates) {
        if (processed.has(other.index)) continue
        
        const otherPattern = prices.slice(other.index, other.index + windowSize)
        const distance = this.euclideanDistance(
          this.zNormalize(pattern),
          this.zNormalize(otherPattern)
        )
        
        if (distance < this.config.motifThreshold) {
          motifIndices.push(other.index)
          processed.add(other.index)
        }
      }
      
      if (motifIndices.length >= 2) {
        // Calculate average distance between motif members
        let totalDistance = 0
        let comparisons = 0
        
        for (let i = 0; i < motifIndices.length; i++) {
          for (let j = i + 1; j < motifIndices.length; j++) {
            const idx1 = motifIndices[i]!
            const idx2 = motifIndices[j]!
            const pattern1 = prices.slice(idx1, idx1 + windowSize)
            const pattern2 = prices.slice(idx2, idx2 + windowSize)
            totalDistance += this.euclideanDistance(
              this.zNormalize(pattern1),
              this.zNormalize(pattern2)
            )
            comparisons++
          }
        }
        
        motifs.push({
          indices: motifIndices,
          distance: comparisons > 0 ? totalDistance / comparisons : 0,
          occurrences: motifIndices.length,
          pattern,
          lastSeen: Math.max(...motifIndices)
        })
      }
    }
    
    // Sort by occurrences (most frequent first)
    return motifs.sort((a, b) => b.occurrences - a.occurrences).slice(0, this.config.topMotifsCount)
  }
  
  /**
   * Extract discords (anomalies) from matrix profile
   */
  private extractDiscords(
    prices: number[], 
    profile: MatrixProfilePoint[], 
    windowSize: number
  ): Discord[] {
    const discords: Discord[] = []
    const anomalies = profile.filter(p => p.isAnomaly)
    
    // Sort by distance (most anomalous first)
    anomalies.sort((a, b) => b.distance - a.distance)
    
    // Take top anomalies
    for (let i = 0; i < Math.min(5, anomalies.length); i++) {
      const anomaly = anomalies[i]!
      discords.push({
        index: anomaly.index,
        distance: anomaly.distance,
        pattern: prices.slice(anomaly.index, anomaly.index + windowSize),
        timestamp: Date.now()
      })
    }
    
    return discords
  }
  
  /**
   * Calculate adaptive window size based on market conditions
   */
  private calculateAdaptiveWindowSize(prices: number[]): number {
    // Calculate recent volatility
    const recentPrices = prices.slice(-30)
    const returns = []
    
    for (let i = 1; i < recentPrices.length; i++) {
      returns.push((recentPrices[i]! - recentPrices[i-1]!) / recentPrices[i-1]!)
    }
    
    const volatility = Math.sqrt(
      returns.reduce((sum, r) => sum + r * r, 0) / returns.length
    )
    
    // Higher volatility = smaller window (react faster)
    // Lower volatility = larger window (capture longer patterns)
    const normalizedVol = Math.min(1, volatility * 100) // Normalize to 0-1
    const windowSize = Math.round(
      this.config.maxWindowSize - 
      (this.config.maxWindowSize - this.config.minWindowSize) * normalizedVol
    )
    
    return Math.max(this.config.minWindowSize, Math.min(this.config.maxWindowSize, windowSize))
  }
  
  /**
   * Update tracked motifs with new analysis
   */
  private updateMotifs(analysis: MatrixProfileAnalysis): void {
    // Merge new motifs with existing ones
    for (const newMotif of analysis.motifs) {
      const similar = this.currentMotifs.find(existing => {
        const distance = this.euclideanDistance(
          this.zNormalize(existing.pattern),
          this.zNormalize(newMotif.pattern)
        )
        return distance < this.config.motifThreshold
      })
      
      if (similar) {
        // Update existing motif
        similar.occurrences += newMotif.occurrences
        similar.lastSeen = newMotif.lastSeen
        similar.indices.push(...newMotif.indices)
      } else {
        // Add new motif
        this.currentMotifs.push(newMotif)
      }
    }
    
    // Keep only recent and frequent motifs
    this.currentMotifs = this.currentMotifs
      .filter(m => m.occurrences >= this.config.minPatternMatches)
      .sort((a, b) => b.occurrences - a.occurrences)
      .slice(0, this.config.topMotifsCount * 2)
  }
  
  /**
   * Update tracked discords with new analysis
   */
  private updateDiscords(analysis: MatrixProfileAnalysis): void {
    this.recentDiscords.push(...analysis.discords)
    
    // Keep only recent discords (last 20)
    this.recentDiscords = this.recentDiscords
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 20)
  }
  
  /**
   * Check if current pattern matches profitable motifs
   */
  private checkMotifSignal(_currentPrice: number, analysis: MatrixProfileAnalysis): AgentSignal {
    if (analysis.motifs.length === 0) {
      return this.createSignal('hold', 0.4, 'No motifs detected')
    }
    
    // Get the most recent pattern
    const recentIndex = analysis.profile.length - 1
    const recentPoint = analysis.profile[recentIndex]
    
    if (!recentPoint?.isMotif) {
      return this.createSignal('hold', 0.4, 'Current pattern is not a motif')
    }
    
    // Find which motif this belongs to
    const matchingMotif = analysis.motifs.find(motif => 
      motif.indices.includes(recentIndex)
    )
    
    if (matchingMotif && matchingMotif.occurrences >= this.config.minPatternMatches) {
      // Check historical performance after this motif
      const historicalReturns = this.analyzeMotifReturns(matchingMotif)
      
      if (historicalReturns.avgReturn > 0.0005 && historicalReturns.winRate > 0.55) {
        return this.createSignal(
          'buy',
          Math.min(0.85, 0.5 + matchingMotif.occurrences * 0.05),
          `Motif pattern detected: ${matchingMotif.occurrences} occurrences, ` +
          `${(historicalReturns.winRate * 100).toFixed(0)}% win rate`
        )
      } else if (historicalReturns.avgReturn < -0.0005 && historicalReturns.winRate < 0.45) {
        return this.createSignal(
          'sell',
          Math.min(0.85, 0.5 + matchingMotif.occurrences * 0.05),
          `Bearish motif: ${matchingMotif.occurrences} occurrences, ` +
          `${(historicalReturns.avgReturn * 100).toFixed(2)}% avg return`
        )
      }
    }
    
    return this.createSignal('hold', 0.5, 'Motif detected but insufficient confidence')
  }
  
  /**
   * Check for anomaly/discord signals
   */
  private checkDiscordSignal(_currentPrice: number, analysis: MatrixProfileAnalysis): AgentSignal {
    if (analysis.discords.length === 0) {
      return this.createSignal('hold', 0.4, 'No discords detected')
    }
    
    // Check if current pattern is anomalous
    const recentIndex = analysis.profile.length - 1
    const recentPoint = analysis.profile[recentIndex]
    
    if (!recentPoint?.isAnomaly) {
      return this.createSignal('hold', 0.4, 'Current pattern is normal')
    }
    
    // Anomaly after stable period suggests reversal
    const previousMotifs = this.currentMotifs.filter(m => 
      m.lastSeen < recentIndex && m.lastSeen > recentIndex - 20
    )
    
    if (previousMotifs.length > 0 && recentPoint.distance > analysis.meanDistance + 2 * analysis.stdDistance) {
      // Strong anomaly after stable pattern
      return this.createSignal(
        'sell',
        0.65,
        `Strong discord detected: ${recentPoint.distance.toFixed(3)} distance ` +
        `(${((recentPoint.distance - analysis.meanDistance) / analysis.stdDistance).toFixed(1)} std devs)`
      )
    }
    
    // Even without previous motifs, strong anomalies can signal opportunities
    if (recentPoint.distance > analysis.meanDistance + 2.5 * analysis.stdDistance) {
      return this.createSignal(
        'buy', // Counter-trend opportunity
        0.6,
        `Extreme discord: ${recentPoint.distance.toFixed(3)} distance, potential reversal`
      )
    }
    
    return this.createSignal('hold', 0.5, 'Discord detected but insufficient confidence')
  }
  
  /**
   * Check for pattern evolution signals
   */
  private checkPatternEvolution(_analysis: MatrixProfileAnalysis): AgentSignal {
    if (this.matrixProfileHistory.length < 5) {
      return this.createSignal('hold', 0.4, 'Insufficient pattern history')
    }
    
    // Check if motif patterns are evolving (changing)
    const recentHistory = this.matrixProfileHistory.slice(-5)
    const motifCounts = recentHistory.map(h => h.motifs.length)
    const discordCounts = recentHistory.map(h => h.discords.length)
    
    // Increasing discords suggests market regime change
    const discordTrend = this.calculateTrend(discordCounts)
    if (discordTrend > 0.5) {
      return this.createSignal(
        'sell',
        0.65,
        `Pattern breakdown: discord count increasing (${discordCounts[0]} → ${discordCounts[discordCounts.length - 1]})`
      )
    }
    
    // Decreasing motifs suggests loss of structure
    const motifTrend = this.calculateTrend(motifCounts)
    if (motifTrend < -0.5) {
      return this.createSignal(
        'sell',
        0.6,
        `Pattern dissolution: motif count decreasing (${motifCounts[0]} → ${motifCounts[motifCounts.length - 1]})`
      )
    }
    
    return this.createSignal('hold', 0.5, 'Pattern evolution stable')
  }
  
  /**
   * Analyze historical returns after motif occurrences
   */
  private analyzeMotifReturns(motif: Motif): { avgReturn: number, winRate: number } {
    const returns: number[] = []
    const windowSize = motif.pattern.length
    
    // Look at what happened after each motif occurrence
    for (const index of motif.indices) {
      if (index + windowSize + 5 < this.priceBuffer.length) {
        const priceAtMotif = this.priceBuffer[index + windowSize - 1]!
        const futurePrice = this.priceBuffer[index + windowSize + 5]! // 5 periods later
        const return_ = (futurePrice - priceAtMotif) / priceAtMotif
        returns.push(return_)
      }
    }
    
    if (returns.length === 0) return { avgReturn: 0, winRate: 0.5 }
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length
    const winRate = returns.filter(r => r > 0).length / returns.length
    
    return { avgReturn, winRate }
  }
  
  /**
   * Calculate trend from series of values
   */
  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0
    
    // Simple linear regression
    const n = values.length
    const sumX = (n * (n - 1)) / 2
    const sumY = values.reduce((a, b) => a + b, 0)
    const sumXY = values.reduce((sum, y, x) => sum + x * y, 0)
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    return slope
  }
  
  protected async onReset(): Promise<void> {
    this.priceBuffer = []
    this.matrixProfileHistory = []
    this.currentMotifs = []
    this.recentDiscords = []
    this.distanceCache.clear()
  }
}