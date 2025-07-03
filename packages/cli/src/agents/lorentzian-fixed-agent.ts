import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface LorentzianConfig {
  lookbackPeriod?: number
  similarityThreshold?: number
  minMatches?: number
  momentumPeriod?: number
}

interface PricePoint {
  price: number
  momentum: number
  timestamp: number
}

/**
 * Fixed Lorentzian Distance Agent
 * 
 * A simplified and working version of the Lorentzian distance-based pattern matching.
 * Uses Lorentzian distance to find similar historical price patterns and predict future movement.
 */
export class LorentzianFixedAgent extends BaseAgent {
  protected readonly config: Required<LorentzianConfig>
  private priceHistory: PricePoint[] = []
  
  constructor(metadata: any, logger?: any, config?: LorentzianConfig) {
    super(metadata, logger)
    
    this.config = {
      lookbackPeriod: config?.lookbackPeriod ?? 100,
      similarityThreshold: config?.similarityThreshold ?? 0.5, // More lenient threshold
      minMatches: config?.minMatches ?? 2, // Require fewer matches
      momentumPeriod: config?.momentumPeriod ?? 5
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Lorentzian Fixed Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < this.config.momentumPeriod + 1) {
      return this.createSignal('hold', 0.3, 'Insufficient data')
    }
    
    // Calculate current momentum
    const currentMomentum = this.calculateMomentum(candles)
    
    // Create current price point
    const currentPoint: PricePoint = {
      price: currentPrice,
      momentum: currentMomentum,
      timestamp: Date.now()
    }
    
    // Add to history
    this.priceHistory.push(currentPoint)
    if (this.priceHistory.length > this.config.lookbackPeriod) {
      this.priceHistory.shift()
    }
    
    // Need minimum history
    if (this.priceHistory.length < 20) {
      return this.createSignal('hold', 0.4, `Building history: ${this.priceHistory.length}/20`)
    }
    
    // Find similar historical patterns
    const matches = this.findSimilarPatterns(currentPoint)
    
    // Debug: log some distances occasionally
    if (Math.random() < 0.05 && this.priceHistory.length > 30) {
      const sampleDistances = this.priceHistory.slice(-30, -5).map(p => ({
        price: p.price,
        distance: this.lorentzianDistance(currentPoint, p)
      }))
      const minDist = Math.min(...sampleDistances.map(d => d.distance))
      this.logger?.debug('Lorentzian distances', {
        currentPrice: currentPrice.toFixed(2),
        minDistance: minDist.toFixed(4),
        threshold: this.config.similarityThreshold,
        matches: matches.length
      })
    }
    
    if (matches.length < this.config.minMatches) {
      return this.createSignal('hold', 0.5, `Found ${matches.length} patterns, need ${this.config.minMatches}`)
    }
    
    // Analyze what happened after similar patterns
    const prediction = this.analyzePrediction(matches)
    
    if (Math.abs(prediction.expectedReturn) > 0.002) { // 0.2% threshold
      const action = prediction.expectedReturn > 0 ? 'buy' : 'sell'
      const signal = this.createSignal(
        action,
        prediction.confidence,
        `Lorentzian match: ${matches.length} patterns, expected return: ${(prediction.expectedReturn * 100).toFixed(2)}%`
      )
      
      return action === 'sell' ? enforceNoShorting(signal, context) : signal
    }
    
    return this.createSignal(
      'hold',
      0.5,
      `Lorentzian analysis: ${matches.length} patterns found, no clear direction`
    )
  }
  
  private calculateMomentum(candles: readonly any[]): number {
    const period = this.config.momentumPeriod
    if (candles.length < period + 1) return 0
    
    const recentCandles = candles.slice(-period - 1)
    const oldPrice = recentCandles[0].close
    const newPrice = recentCandles[recentCandles.length - 1].close
    
    if (oldPrice === 0) return 0
    
    return (newPrice - oldPrice) / oldPrice / period
  }
  
  private lorentzianDistance(a: PricePoint, b: PricePoint): number {
    // Normalize price difference by average price
    const avgPrice = (a.price + b.price) / 2
    if (avgPrice === 0) return Infinity
    
    const priceDiff = (a.price - b.price) / avgPrice
    
    // Scale momentum difference to be comparable to price diff
    const momentumDiff = (a.momentum - b.momentum) * 10 // Scale up momentum impact
    
    // Lorentzian distance: log(1 + x²)
    const squaredDiff = priceDiff * priceDiff + momentumDiff * momentumDiff
    const distance = Math.log(1 + squaredDiff)
    
    return isFinite(distance) ? distance : Infinity
  }
  
  private findSimilarPatterns(current: PricePoint): { point: PricePoint, distance: number, index: number }[] {
    const matches: { point: PricePoint, distance: number, index: number }[] = []
    
    // Don't match with very recent points
    const searchHistory = this.priceHistory.slice(0, -5)
    
    for (let i = 0; i < searchHistory.length; i++) {
      const historical = searchHistory[i]!
      const distance = this.lorentzianDistance(current, historical)
      
      if (distance < this.config.similarityThreshold) {
        matches.push({ point: historical, distance, index: i })
      }
    }
    
    // Sort by distance (closest first)
    matches.sort((a, b) => a.distance - b.distance)
    
    return matches
  }
  
  private analyzePrediction(matches: { point: PricePoint, distance: number, index: number }[]): {
    expectedReturn: number,
    confidence: number
  } {
    let totalReturn = 0
    let validOutcomes = 0
    
    for (const match of matches) {
      // Look at what happened 5 periods after this pattern
      const futureIndex = match.index + 5
      
      if (futureIndex < this.priceHistory.length - 5) {
        const futurePoint = this.priceHistory[futureIndex]!
        const return_ = (futurePoint.price - match.point.price) / match.point.price
        
        // Weight by inverse distance (closer patterns = more weight)
        const weight = 1 / (1 + match.distance)
        totalReturn += return_ * weight
        validOutcomes += weight
      }
    }
    
    if (validOutcomes === 0) {
      return { expectedReturn: 0, confidence: 0 }
    }
    
    const expectedReturn = totalReturn / validOutcomes
    const confidence = Math.min(0.85, 0.5 + validOutcomes * 0.1)
    
    return { expectedReturn, confidence }
  }
  
  protected async onReset(): Promise<void> {
    this.priceHistory = []
  }
}