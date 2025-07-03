import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface ClaudeConfig {
  /** Claude API model to use */
  model?: string
  /** Maximum tokens for response */
  maxTokens?: number
  /** Temperature for response variation */
  temperature?: number
  /** Number of recent analyses to include for context */
  contextWindow?: number
  /** Minimum confidence to act on Claude's recommendation */
  confidenceThreshold?: number
  /** Retry attempts for API failures */
  maxRetries?: number
  /** Cache duration for similar market conditions (minutes) */
  cacheDuration?: number
}

interface ClaudeAnalysis {
  pattern: string // Identified pattern name
  confidence: number // 0-1
  action: 'buy' | 'sell' | 'hold'
  reasoning: string
  priceTarget?: number
  stopLoss?: number
  timeframe: string // Expected duration
  relatedPatterns: string[] // Other patterns detected
  marketContext: string // Overall market assessment
}

interface PatternMemory {
  pattern: string
  timestamp: number
  marketConditions: string
  outcome?: 'success' | 'failure' | 'pending'
  actualReturn?: number
  feedback?: string
}

interface PromptContext {
  recentCandles: any[]
  indicators: Record<string, number>
  volume: number[]
  volatility: number
  trend: string
  support: number
  resistance: number
  patternHistory: PatternMemory[]
}

/**
 * Claude Pattern Recognition Agent
 * 
 * Leverages Claude's advanced pattern recognition capabilities to analyze complex
 * market patterns that are difficult to code explicitly. Uses the Anthropic API
 * to process market data and generate trading insights.
 * 
 * Key Features:
 * - **Visual Pattern Recognition**: Converts price data to descriptions Claude can analyze
 * - **Natural Language Reasoning**: Gets explanations for trading decisions
 * - **Adaptive Learning**: Includes feedback mechanism to improve over time
 * - **Context Awareness**: Provides market context for better analysis
 * - **Fallback Logic**: Graceful degradation when API is unavailable
 * 
 * Trading Process:
 * 1. Format market data into comprehensive prompts. **Respect context length and cost**.
 * 2. Send to Claude for pattern analysis
 * 3. Parse and validate Claude's response
 * 4. Generate trading signals with confidence scores
 * 5. Track outcomes for feedback learning
 * 
 * @todo Implement Anthropic API client setup using ANTHROPIC_API_KEY
 * @todo Create comprehensive prompt templates for market analysis
 * @todo Implement response parsing and validation
 * @todo Add caching layer for similar market conditions
 * @todo Create feedback loop for pattern success tracking
 * @todo Implement fallback logic for API failures
 */
export class ClaudePatternAgent extends BaseAgent {
  protected readonly config: Required<ClaudeConfig>
  private patternMemory: PatternMemory[] = []
  private readonly responseCache = new Map<string, { analysis: ClaudeAnalysis, timestamp: number }>()
  private lastApiCall = 0
  
  constructor(metadata: any, logger?: any, config?: ClaudeConfig) {
    super(metadata, logger)
    
    this.config = {
      model: config?.model ?? 'claude-3-opus-20240229', // Latest Claude model
      maxTokens: config?.maxTokens ?? 1000,
      temperature: config?.temperature ?? 0.3, // Lower for consistency
      contextWindow: config?.contextWindow ?? 10, // Recent analyses
      confidenceThreshold: config?.confidenceThreshold ?? 0.7,
      maxRetries: config?.maxRetries ?? 3,
      cacheDuration: config?.cacheDuration ?? 15 // 15 minutes
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Claude Pattern Agent initialized', {
      model: this.config.model,
      hasApiKey: !!process.env.ANTHROPIC_API_KEY
    })
    
    // @todo Initialize Anthropic client
    // this.apiClient = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    // Check if API is available (demo mode if no key)
    const isDemo = !process.env.ANTHROPIC_API_KEY
    
    // Generate cache key
    const cacheKey = this.generateCacheKey(context)
    
    // Check cache first
    const cached = this.getCachedAnalysis(cacheKey)
    if (cached) {
      const signal = this.createSignal(
        cached.action,
        cached.confidence,
        `[Cached] ${cached.pattern}: ${cached.reasoning}`
      )
      return enforceNoShorting(signal, context)
    }
    
    // Prepare context (would be used for actual API call)
    this.preparePromptContext(context)
    
    // In demo mode, use pattern recognition simulation
    if (isDemo) {
      return this.performFallbackAnalysis(context)
    }
    
    // Rate limiting
    const timeSinceLastCall = Date.now() - this.lastApiCall
    if (timeSinceLastCall < 1000) { // 1 second minimum between calls
      return this.performFallbackAnalysis(context)
    }
    
    try {
      // Generate comprehensive prompt
      const promptContext = this.preparePromptContext(context)
      const analysisPrompt = this.generateAnalysisPrompt(promptContext)
      
      // Add feedback if available
      const feedbackPrompt = this.generateFeedbackPrompt(this.patternMemory)
      const fullPrompt = feedbackPrompt 
        ? `${feedbackPrompt}\n\n${analysisPrompt}` 
        : analysisPrompt
      
      // Call API (or simulate)
      this.lastApiCall = Date.now()
      const response = await this.callClaudeAPI(fullPrompt)
      
      // Parse response
      const analysis = this.parseClaudeResponse(response)
      
      // Validate analysis
      if (analysis && this.validateAnalysis(analysis, context)) {
        // Cache the result
        this.responseCache.set(cacheKey, {
          analysis,
          timestamp: Date.now()
        })
        
        // Record in pattern memory
        this.patternMemory.push({
          pattern: analysis.pattern,
          timestamp: Date.now(),
          marketConditions: this.summarizeMarketConditions(context),
          outcome: 'pending'
        })
        
        // Keep memory limited
        if (this.patternMemory.length > 100) {
          this.patternMemory = this.patternMemory.slice(-100)
        }
        
        const signal = this.createSignal(
          analysis.action,
          analysis.confidence * this.config.confidenceThreshold,
          `${analysis.pattern}: ${analysis.reasoning}`
        )
        return enforceNoShorting(signal, context)
      }
      
      // Invalid analysis, use fallback
      return this.performFallbackAnalysis(context)
      
    } catch (error) {
      this.logger?.warn('Claude API error, using fallback', { error })
      return this.performFallbackAnalysis(context)
    }
  }
  
  /**
   * Prepare market data for Claude analysis
   */
  private preparePromptContext(context: MarketContext): PromptContext {
    const { candles } = context
    if (candles.length < 20) {
      return {
        recentCandles: [],
        indicators: {},
        volume: [],
        volatility: 0,
        trend: 'neutral',
        support: 0,
        resistance: 0,
        patternHistory: this.patternMemory.slice(-5)
      }
    }
    
    // Get recent candles
    const recentCandles = candles.slice(-20)
    const prices = recentCandles.map(c => c.close)
    const volumes = recentCandles.map(c => c.volume)
    
    // Calculate basic indicators
    const sma20 = prices.reduce((a, b) => a + b, 0) / prices.length
    const priceStdDev = Math.sqrt(prices.reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / prices.length)
    const volatility = priceStdDev / sma20
    
    // Determine trend
    const firstPrice = prices[0]!
    const lastPrice = prices[prices.length - 1]!
    const priceChange = (lastPrice - firstPrice) / firstPrice
    const trend = priceChange > 0.02 ? 'bullish' : priceChange < -0.02 ? 'bearish' : 'neutral'
    
    // Find support/resistance
    const highs = recentCandles.map(c => c.high)
    const lows = recentCandles.map(c => c.low)
    const resistance = Math.max(...highs)
    const support = Math.min(...lows)
    
    return {
      recentCandles: recentCandles.slice(-10),
      indicators: {
        sma20,
        volatility,
        priceChange,
        volumeAvg: volumes.reduce((a, b) => a + b, 0) / volumes.length
      },
      volume: volumes,
      volatility,
      trend,
      support,
      resistance,
      patternHistory: this.patternMemory.slice(-5)
    }
  }
  
  /**
   * Generate analysis prompt for Claude
   */
  private generateAnalysisPrompt(context: PromptContext): string {
    const { recentCandles, indicators, volume, volatility, trend, support, resistance, patternHistory } = context
    
    let prompt = `You are an expert technical analyst. Analyze the following market data and identify trading patterns.

`
    
    // Market overview
    prompt += `MARKET OVERVIEW:\n`
    prompt += `- Current Trend: ${trend}\n`
    prompt += `- Volatility: ${(volatility * 100).toFixed(2)}%\n`
    prompt += `- Support Level: $${support.toFixed(2)}\n`
    prompt += `- Resistance Level: $${resistance.toFixed(2)}\n`
    prompt += `- Average Volume: ${indicators.volumeAvg?.toFixed(0) || 'N/A'}\n\n`
    
    // Recent price action
    prompt += `RECENT PRICE ACTION (last ${recentCandles.length} candles):\n`
    recentCandles.forEach((candle, i) => {
      prompt += `${i + 1}. O: ${candle.open.toFixed(2)}, H: ${candle.high.toFixed(2)}, L: ${candle.low.toFixed(2)}, C: ${candle.close.toFixed(2)}, V: ${candle.volume}\n`
    })
    prompt += `\n`
    
    // Key indicators
    prompt += `KEY INDICATORS:\n`
    prompt += `- SMA20: $${indicators.sma20?.toFixed(2) || 'N/A'}\n`
    prompt += `- Price Change: ${((indicators.priceChange || 0) * 100).toFixed(2)}%\n`
    prompt += `- Volume Trend: ${volume.slice(-3).map(v => v > indicators.volumeAvg! ? '↑' : '↓').join('')}\n\n`
    
    // Pattern history
    if (patternHistory.length > 0) {
      prompt += `RECENT PATTERN HISTORY:\n`
      patternHistory.forEach((mem, i) => {
        prompt += `${i + 1}. ${mem.pattern} - ${mem.outcome || 'pending'}`
        if (mem.actualReturn) {
          prompt += ` (${(mem.actualReturn * 100).toFixed(2)}% return)`
        }
        prompt += `\n`
      })
      prompt += `\n`
    }
    
    // Analysis request
    prompt += `ANALYSIS REQUEST:\n`
    prompt += `1. Identify the primary chart pattern (if any)\n`
    prompt += `2. Assess pattern reliability (0-1 confidence)\n`
    prompt += `3. Recommend action: buy, sell, or hold\n`
    prompt += `4. Provide price targets if applicable\n`
    prompt += `5. Suggest stop loss levels\n`
    prompt += `6. Estimate timeframe for pattern completion\n`
    prompt += `7. List any secondary patterns observed\n`
    prompt += `8. Provide overall market context assessment\n\n`
    
    prompt += `Format your response as a structured analysis with clear sections.`
    
    return prompt
  }
  
  /**
   * Call Claude API with retry logic
   */
  private async callClaudeAPI(prompt: string): Promise<string> {
    // Note: This would require actual Anthropic client setup
    // For now, return a simulated response format
    
    let attempts = 0
    const maxAttempts = this.config.maxRetries
    
    while (attempts < maxAttempts) {
      try {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 100))
        
        // In production, this would be:
        // const response = await this.apiClient.messages.create({
        //   model: this.config.model,
        //   max_tokens: this.config.maxTokens,
        //   temperature: this.config.temperature,
        //   messages: [{ role: 'user', content: prompt }]
        // })
        // return response.content[0].text
        
        // Simulated response format
        return `PATTERN ANALYSIS:

Primary Pattern: Ascending Triangle
Confidence: 0.75
Action: buy
Reasoning: Price has been making higher lows while testing resistance at $50,500. Volume is increasing on upward moves.

Price Targets:
- Target 1: $51,200 (measured move)
- Target 2: $52,000 (extended target)

Stop Loss: $49,800 (below recent support)

Timeframe: 2-4 hours for breakout

Secondary Patterns:
- Bull flag on 15m timeframe
- Volume accumulation pattern

Market Context: Overall bullish momentum with healthy consolidation. Market breadth positive.`
        
      } catch (error) {
        attempts++
        if (attempts >= maxAttempts) {
          throw error
        }
        
        // Exponential backoff
        const backoffMs = Math.min(1000 * Math.pow(2, attempts), 30000)
        await new Promise(resolve => setTimeout(resolve, backoffMs))
      }
    }
    
    return ''
  }
  
  /**
   * Parse Claude's response into structured analysis
   */
  private parseClaudeResponse(response: string): ClaudeAnalysis | null {
    try {
      const lines = response.split('\n')
      const analysis: Partial<ClaudeAnalysis> = {
        relatedPatterns: [],
        marketContext: ''
      }
      
      let currentSection = ''
      
      for (const line of lines) {
        const trimmed = line.trim()
        
        // Pattern detection
        if (trimmed.toLowerCase().includes('primary pattern:')) {
          analysis.pattern = trimmed.split(':')[1]?.trim() || 'Unknown'
        }
        // Confidence extraction
        else if (trimmed.toLowerCase().includes('confidence:')) {
          const confStr = trimmed.split(':')[1]?.trim() || '0'
          analysis.confidence = parseFloat(confStr) || 0.5
        }
        // Action extraction
        else if (trimmed.toLowerCase().includes('action:')) {
          const action = trimmed.split(':')[1]?.trim().toLowerCase() || 'hold'
          if (['buy', 'sell', 'hold'].includes(action)) {
            analysis.action = action as 'buy' | 'sell' | 'hold'
          }
        }
        // Reasoning extraction
        else if (trimmed.toLowerCase().includes('reasoning:')) {
          analysis.reasoning = trimmed.split(':').slice(1).join(':').trim()
        }
        // Price targets
        else if (trimmed.toLowerCase().includes('target 1:')) {
          const targetStr = (/\$?([0-9,]+\.?[0-9]*)/.exec(trimmed))?.[1]
          if (targetStr) {
            analysis.priceTarget = parseFloat(targetStr.replace(',', ''))
          }
        }
        // Stop loss
        else if (trimmed.toLowerCase().includes('stop loss:')) {
          const stopStr = (/\$?([0-9,]+\.?[0-9]*)/.exec(trimmed))?.[1]
          if (stopStr) {
            analysis.stopLoss = parseFloat(stopStr.replace(',', ''))
          }
        }
        // Timeframe
        else if (trimmed.toLowerCase().includes('timeframe:')) {
          analysis.timeframe = trimmed.split(':').slice(1).join(':').trim()
        }
        // Secondary patterns
        else if (trimmed.toLowerCase().includes('secondary patterns:')) {
          currentSection = 'secondary'
        }
        else if (currentSection === 'secondary' && trimmed.startsWith('-')) {
          analysis.relatedPatterns!.push(trimmed.substring(1).trim())
        }
        // Market context
        else if (trimmed.toLowerCase().includes('market context:')) {
          analysis.marketContext = trimmed.split(':').slice(1).join(':').trim()
        }
      }
      
      // Validate required fields
      if (analysis.pattern && analysis.confidence !== undefined && analysis.action && analysis.reasoning) {
        return analysis as ClaudeAnalysis
      }
      
      return null
    } catch (error) {
      this.logger?.error('Failed to parse Claude response', { error })
      return null
    }
  }
  
  /**
   * Validate Claude's analysis for consistency
   */
  private validateAnalysis(analysis: ClaudeAnalysis, context: MarketContext): boolean {
    const { currentPrice, candles } = context
    
    // Basic validation
    if (!analysis?.pattern || !analysis.action) {
      return false
    }
    
    // Confidence should be reasonable
    if (analysis.confidence < 0 || analysis.confidence > 1) {
      return false
    }
    
    // Price targets should make sense
    if (analysis.priceTarget) {
      const targetDistance = Math.abs(analysis.priceTarget - currentPrice) / currentPrice
      // Target should be within reasonable range (not more than 20% away)
      if (targetDistance > 0.2) {
        return false
      }
      
      // Buy signal should have target above current price
      if (analysis.action === 'buy' && analysis.priceTarget < currentPrice) {
        return false
      }
      
      // Sell signal should have target below current price
      if (analysis.action === 'sell' && analysis.priceTarget > currentPrice) {
        return false
      }
    }
    
    // Stop loss validation
    if (analysis.stopLoss) {
      // Buy signal should have stop loss below current price
      if (analysis.action === 'buy' && analysis.stopLoss > currentPrice) {
        return false
      }
      
      // Sell signal should have stop loss above current price
      if (analysis.action === 'sell' && analysis.stopLoss < currentPrice) {
        return false
      }
    }
    
    // Pattern should match market conditions
    const recentPrices = candles.slice(-10).map(c => c.close)
    const priceChange = (recentPrices[recentPrices.length - 1]! - recentPrices[0]!) / recentPrices[0]!
    
    // Bullish patterns shouldn't appear in strong downtrends
    if (analysis.pattern.toLowerCase().includes('bull') && priceChange < -0.05) {
      return false
    }
    
    // Bearish patterns shouldn't appear in strong uptrends
    if (analysis.pattern.toLowerCase().includes('bear') && priceChange > 0.05) {
      return false
    }
    
    return true
  }
  
  /**
   * Generate cache key from market conditions
   */
  private generateCacheKey(context: MarketContext): string {
    const { currentPrice, candles } = context
    if (candles.length < 5) return 'insufficient_data'
    
    // Create key from recent price action (rounded to reduce granularity)
    const recentPrices = candles.slice(-5).map(c => Math.round(c.close / 10) * 10)
    const priceKey = recentPrices.join('_')
    const volumeKey = candles.slice(-5).map(c => Math.round(c.volume / 1000)).join('_')
    
    return `${priceKey}_${volumeKey}`
  }
  
  /**
   * Check if cached analysis is still valid
   */
  private getCachedAnalysis(key: string): ClaudeAnalysis | null {
    const cached = this.responseCache.get(key)
    if (!cached) return null
    
    // Check if cache is still fresh
    const age = Date.now() - cached.timestamp
    if (age > this.config.cacheDuration * 60 * 1000) {
      this.responseCache.delete(key)
      return null
    }
    
    return cached.analysis
  }
  
  /**
   * Update pattern memory with outcome feedback
   */
  public updatePatternFeedback(
    pattern: string, 
    outcome: 'success' | 'failure',
    actualReturn: number
  ): void {
    // Find the most recent occurrence of this pattern
    const patternIndex = this.patternMemory.findIndex(m => 
      m.pattern === pattern && m.outcome === 'pending'
    )
    
    if (patternIndex >= 0) {
      this.patternMemory[patternIndex]!.outcome = outcome
      this.patternMemory[patternIndex]!.actualReturn = actualReturn
      this.patternMemory[patternIndex]!.feedback = 
        outcome === 'success' 
          ? `Pattern completed successfully with ${(actualReturn * 100).toFixed(2)}% return`
          : `Pattern failed with ${(actualReturn * 100).toFixed(2)}% loss`
    }
    
    // Clean up old memory entries
    const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000)
    this.patternMemory = this.patternMemory.filter(m => m.timestamp > oneWeekAgo)
  }
  
  /**
   * Generate feedback prompt for pattern learning
   */
  private generateFeedbackPrompt(memory: PatternMemory[]): string {
    if (memory.length === 0) return ''
    
    let prompt = `Based on recent pattern outcomes, please adjust your analysis approach:\n\n`
    
    // Summarize successes
    const successes = memory.filter(m => m.outcome === 'success')
    if (successes.length > 0) {
      prompt += `SUCCESSFUL PATTERNS (${successes.length}):\n`
      successes.forEach(s => {
        prompt += `- ${s.pattern}: ${(s.actualReturn! * 100).toFixed(2)}% return\n`
      })
      prompt += `\n`
    }
    
    // Summarize failures
    const failures = memory.filter(m => m.outcome === 'failure')
    if (failures.length > 0) {
      prompt += `FAILED PATTERNS (${failures.length}):\n`
      failures.forEach(f => {
        prompt += `- ${f.pattern}: ${(f.actualReturn! * 100).toFixed(2)}% loss\n`
      })
      prompt += `\n`
    }
    
    // Calculate success rate by pattern type
    const patternStats = new Map<string, { success: number, total: number }>()
    memory.filter(m => m.outcome !== 'pending').forEach(m => {
      const stats = patternStats.get(m.pattern) || { success: 0, total: 0 }
      stats.total++
      if (m.outcome === 'success') stats.success++
      patternStats.set(m.pattern, stats)
    })
    
    prompt += `PATTERN SUCCESS RATES:\n`
    for (const [pattern, stats] of patternStats) {
      const rate = (stats.success / stats.total * 100).toFixed(1)
      prompt += `- ${pattern}: ${rate}% (${stats.success}/${stats.total})\n`
    }
    prompt += `\n`
    
    prompt += `Please consider these outcomes in your analysis and adjust confidence levels accordingly.`
    
    return prompt
  }
  
  /**
   * Fallback analysis when API is unavailable
   */
  private performFallbackAnalysis(context: MarketContext): AgentSignal {
    const { currentPrice, candles } = context
    
    if (candles.length < 20) {
      return this.createSignal('hold', 0.3, 'Insufficient data for pattern analysis')
    }
    
    // Simple pattern recognition
    const patterns: Array<{ name: string, detected: boolean, action: 'buy' | 'sell' | 'hold', confidence: number }> = []
    
    // 1. Head and Shoulders pattern
    const headAndShoulders = this.detectHeadAndShoulders(candles)
    if (headAndShoulders.detected) {
      patterns.push({
        name: 'Head and Shoulders',
        detected: true,
        action: 'sell',
        confidence: headAndShoulders.confidence
      })
    }
    
    // 2. Double Bottom pattern
    const doubleBottom = this.detectDoubleBottom(candles)
    if (doubleBottom.detected) {
      patterns.push({
        name: 'Double Bottom',
        detected: true,
        action: 'buy',
        confidence: doubleBottom.confidence
      })
    }
    
    // 3. Trend channel breakout
    const breakout = this.detectBreakout(candles, currentPrice)
    if (breakout.detected) {
      patterns.push({
        name: `${breakout.direction} Breakout`,
        detected: true,
        action: breakout.direction === 'Bullish' ? 'buy' : 'sell',
        confidence: breakout.confidence
      })
    }
    
    // Select strongest pattern
    if (patterns.length > 0) {
      const bestPattern = patterns.reduce((best, p) => p.confidence > best.confidence ? p : best)
      
      // Store in memory for learning
      this.patternMemory.push({
        pattern: bestPattern.name,
        timestamp: Date.now(),
        marketConditions: `Price: ${currentPrice}, Patterns: ${patterns.length}`,
        outcome: 'pending'
      })
      
      // Keep memory limited
      if (this.patternMemory.length > 50) {
        this.patternMemory.shift()
      }
      
      const signal = this.createSignal(
        bestPattern.action,
        bestPattern.confidence,
        `[Fallback] ${bestPattern.name} pattern detected`
      )
      return enforceNoShorting(signal, context)
    }
    
    // No clear pattern
    return this.createSignal('hold', 0.4, '[Fallback] No clear pattern detected')
  }
  
  /**
   * Detect head and shoulders pattern
   */
  private detectHeadAndShoulders(candles: readonly any[]): { detected: boolean, confidence: number } {
    if (candles.length < 15) return { detected: false, confidence: 0 }
    
    const highs = candles.slice(-15).map(c => c.high)
    
    // Look for 3 peaks pattern
    const peaks: number[] = []
    for (let i = 1; i < highs.length - 1; i++) {
      if (highs[i]! > highs[i-1]! && highs[i]! > highs[i+1]!) {
        peaks.push(i)
      }
    }
    
    if (peaks.length >= 3) {
      const [left, head, right] = peaks.slice(-3)
      const leftHigh = highs[left!]!
      const headHigh = highs[head!]!
      const rightHigh = highs[right!]!
      
      // Head should be highest, shoulders roughly equal
      if (headHigh > leftHigh && headHigh > rightHigh) {
        const shoulderDiff = Math.abs(leftHigh - rightHigh) / leftHigh
        if (shoulderDiff < 0.02) { // 2% tolerance
          return { detected: true, confidence: 0.7 - shoulderDiff * 10 }
        }
      }
    }
    
    return { detected: false, confidence: 0 }
  }
  
  /**
   * Detect double bottom pattern
   */
  private detectDoubleBottom(candles: readonly any[]): { detected: boolean, confidence: number } {
    if (candles.length < 15) return { detected: false, confidence: 0 }
    
    const lows = candles.slice(-15).map(c => c.low)
    
    // Look for 2 troughs pattern
    const troughs: number[] = []
    for (let i = 1; i < lows.length - 1; i++) {
      if (lows[i]! < lows[i-1]! && lows[i]! < lows[i+1]!) {
        troughs.push(i)
      }
    }
    
    if (troughs.length >= 2) {
      const [first, second] = troughs.slice(-2)
      const firstLow = lows[first!]!
      const secondLow = lows[second!]!
      
      // Bottoms should be roughly equal
      const bottomDiff = Math.abs(firstLow - secondLow) / firstLow
      if (bottomDiff < 0.015) { // 1.5% tolerance
        // Check for rise between bottoms
        const betweenHigh = Math.max(...lows.slice(first! + 1, second))
        if (betweenHigh > firstLow * 1.02) { // At least 2% rise
          return { detected: true, confidence: 0.65 + (0.015 - bottomDiff) * 20 }
        }
      }
    }
    
    return { detected: false, confidence: 0 }
  }
  
  /**
   * Detect breakout from channel
   */
  private detectBreakout(candles: readonly any[], currentPrice: number): { 
    detected: boolean, 
    direction: 'Bullish' | 'Bearish', 
    confidence: number 
  } {
    if (candles.length < 20) return { detected: false, direction: 'Bullish', confidence: 0 }
    
    const recent = candles.slice(-20)
    const highs = recent.map(c => c.high)
    const lows = recent.map(c => c.low)
    
    // Calculate channel boundaries
    const resistanceLevel = highs.slice(0, -1).reduce((max, h) => Math.max(max, h), 0)
    const supportLevel = lows.slice(0, -1).reduce((min, l) => Math.min(min, l), Infinity)
    
    // Check for breakout
    const breakoutMargin = 0.003 // 0.3% beyond level
    
    if (currentPrice > resistanceLevel * (1 + breakoutMargin)) {
      // Bullish breakout
      const strength = (currentPrice - resistanceLevel) / resistanceLevel
      return { 
        detected: true, 
        direction: 'Bullish', 
        confidence: Math.min(0.8, 0.5 + strength * 10)
      }
    } else if (currentPrice < supportLevel * (1 - breakoutMargin)) {
      // Bearish breakout
      const strength = (supportLevel - currentPrice) / supportLevel
      return { 
        detected: true, 
        direction: 'Bearish', 
        confidence: Math.min(0.8, 0.5 + strength * 10)
      }
    }
    
    return { detected: false, direction: 'Bullish', confidence: 0 }
  }
  
  /**
   * Format candles data for natural language description
   */
  private describePriceAction(candles: readonly any[]): string {
    if (candles.length < 2) return 'Insufficient price data'
    
    const recent = candles.slice(-10)
    const descriptions: string[] = []
    
    // Overall trend
    const firstClose = recent[0]?.close || 0
    const lastClose = recent[recent.length - 1]?.close || 0
    const overallChange = ((lastClose - firstClose) / firstClose) * 100
    
    if (Math.abs(overallChange) < 0.5) {
      descriptions.push('Price is consolidating sideways')
    } else if (overallChange > 2) {
      descriptions.push(`Strong uptrend with ${overallChange.toFixed(1)}% gain`)
    } else if (overallChange < -2) {
      descriptions.push(`Strong downtrend with ${Math.abs(overallChange).toFixed(1)}% loss`)
    } else if (overallChange > 0) {
      descriptions.push(`Mild uptrend with ${overallChange.toFixed(1)}% gain`)
    } else {
      descriptions.push(`Mild downtrend with ${Math.abs(overallChange).toFixed(1)}% loss`)
    }
    
    // Volatility
    const ranges = recent.map(c => c.high - c.low)
    const avgRange = ranges.reduce((a, b) => a + b, 0) / ranges.length
    const avgPrice = recent.reduce((sum, c) => sum + c.close, 0) / recent.length
    const volatility = (avgRange / avgPrice) * 100
    
    if (volatility > 3) {
      descriptions.push(`High volatility (${volatility.toFixed(1)}% average range)`)
    } else if (volatility < 1) {
      descriptions.push(`Low volatility (${volatility.toFixed(1)}% average range)`)
    } else {
      descriptions.push(`Moderate volatility (${volatility.toFixed(1)}% average range)`)
    }
    
    // Recent momentum
    const lastThree = recent.slice(-3)
    const recentTrend = lastThree.map((c, i) => 
      i > 0 ? (c.close > lastThree[i-1]!.close ? '↑' : '↓') : ''
    ).filter(Boolean).join('')
    
    if (recentTrend === '↑↑') {
      descriptions.push('Strong bullish momentum in recent candles')
    } else if (recentTrend === '↓↓') {
      descriptions.push('Strong bearish momentum in recent candles')
    } else {
      descriptions.push('Mixed momentum in recent candles')
    }
    
    // Volume trend
    const volumes = recent.map(c => c.volume)
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length
    const recentVolume = volumes.slice(-3).reduce((a, b) => a + b, 0) / 3
    
    if (recentVolume > avgVolume * 1.5) {
      descriptions.push('Volume increasing significantly')
    } else if (recentVolume < avgVolume * 0.7) {
      descriptions.push('Volume decreasing')
    }
    
    return descriptions.join('. ') + '.'
  }
  
  /**
   * Generate market condition summary
   */
  private summarizeMarketConditions(context: MarketContext): string {
    const { currentPrice, candles } = context
    if (candles.length < 20) return 'Limited market data available'
    
    const summaryParts: string[] = []
    
    // Price position relative to recent range
    const recentHighs = candles.slice(-20).map(c => c.high)
    const recentLows = candles.slice(-20).map(c => c.low)
    const rangeHigh = Math.max(...recentHighs)
    const rangeLow = Math.min(...recentLows)
    const pricePosition = (currentPrice - rangeLow) / (rangeHigh - rangeLow)
    
    if (pricePosition > 0.8) {
      summaryParts.push('Price near recent highs')
    } else if (pricePosition < 0.2) {
      summaryParts.push('Price near recent lows')
    } else {
      summaryParts.push('Price in middle of recent range')
    }
    
    // Trend strength
    const ma5 = candles.slice(-5).reduce((sum, c) => sum + c.close, 0) / 5
    const ma20 = candles.slice(-20).reduce((sum, c) => sum + c.close, 0) / 20
    
    if (ma5 > ma20 * 1.02) {
      summaryParts.push('short-term uptrend intact')
    } else if (ma5 < ma20 * 0.98) {
      summaryParts.push('short-term downtrend active')
    } else {
      summaryParts.push('trend neutral')
    }
    
    // Support/Resistance proximity
    const support = Math.min(...recentLows.slice(-10))
    const resistance = Math.max(...recentHighs.slice(-10))
    
    if (Math.abs(currentPrice - resistance) / currentPrice < 0.01) {
      summaryParts.push('testing resistance')
    } else if (Math.abs(currentPrice - support) / currentPrice < 0.01) {
      summaryParts.push('testing support')
    }
    
    // Market breadth (using volume as proxy)
    const recentVolumes = candles.slice(-5).map(c => c.volume)
    const avgRecentVolume = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length
    const historicalVolume = candles.slice(-20, -5).reduce((sum, c) => sum + c.volume, 0) / 15
    
    if (avgRecentVolume > historicalVolume * 1.3) {
      summaryParts.push('with increasing participation')
    } else if (avgRecentVolume < historicalVolume * 0.7) {
      summaryParts.push('on declining volume')
    }
    
    return summaryParts.join(', ') + '.'
  }
  
  protected async onReset(): Promise<void> {
    this.patternMemory = []
    this.responseCache.clear()
    this.lastApiCall = 0
  }
}