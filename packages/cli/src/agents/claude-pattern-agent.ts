import { Anthropic } from '@anthropic-ai/sdk'
import { BaseAgent } from '@trdr/core'
import type { AgentMetadata, AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import type { SignalAction } from '@trdr/core/src/agents/types'
import { toEpochDate } from '@trdr/shared'
import type { Candle } from '@trdr/shared/src/types/market-data'
import * as fs from 'fs'

const MODEL_M = 'claude-sonnet-4-20250514'
// const MODEL_S = 'claude-3-5-haiku-20241022' // Remove unused model
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY
const TEMP = 0.05

// Type for Claude LLM JSON response
interface ClaudeSignalResponse {
  p: string; // pattern
  c: number; // confidence
  a: 'buy' | 'sell' | 'hold'; // action
  r: string; // reasoning
  t?: number; // price target
  s?: number; // stop loss
}

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
  /** Debug flag */
  debug?: boolean
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
  recentCandles: Candle[]
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
  // @ts-ignore - unused variable (reserved for future use)
  private apiClient?: Anthropic
  private readonly debug: boolean
  private lastClaudeCallIndex = -100
  
  // Config/env override helpers
  // @ts-ignore - unused variable (reserved for future use)
  private getClaudeModel(): string {
    return process.env.CLAUDE_MODEL || this.config.model
  }

  // Debug logging utility
  // @ts-ignore - unused variable (reserved for future use)
  private logDebug(...args: unknown[]): void {
    if (this.debug) {
      // eslint-disable-next-line no-console
      console.log('[ClaudeAgent]', ...args)
    }
  }

  // Use class fields for patternHistoryExample and verbose instruction
  // Use class fields for patternHistoryExample and instruction
  private readonly patternHistoryExample = '[{"pattern":"bull flag","timestamp":1710000000000,"marketConditions":"RSI oversold, MACD crossover","outcome":"success"},{"pattern":"double top","timestamp":1710000001000,"marketConditions":"Bearish divergence","outcome":"failure"}]'
  private readonly instruction = [
    'For "p", you must always output a real, plausible technical analysis pattern name (e.g., "bull flag", "double top", "head and shoulders", "rectangle", "ascending triangle", etc.). Do not output generic or placeholder values. If you cannot find a perfect match, pick the closest pattern from the list above.',
    'You must always output a real, plausible pattern name for "p". If you do not, your answer will be discarded.',
    '{"a":"buy|sell|hold","c":0-1,"r":"reason","p":"pattern","t":number,"s":number}',
    'For "t" (target) and "s" (stop), always output the actual price value, not percent, basis points, or normalized values. Use the same units as the closes in the context.',
    'Example: {"a":"buy","c":0.82,"r":"RSI oversold, MACD crossover, strong support","p":"bull flag","t":43250,"s":42800}',
    'Example: {"a":"sell","c":0.75,"r":"Bearish divergence, resistance rejection","p":"double top","t":42000,"s":43500}',
    'Example: {"a":"hold","c":0.55,"r":"No clear trend, low volatility","p":"rectangle","t":null,"s":null}',
    'Recent pattern history (last 5):',
    this.patternHistoryExample,
    'Use the above pattern history to inform your pattern choice for "p". If a similar pattern occurred recently, consider referencing it.',
    'Use real technical pattern names for "p" (e.g., "bull flag", "double top", "head and shoulders", etc.).',
    'Never output "unknown", "none", or similar for "p". If you are uncertain, make your best guess based on the context. If in doubt, pick the closest plausible pattern.',
    'If you output "hold", you must still provide a plausible pattern for "p".',
    'Be concise, do not hedge, do not say you are an AI.'
  ].join('\n')

  constructor(metadata: AgentMetadata, config?: ClaudeConfig) {
    super(metadata)
    
    this.config = {
      model: config?.model ?? MODEL_M,
      maxTokens: config?.maxTokens ?? 1000,
      temperature: config?.temperature ?? TEMP,
      contextWindow: config?.contextWindow ?? 10,
      confidenceThreshold: config?.confidenceThreshold ?? 0.7,
      maxRetries: config?.maxRetries ?? 2,
      cacheDuration: config?.cacheDuration ?? 10,
      debug: config?.debug ?? false
    }
    this.debug = Boolean(this.config.debug || process.env.CLAUDE_DEBUG)
  }
  
  // eslint-disable-next-line @typescript-eslint/require-await
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Claude Pattern Agent initialized', {
      model: this.config.model,
      hasApiKey: !!ANTHROPIC_API_KEY
    })
    if (ANTHROPIC_API_KEY) {
      this.apiClient = new Anthropic({ apiKey: ANTHROPIC_API_KEY })
    }
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    // Check if API key is available, use fallback if not
    if (!ANTHROPIC_API_KEY) {
      return this.fallbackAnalysis(context)
    }
    
    if (!this.shouldCallClaude({ candles: context.candles })) {
      return this.createSignal('hold', 0.0, 'market quiet')
    }
    
    // Check cache first
    const cacheKey = this.generateCacheKey(context)
    const cachedAnalysis = this.getCachedAnalysis(cacheKey)
    if (cachedAnalysis) {
      return this.createSignal(
        cachedAnalysis.action,
        cachedAnalysis.confidence,
        `[Cached] ${cachedAnalysis.reasoning}`,
        undefined,
        cachedAnalysis.priceTarget,
        cachedAnalysis.stopLoss
      )
    }
    
    // Use preparePromptContext for richer context
    const contextWindow = this.config.contextWindow || 10
    const promptContext = this.preparePromptContext(context, contextWindow)
    const prompt = this.generateAnalysisPrompt(promptContext)
    let text = ''
    try {
      text = await this.callClaudeAPI(prompt, context.currentPrice)
    } catch {
      return this.fallbackAnalysis(context)
    }
    // Parse and validate the Claude response
    let analysis: ClaudeAnalysis | null = null
    try {
      const parsed: unknown = JSON.parse(text)
      if (this.isClaudeResponse(parsed)) {
        analysis = {
          pattern: typeof parsed.p === 'string' ? parsed.p : '',
          confidence: Math.max(0, Math.min(1, typeof parsed.c === 'number' ? parsed.c : 0)),
          action: ['buy', 'sell', 'hold'].includes(parsed.a) ? parsed.a as SignalAction : 'hold',
          reasoning: typeof parsed.r === 'string' ? parsed.r : '',
          priceTarget: typeof parsed.t === 'number' && parsed.t > 0 ? parsed.t : undefined,
          stopLoss: typeof parsed.s === 'number' && parsed.s > 0 ? parsed.s : undefined,
          timeframe: '',
          relatedPatterns: [],
          marketContext: ''
        }
      }
    } catch { /* intentionally ignore parse errors */ }
    if (!analysis || !this.validateAnalysis(analysis, context)) {
      return this.createSignal('hold', analysis?.confidence ?? 0.3, 'Claude analysis failed validation')
    }

    // Add to pattern memory
    this.patternMemory.push({
      pattern: analysis.pattern || '',
      timestamp: Date.now(),
      marketConditions: analysis.reasoning || '',
      outcome: 'pending'
    })
    // Clean up old memory entries (keep only last week)
    const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000)
    this.patternMemory = this.patternMemory.filter(m => m.timestamp > oneWeekAgo)
    
    // Cache successful analysis
    this.responseCache.set(cacheKey, { analysis, timestamp: Date.now() })
    
    const action = analysis.action
    const confidence = analysis.confidence
    const reason = analysis.reasoning
    const priceTarget = analysis.priceTarget
    const stopLoss = analysis.stopLoss
    if (confidence < 0.5) {
      return this.createSignal('hold', confidence, 'Claude confidence too low')
    }
    if (typeof priceTarget === 'number' && typeof stopLoss === 'number') {
      return this.createSignal(action, confidence, reason, undefined, priceTarget, stopLoss)
    } else if (typeof priceTarget === 'number') {
      return this.createSignal(action, confidence, reason, undefined, priceTarget)
    } else {
      return this.createSignal(action, confidence, reason)
    }
  }
  
  /**
   * Fallback analysis when Claude API is unavailable
   */
  private fallbackAnalysis(context: MarketContext): AgentSignal {
    // Check cache first even in fallback mode
    const cacheKey = this.generateCacheKey(context)
    const cachedAnalysis = this.getCachedAnalysis(cacheKey)
    if (cachedAnalysis) {
      return this.createSignal(
        cachedAnalysis.action,
        cachedAnalysis.confidence,
        `[Cached] ${cachedAnalysis.reasoning}`,
        undefined,
        cachedAnalysis.priceTarget,
        cachedAnalysis.stopLoss
      )
    }
    
    // Create fallback analysis
    const analysis: ClaudeAnalysis = {
      pattern: 'fallback',
      confidence: 0.3,
      action: 'hold',
      reasoning: '[Fallback] Claude API unavailable - holding position',
      timeframe: '',
      relatedPatterns: [],
      marketContext: ''
    }
    
    // Cache the fallback analysis
    this.responseCache.set(cacheKey, { analysis, timestamp: Date.now() })
    
    return this.createSignal('hold', 0.3, '[Fallback] Claude API unavailable - holding position')
  }
  
  /**
   * Prepare market data for Claude analysis
   */
  private preparePromptContext(context: MarketContext, contextWindow = 29): PromptContext {
    const { candles, indicators } = context
    const recentCandles = candles.slice(-contextWindow)
    const prices = recentCandles.map(c => c.close)
    const volume = recentCandles.map(c => c.volume)
    // Calculate basic indicators
    const sma20 = prices.length ? prices.reduce((a, b) => a + b, 0) / prices.length : 0
    const priceStdDev = prices.length ? Math.sqrt(prices.reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / prices.length) : 0
    const volatility = sma20 ? priceStdDev / sma20 : 0
    // Determine trend
    const firstPrice = prices[0] ?? 0
    const lastPrice = prices[prices.length - 1] ?? 0
    const priceChange = firstPrice ? (lastPrice - firstPrice) / firstPrice : 0
    const trend = priceChange > 0.02 ? 'bullish' : priceChange < -0.02 ? 'bearish' : 'neutral'
    // Find support/resistance
    const highs = recentCandles.map(c => c.high)
    const lows = recentCandles.map(c => c.low)
    const resistance = highs.length ? Math.max(...highs) : 0
    const support = lows.length ? Math.min(...lows) : 0
    // Add more indicators if available
    const rsi = indicators?.RSI?.value ?? 0
    const macd = indicators?.MACD?.value ?? 0
    const ema20 = indicators?.EMA20?.value ?? 0
    const atr = indicators?.ATR?.value ?? 0
    const vwap = indicators?.VWAP?.value ?? 0
    const volumeProfile = indicators?.VolumeProfile?.value ?? 0
    return {
      recentCandles: recentCandles.slice(-contextWindow),
      indicators: {
        sma20,
        ema20,
        vwap,
        atr,
        volumeProfile,
        volatility,
        priceChange,
        rsi,
        macd
      },
      volume,
      volatility,
      trend,
      support,
      resistance,
      patternHistory: this.patternMemory.slice(-5)
    }
  }
  
  /**
   * Prompt optimization: Use compact tabular/CSV/JSON for candles, only essential indicators, minimal pattern history, and terse instructions.
   */
  private generateAnalysisPrompt(context: PromptContext): string {
    const contextWindow = this.config.contextWindow || 10
    const closes = context.recentCandles.slice(-contextWindow).map((c) => c.close)
    const ctx = {
      indicators: context.indicators,
      trend: context.trend,
      support: context.support,
      resistance: context.resistance,
      closes: closes.length > 0 ? closes : [43200, 43300, 43400, 43500, 43450, 43400, 43350, 43300, 43250, 43200],
      patternHistory: context.patternHistory.length > 0 ? context.patternHistory : [
        { pattern: 'bull flag', timestamp: Date.now() - 86400000, marketConditions: 'RSI oversold, MACD crossover', outcome: 'success' },
        { pattern: 'double top', timestamp: Date.now() - 43200000, marketConditions: 'Bearish divergence', outcome: 'failure' }
      ]
    }
    return `${this.instruction}\n${JSON.stringify(ctx)}`
  }
  
  // Helper to call Claude with a specific model
  private async callClaudeWithModel(_prompt: string, model: string): Promise<ClaudeSignalResponse | null> {
    const { Anthropic } = await import('@anthropic-ai/sdk')
    const apiKey: string | undefined = ANTHROPIC_API_KEY
    if (!apiKey) throw new Error('Missing ANTHROPIC_API_KEY in environment')
    const client = new Anthropic({ apiKey })
    let systemPrompt = [
      this.instruction,
      'You are a rich, astute, and successful technical trader. Think deeply, step-by-step.',
      'Analyze the following price action and provide a buy/sell/hold signal. Reply with a single-line JSON object only, matching the schema and example below.',
      _prompt
    ].join('\n')
    let responseText = ''
    let attempts = 0
    while (attempts < 2) {
      const response = await client.messages.create({
        model,
        max_tokens: 1000,
        temperature: TEMP,
        messages: [
          { role: 'user', content: systemPrompt }
        ]
      })
      const xs = response.content?.[0] as Partial<{ type: 'text', text: string }>
      if (xs.type === 'text' && xs.text) {
        responseText = xs.text.replace(/^```json\n/, '').replace(/\n```$/, '')
      } else {
        responseText = ''
      }
      try {
        const parsed: unknown = JSON.parse(responseText)
        if (this.isClaudeResponse(parsed)) return parsed
        return null
      } catch {
        systemPrompt = 'Your last response was invalid. Only reply with JSON as described. ' + _prompt
        attempts++
      }
    }
    return null
  }

  // Call both Haiku and Sonnet, compare, and return the best signal
  private async callClaudeAPI(_prompt: string, currentPrice: number): Promise<string> {
    const tasks: Promise<ClaudeSignalResponse | null>[] = [
      this.callClaudeWithModel(_prompt, 'claude-3-5-haiku-20241022'),
      this.callClaudeWithModel(_prompt, 'claude-sonnet-4-20250514')
    ]
    const [haiku, sonnet] = await Promise.all(tasks)

    if (haiku && this.isClaudeResponse(haiku)) this.normalizeTS(haiku, currentPrice)
    if (sonnet && this.isClaudeResponse(sonnet)) this.normalizeTS(sonnet, currentPrice)
    // console.log('[ClaudeAgent] HAiKU:', haiku)
    // console.log('[ClaudeAgent] SONNET:', sonnet)
    
    // If both are valid and agree on action and pattern, use as strong signal
    if (
      this.isClaudeResponse(haiku) &&
      this.isClaudeResponse(sonnet) &&
      this.normStr(haiku.a) === this.normStr(sonnet.a)
      //this.normStr(haiku.p) === this.normStr(sonnet.p) &&
      // this.nearlyEqual(haiku.t, sonnet.t) &&
      // this.nearlyEqual(haiku.s, sonnet.s)
    ) {
      // console.log('[ClaudeAgent] AGREEMENT (normalized):', haiku.a, haiku.p, 't:', haiku.t, sonnet.t, 's:', haiku.s, sonnet.s)
      return JSON.stringify(haiku)
    }
    // Otherwise, treat as weak/hold
    // console.log('[ClaudeAgent] NO AGREEMENT, fallback to hold or weak signal')
    return JSON.stringify({ a: 'hold', c: 0.3, r: 'No strong consensus between models', p: '', t: null, s: null })
  }
  
  /**
   * Parse Claude's response into structured analysis
   */
  // @ts-ignore - unused variable (reserved for future use)
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
    const { candles } = context
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
  // @ts-ignore - unused variable (reserved for future use)
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
   * Summarize market conditions
   */
  // @ts-ignore - unused variable (reserved for future use)
  private summarizeMarketConditions(_context: MarketContext): string {
    return ''
  }

  // Test utility: run on a small candle set
  public async testOnCandles(candles: Candle[], currentPrice: number): Promise<void> {
    // Use preparePromptContext for type safety
    const lastTimestamp = candles[candles.length - 1]?.timestamp ?? Date.now()
    const epochLastTimestamp = toEpochDate(lastTimestamp)
    const dummyContext = {
      candles,
      indicators: {
        sma: { value: 100, timestamp: epochLastTimestamp },
        rsi: { value: 50, timestamp: epochLastTimestamp }
      },
      trend: 'up',
      support: Math.min(...candles.map(c => c.low)),
      resistance: Math.max(...candles.map(c => c.high)),
      currentPrice: candles[candles.length - 1]?.close ?? 0,
      symbol: 'TEST',
      volume: candles.map(c => c.volume),
      timestamp: epochLastTimestamp,
      currentPosition: 0
    }
    const promptContext = this.preparePromptContext(dummyContext, 10)
    const prompt = this.generateAnalysisPrompt(promptContext)
    await this.callClaudeAPI(prompt, currentPrice)
  }

  // Log prompt/response pairs to a debug file if debug enabled
  // @ts-ignore - unused variable (reserved for future use)
  private logPromptResponse(prompt: string, response: string): void {
    if (!this.debug) return
    const logLine = `[${new Date().toISOString()}]\nPROMPT:\n${prompt}\nRESPONSE:\n${response}\n\n`
    fs.appendFileSync('claude-agent-debug.log', logLine)
  }

  // Validate output against the JSON schema
  // @ts-ignore - unused variable (reserved for future use)
  private validateClaudeSchema(obj: unknown): boolean {
    if (!obj || typeof obj !== 'object') return false
    const o = obj as Record<string, unknown>
    if (!['buy', 'sell', 'hold'].includes(o.a as string)) return false
    if (typeof o.c !== 'number' || o.c < 0 || o.c > 1) return false
    if (typeof o.r !== 'string') return false
    if ('t' in o && typeof o.t !== 'number') return false
    if ('s' in o && typeof o.s !== 'number') return false
    return true
  }

  // Gatekeeping: only call Claude if market is not flat
  private shouldCallClaude(context: PromptContext | { candles?: ReadonlyArray<{ close: number }> }): boolean {
    const contextWindow = this.config.contextWindow || 10
    let closes: number[] = []
    let candleCount = 0
    // Type guard for PromptContext
    if ('recentCandles' in context && Array.isArray(context.recentCandles)) {
      closes = context.recentCandles.slice(-contextWindow).map(c => c.close)
      candleCount = context.recentCandles.length
    } else if ('candles' in context && Array.isArray(context.candles)) {
      const arr = context.candles
      const isCandleArray = arr.every(
        (c): c is { close: number } => typeof c === 'object' && c !== null && 'close' in c && typeof (c as { close: unknown }).close === 'number'
      )
      if (isCandleArray) {
        closes = [...arr].slice(-contextWindow).map(c => c.close)
        candleCount = arr.length
      }
    }
    // Cooldown: only allow if at least 10 candles since last call
    if (candleCount - this.lastClaudeCallIndex < 10) return false
    if (closes.length < 2 || closes[0] === undefined || closes[closes.length - 1] === undefined) return false
    const firstClose = closes[0]
    const lastClose = closes[closes.length - 1]
    if (firstClose === undefined || lastClose === undefined) return false
    const mean = closes.reduce((a, b) => a + b, 0) / closes.length
    const variance = closes.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / closes.length
    const stddev = Math.sqrt(variance)
    const stddevThreshold = mean * 0.01 // 1%
    const priceChange = Math.abs(lastClose - firstClose) / firstClose
    const priceChangeThreshold = 0.01 // 1%
    const pass = stddev > stddevThreshold && priceChange > priceChangeThreshold
    if (pass) {
      this.lastClaudeCallIndex = candleCount
    }
    return pass
  }
  
  // eslint-disable-next-line @typescript-eslint/require-await
  protected async onReset(): Promise<void> {
    this.responseCache.clear()
    // this.patternMemory = []
    this.lastClaudeCallIndex = -100
  }

  // Type guard for Claude response
  private isClaudeResponse(obj: unknown): obj is ClaudeSignalResponse {
    return !!obj && typeof obj === 'object' &&
      typeof (obj as { a?: unknown }).a === 'string' &&
      typeof (obj as { p?: unknown }).p === 'string'
  }

  // Helper: compare numbers with tolerance
  // private nearlyEqual(a?: number, b?: number, relTol = 0.02, absTol = 0.5): boolean {
  //   if (typeof a !== 'number' || typeof b !== 'number') return false // treat missing as not matching
  //   if (a === b) return true
  //   const diff = Math.abs(a - b)
  //   return diff <= absTol || diff / Math.max(Math.abs(a), Math.abs(b)) <= relTol
  // }

  // Helper: normalize string for comparison
  private normStr(s?: string): string {
    return (s || '').trim().toLowerCase()
  }

  // Normalize t/s units if they are obviously off (e.g., 50x price)
  private normalizeTS(obj: { t?: number, s?: number } | undefined, currentPrice: number): void {
    // console.log(obj, currentPrice)
    while (obj?.t && obj.t > currentPrice * 5) obj.t /= 10
    while (obj?.s && obj.s > currentPrice * 5) obj.s /= 10
    // console.log(obj)
  }
}

// Minimal Anthropic API sanity check script
if (require.main === module) {
  void (async () => {
    // Import Anthropic SDK using import()
    const { Anthropic } = await import('@anthropic-ai/sdk')
    const apiKey: string | undefined = ANTHROPIC_API_KEY
    if (!apiKey) {
      console.error('Missing ANTHROPIC_API_KEY in environment')
      process.exit(1)
    }
    const client = new Anthropic({ apiKey })
    const prompt = 'Reply with {"a":"hold","c":0.3,"r":"test"}'
    try {
      const response: unknown = await client.messages.create({
        model: MODEL_M,
        max_tokens: 32,
        temperature: TEMP,
        messages: [{ role: 'user', content: prompt }]
      })
      let content = ''
      if (response && typeof response === 'object' && 'content' in response) {
        const respContent = (response as { content?: unknown }).content
        if (Array.isArray(respContent)) {
          content = respContent
            .filter((b): b is { text: string } => typeof b === 'object' && b !== null && 'text' in b && typeof (b as { text: unknown }).text === 'string')
            .map((b) => b.text)
            .join('')
        } else if (respContent && typeof respContent === 'object' && 'text' in respContent && typeof (respContent as { text: unknown }).text === 'string') {
          content = (respContent as { text: string }).text
        }
      }
      // console.log('Claude Sonnet response:', content)
      if (content && content.trim().length > 0) {
        process.exit(0)
      } else {
        process.exit(1)
      }
    } catch (err) {
      console.error('Claude API error:', err)
      process.exit(1)
    }
  })()
}