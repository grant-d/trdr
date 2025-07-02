import type {
  IPositionSizingStrategy,
  PositionSizingInput,
  PositionSizingOutput,
  BacktestTrade,
  BacktestResults,
  SizingAdjustment
} from '../interfaces'

/**
 * Fixed Fractional position sizing strategy.
 * 
 * Risks a fixed percentage of account balance on each trade.
 * Simple and widely used risk management approach.
 */
export class FixedFractionalStrategy implements IPositionSizingStrategy {
  readonly name = 'Fixed Fractional'
  readonly description = 'Risks a fixed percentage of account balance on each trade'
  
  private readonly riskPercentage: number
  
  /**
   * Creates a new Fixed Fractional strategy
   * @param riskPercentage - Percentage of account to risk per trade (default: 0.01 = 1%)
   */
  constructor(riskPercentage: number = 0.01) {
    this.riskPercentage = riskPercentage
  }
  
  calculateSize(input: PositionSizingInput): PositionSizingOutput {
    const adjustments: SizingAdjustment[] = []
    const warnings: string[] = []
    
    // Calculate risk amount
    let riskAmount = input.riskParams.accountBalance * this.riskPercentage
    let adjustedRiskPercentage = this.riskPercentage
    
    // Apply confidence adjustment
    riskAmount *= input.confidence
    adjustedRiskPercentage *= input.confidence
    
    // Apply market condition adjustments
    if (input.marketConditions.volatility > 0.7) {
      const volatilityFactor = 1 - ((input.marketConditions.volatility - 0.7) * 0.5)
      riskAmount *= volatilityFactor
      adjustedRiskPercentage *= volatilityFactor
      adjustments.push({
        type: 'volatility',
        factor: volatilityFactor,
        reason: 'High volatility risk reduction'
      })
    }
    
    // Apply drawdown adjustment if provided
    if (input.historicalMetrics) {
      const drawdownFactor = Math.max(0.5, 1 - input.historicalMetrics.maxDrawdown)
      if (drawdownFactor < 1) {
        riskAmount *= drawdownFactor
        adjustedRiskPercentage *= drawdownFactor
        adjustments.push({
          type: 'drawdown',
          factor: drawdownFactor,
          reason: `Drawdown adjustment: ${(input.historicalMetrics.maxDrawdown * 100).toFixed(1)}%`
        })
      }
    }
    
    // Calculate position size based on stop loss distance
    const stopDistance = Math.abs(input.entryPrice - input.stopLoss)
    const positionSize = riskAmount / stopDistance
    const positionValue = positionSize * input.entryPrice
    
    // Apply portfolio risk limits
    const currentRisk = (input.riskParams.currentExposure / input.riskParams.accountBalance) + adjustedRiskPercentage
    if (currentRisk > input.riskParams.maxPortfolioRisk) {
      const scaleFactor = (input.riskParams.maxPortfolioRisk - input.riskParams.currentExposure / input.riskParams.accountBalance) / adjustedRiskPercentage
      if (scaleFactor <= 0) {
        return {
          positionSize: 0,
          positionValue: 0,
          riskAmount: 0,
          riskPercentage: 0,
          method: this.name,
          confidence: 0,
          reasoning: 'Portfolio risk limit reached',
          warnings: ['Cannot add position - portfolio risk limit exceeded'],
          adjustments
        }
      }
      
      adjustments.push({
        type: 'risk_limit',
        factor: scaleFactor,
        reason: 'Portfolio risk limit adjustment'
      })
      
      return {
        positionSize: positionSize * scaleFactor,
        positionValue: positionValue * scaleFactor,
        riskAmount: riskAmount * scaleFactor,
        riskPercentage: adjustedRiskPercentage * scaleFactor,
        method: this.name,
        confidence: input.confidence * 0.8,
        reasoning: `Fixed ${(this.riskPercentage * 100).toFixed(1)}% risk, adjusted for conditions`,
        warnings: [...warnings, 'Position reduced to meet portfolio risk limits'],
        adjustments
      }
    }
    
    return {
      positionSize,
      positionValue,
      riskAmount,
      riskPercentage: adjustedRiskPercentage,
      method: this.name,
      confidence: input.confidence,
      reasoning: `Fixed ${(this.riskPercentage * 100).toFixed(1)}% risk per trade`,
      warnings,
      adjustments
    }
  }
  
  validate(input: PositionSizingInput): { valid: boolean; errors: string[] } {
    const errors: string[] = []
    
    if (input.entryPrice <= 0) {
      errors.push('Entry price must be positive')
    }
    
    if (input.stopLoss <= 0) {
      errors.push('Stop loss must be positive')
    }
    
    if (input.side === 'buy' && input.stopLoss >= input.entryPrice) {
      errors.push('Stop loss must be below entry price for buy orders')
    }
    
    if (input.side === 'sell' && input.stopLoss <= input.entryPrice) {
      errors.push('Stop loss must be above entry price for sell orders')
    }
    
    if (input.riskParams.accountBalance <= 0) {
      errors.push('Account balance must be positive')
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }
  
  getDefaultParams(): Record<string, any> {
    return {
      riskPercentage: this.riskPercentage
    }
  }
  
  backtest(trades: BacktestTrade[], initialBalance: number): BacktestResults {
    let balance = initialBalance
    let peakBalance = initialBalance
    let maxDrawdown = 0
    let wins = 0
    let losses = 0
    let totalProfit = 0
    let totalLoss = 0
    
    const equityCurve: Array<{ timestamp: Date; equity: number }> = []
    const drawdownCurve: Array<{ timestamp: Date; drawdown: number }> = []
    const returns: number[] = []
    
    // Add initial point
    equityCurve.push({ timestamp: trades[0]?.timestamp || new Date(), equity: balance })
    drawdownCurve.push({ timestamp: trades[0]?.timestamp || new Date(), drawdown: 0 })
    
    for (const trade of trades) {
      // Create mock input for position sizing
      const input: PositionSizingInput = {
        side: trade.side,
        entryPrice: trade.entryPrice,
        stopLoss: trade.stopLoss,
        winRate: trade.winRate,
        riskRewardRatio: trade.riskRewardRatio,
        confidence: trade.confidence,
        riskParams: {
          accountBalance: balance,
          maxRiskPerTrade: this.riskPercentage,
          maxPortfolioRisk: this.riskPercentage * 3,
          currentExposure: 0,
          openPositions: 0,
          maxPositions: 5,
          riskFreeRate: 0.02
        },
        marketConditions: trade.marketConditions,
        historicalMetrics: {
          avgWinRate: wins / Math.max(1, wins + losses),
          avgRiskReward: totalProfit / Math.max(1, wins) / Math.max(1, totalLoss / Math.max(1, losses)),
          maxConsecutiveLosses: 0,
          currentConsecutiveLosses: 0,
          sharpeRatio: 0,
          maxDrawdown,
          profitFactor: totalProfit / Math.max(1, totalLoss)
        }
      }
      
      // Calculate position size
      const sizing = this.calculateSize(input)
      
      if (sizing.positionSize > 0) {
        // Calculate P&L
        const priceDiff = trade.exitPrice - trade.entryPrice
        const pnl = trade.side === 'buy' ? priceDiff : -priceDiff
        const pnlAmount = pnl * sizing.positionSize
        
        // Update balance
        balance += pnlAmount
        
        // Track wins/losses
        if (pnlAmount > 0) {
          wins++
          totalProfit += pnlAmount
        } else {
          losses++
          totalLoss += Math.abs(pnlAmount)
        }
        
        // Calculate return
        const returnPct = pnlAmount / (balance - pnlAmount)
        returns.push(returnPct)
        
        // Update peak and drawdown
        if (balance > peakBalance) {
          peakBalance = balance
        }
        const currentDrawdown = (peakBalance - balance) / peakBalance
        maxDrawdown = Math.max(maxDrawdown, currentDrawdown)
        
        // Record equity and drawdown
        equityCurve.push({ timestamp: trade.timestamp, equity: balance })
        drawdownCurve.push({ timestamp: trade.timestamp, drawdown: currentDrawdown })
      }
    }
    
    // Calculate metrics
    const totalReturn = (balance - initialBalance) / initialBalance
    const lastTrade = trades[trades.length - 1]
    const firstTrade = trades[0]
    const years = trades.length > 1 && lastTrade && firstTrade
      ? (lastTrade.timestamp.getTime() - firstTrade.timestamp.getTime()) / (365 * 24 * 60 * 60 * 1000)
      : 1
    const annualizedReturn = years > 0 ? Math.pow(1 + totalReturn, 1 / years) - 1 : totalReturn
    
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0
    const stdDev = returns.length > 0 ? Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length) : 0
    const sharpeRatio = stdDev > 0 ? (avgReturn - 0.02 / 252) / stdDev * Math.sqrt(252) : 0
    
    const negativeReturns = returns.filter(r => r < 0)
    const downStdDev = negativeReturns.length > 0 ? Math.sqrt(negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length) : 0
    const sortinoRatio = downStdDev > 0 ? (avgReturn - 0.02 / 252) / downStdDev * Math.sqrt(252) : 0
    
    return {
      totalReturn,
      annualizedReturn,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      winRate: (wins + losses) > 0 ? wins / (wins + losses) : 0,
      avgWinLoss: losses > 0 && wins > 0 ? (totalProfit / wins) / (totalLoss / losses) : 0,
      totalTrades: wins + losses,
      profitFactor: totalLoss > 0 ? totalProfit / totalLoss : 0,
      equityCurve,
      drawdownCurve
    }
  }
}