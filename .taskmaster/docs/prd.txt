# Product Requirements Document: Grid Trading Bot with Multi-Agent Trailing Orders

## Executive Summary

Build sophisticated yet maintainable grid trading bot for retail traders on Coinbase. Uses multi-agent architecture with trailing orders. Optimized for swing trading (days not minutes). Key differentiators: self-tuning parameters, AI-enhanced decision making, realistic paper trading mode.

## 1. System Overview

### 1.1 Core Objectives
- **Primary Goal**: Generate consistent profits through automated grid trading with trailing orders
- **Target Users**: Retail traders with $1,000-$100,000 capital
- **Trading Style**: Swing trading (holding periods: hours to days)
- **Exchange**: Coinbase Advanced Trade API (spot trading only)
- **Performance Target**: 15-30% annual return after fees

### 1.2 Key Features
1. Multi-agent decision system with pluggable architecture
2. Trailing buy/sell orders instead of fixed grid levels
3. Self-tuning parameters (minimal configuration)
4. Three operational modes: Live, Paper, Backtest
5. AI integration for complex pattern recognition
6. Comprehensive monitoring and debugging tools
7. State persistence and crash recovery

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React/TypeScript)              │
├─────────────────────────────────────────────────────────────┤
│                      API Layer (Express/WebSocket)            │
├─────────────────────────────────────────────────────────────┤
│                      Core Trading Engine                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │Data Pipeline│  │Agent System  │  │Order Management  │   │
│  │            │  │              │  │                  │   │
│  │-Coinbase WS│  │-10+ Agents   │  │-Trailing Orders  │   │
│  │-REST Backup│  │-Orchestrator │  │-Grid Management  │   │
│  │-Unified API│  │-Competition  │  │-Risk Control     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer (DuckDB)                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack
- **Runtime**: Node.js 20+ with TypeScript 5+
- **Database**: DuckDB (embedded analytical database)
- **Frontend**: React 18+ with Vite
- **API**: Express + Socket.io for real-time updates
- **Testing**: Node.js built-in test runner (node:test) + Testing Library
- **Monitoring**: OpenTelemetry + custom metrics
- **Deployment**: Direct Node.js with PM2 process manager

### 2.3 Project Structure

```
trdr/
├── packages/
│   ├── core/                 # Core trading engine
│   │   ├── src/
│   │   │   ├── agents/       # Trading agents
│   │   │   ├── engine/       # Main trading loop
│   │   │   ├── orders/       # Order management
│   │   │   ├── grid/         # Grid calculations
│   │   │   └── utils/        # Shared utilities
│   │   └── tests/
│   ├── data/                 # Data layer
│   │   ├── src/
│   │   │   ├── db/          # DuckDB integration
│   │   │   ├── feeds/       # Market data feeds
│   │   │   └── models/      # Data models
│   │   └── tests/
│   ├── api/                  # REST/WebSocket API
│   ├── web/                  # React frontend
│   └── shared/               # Shared types/utilities
├── config/                   # Configuration files
├── scripts/                  # Build/deploy scripts
└── docs/                     # Documentation
```

## 3. Core Components

### 3.1 Data Pipeline

#### Protocol Strategy
- **Primary**: WebSocket for real-time data (low latency)
- **Fallback**: REST polling every 30s (reliability)
- **Backtesting**: DuckDB cursor with speed control
- **Paper Trading**: Same live feed, simulated execution

```typescript
interface MarketDataPipeline {
  // Unified interface for all modes
  subscribe(callback: (data: MarketUpdate) => void): void;
  unsubscribe(): void;
  getHistorical(from: Date, to: Date): Promise<Candle[]>;
  getCurrentPrice(): Promise<number>;
}

// Backtesting data feed
class BacktestDataFeed implements MarketDataPipeline {
  private data: Candle[] = [];
  private currentIndex = 0;
  private speed = 1000; // 1000x speed
  
  async loadHistoricalData(symbol: string, start: Date, end: Date) {
    // Load from DuckDB
    this.data = await this.db.query(`
      SELECT * FROM candles 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp
    `, [symbol, start, end]);
  }
  
  async start() {
    // Emit candles at simulated intervals
    const interval = 60000 / this.speed; // 1 min candles
    setInterval(() => {
      if (this.currentIndex < this.data.length) {
        this.emit('candle', this.data[this.currentIndex++]);
      }
    }, interval);
  }
}

class CoinbaseDataFeed implements MarketDataPipeline {
  private ws: WebSocket;
  private restClient: CoinbaseClient;
  private reconnectAttempts = 0;
  
  constructor(private config: DataFeedConfig) {
    this.initializeWebSocket();
    this.setupHeartbeat();
  }
  
  private initializeWebSocket() {
    // Primary: WebSocket for real-time data
    this.ws = new WebSocket('wss://ws-feed.pro.coinbase.com');
    
    // Auto-reconnect with exponential backoff
    this.ws.on('close', () => this.handleReconnect());
    
    // Subscribe to channels
    this.ws.on('open', () => {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['ticker', 'matches'],
        product_ids: [this.config.symbol]
      }));
    });
  }
  
  private setupHeartbeat() {
    // Fallback: REST polling every 30s for redundancy
    setInterval(async () => {
      if (!this.isWebSocketHealthy()) {
        const ticker = await this.restClient.getTicker(this.config.symbol);
        this.emit('ticker', ticker);
      }
    }, 30000);
  }
  
  private handleReconnect() {
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    
    this.logger.warn(`WebSocket disconnected. Reconnecting in ${delay}ms...`);
    
    setTimeout(() => {
      this.initializeWebSocket();
    }, delay);
  }
}

// Network resilience layer
class ResilientNetworkClient {
  private retryConfig = {
    maxRetries: 5,
    initialDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
    jitter: 0.1 // 10% jitter to avoid thundering herd
  };
  
  async executeWithRetry<T>(
    operation: () => Promise<T>,
    context: string
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 0; attempt <= this.retryConfig.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (!this.isRetryable(error) || attempt === this.retryConfig.maxRetries) {
          throw error;
        }
        
        const delay = this.calculateBackoff(attempt);
        this.logger.warn(`${context} failed, retrying in ${delay}ms`, {
          attempt: attempt + 1,
          error: error.message
        });
        
        await this.sleep(delay);
      }
    }
    
    throw lastError!;
  }
  
  private isRetryable(error: any): boolean {
    // Network errors
    if (error.code === 'ECONNRESET' || 
        error.code === 'ETIMEDOUT' ||
        error.code === 'ENOTFOUND') {
      return true;
    }
    
    // HTTP status codes
    const status = error.response?.status;
    if (status === 429 || // Rate limited
        status === 502 || // Bad gateway
        status === 503 || // Service unavailable
        status === 504) { // Gateway timeout
      return true;
    }
    
    // Coinbase specific
    if (error.message?.includes('request timestamp expired')) {
      return true;
    }
    
    return false;
  }
  
  private calculateBackoff(attempt: number): number {
    const exponentialDelay = this.retryConfig.initialDelay * 
      Math.pow(this.retryConfig.backoffMultiplier, attempt);
    
    const clampedDelay = Math.min(exponentialDelay, this.retryConfig.maxDelay);
    
    // Add jitter
    const jitter = clampedDelay * this.retryConfig.jitter * (Math.random() - 0.5);
    
    return Math.round(clampedDelay + jitter);
  }
  
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Order submission with retries
class ResilientOrderManager {
  private networkClient = new ResilientNetworkClient();
  private orderCache = new Map<string, Order>();
  
  async submitOrder(order: Order): Promise<OrderResult> {
    // Cache order in case of failure
    this.orderCache.set(order.id, order);
    
    try {
      const result = await this.networkClient.executeWithRetry(
        async () => {
          // Check if order already submitted (idempotency)
          const existing = await this.checkOrderExists(order.id);
          if (existing) {
            this.logger.info('Order already exists', { orderId: order.id });
            return existing;
          }
          
          // Submit to exchange
          return await this.exchange.createOrder({
            ...order,
            clientOrderId: order.id // Idempotency key
          });
        },
        `Submit order ${order.id}`
      );
      
      // Success - remove from cache
      this.orderCache.delete(order.id);
      return result;
      
    } catch (error) {
      // Final failure - store for manual recovery
      await this.storeFailedOrder(order, error);
      throw error;
    }
  }
  
  // Periodic retry of failed orders
  async retryFailedOrders(): Promise<void> {
    const failedOrders = await this.getFailedOrders();
    
    for (const order of failedOrders) {
      try {
        await this.submitOrder(order);
        await this.markOrderRecovered(order.id);
      } catch (error) {
        this.logger.error('Failed to recover order', {
          orderId: order.id,
          attempts: order.retryAttempts
        });
      }
    }
  }
  
  // Handle connection drops during order updates
  async modifyOrderWithRetry(
    orderId: string, 
    updates: OrderUpdate
  ): Promise<void> {
    await this.networkClient.executeWithRetry(
      async () => {
        // Get latest order state first
        const currentOrder = await this.exchange.getOrder(orderId);
        
        // Check if update still needed
        if (this.isUpdateStillValid(currentOrder, updates)) {
          await this.exchange.modifyOrder(orderId, updates);
        }
      },
      `Modify order ${orderId}`
    );
  }
}
}
```

### 3.2 Multi-Agent System

#### 3.2.1 Agent Interface

```typescript
interface ITradeAgent {
  readonly id: string;
  readonly name: string;
  readonly version: string;
  
  // Lifecycle
  initialize(context: AgentContext): Promise<void>;
  shutdown(): Promise<void>;
  
  // Core functionality
  analyze(market: MarketData): Promise<AgentSignal>;
  updatePerformance(result: TradeResult): void;
  
  // State management
  getState(): AgentState;
  setState(state: AgentState): void;
}

interface AgentSignal {
  agentId: string;
  action: 'TRAIL_BUY' | 'TRAIL_SELL' | 'HOLD';
  confidence: number; // 0-1
  trailDistance: number; // Percentage
  reasoning: Record<string, any>;
  urgency?: 'low' | 'medium' | 'high';
}
```

#### 3.2.2 Agent Categories

##### Core Agents (Required for basic operation)
1. **VolatilityAgent** - ATR-based trail distance, widens trails in calm periods
2. **MomentumAgent** - RSI/MACD divergence, multi-timeframe exhaustion detection  
3. **VolumeProfileAgent** - Volume spike analysis, whale detection
4. **MarketStructureAgent** - Support/resistance, structure break detection
5. **RegimeAgent** - Market condition classifier, adapts strategy to trending/ranging

##### Advanced Agents (Optional, enable for enhanced performance)
6. **AIPatternAgent** - Claude-powered pattern recognition with visual chart analysis
7. **TimeDecayAgent** - Tracks time at grid levels, tightens trails on "stale" prices
8. **MicrostructureAgent** - Spread analysis (simplified for retail traders)
9. **CorrelationAgent** - Cross-asset correlation for mean reversion signals
10. **SentimentAgent** - Contrarian signals at maximum pain points

##### Experimental Agents (Optional, innovative strategies)
11. **EntropyTransitionAgent** - Shannon entropy for phase transitions (chaos→order)
12. **SwarmIntelligenceAgent** - Models market participants as swarm with emergent behavior
13. **HarmonicResonanceAgent** - FFT frequency analysis, beat frequency detection
14. **TopologicalShapeAgent** - Persistent homology to find price "holes" and voids
15. **FractalMemoryAgent** - Self-similar pattern matching with historical memory
16. **QuantumSuperpositionAgent** - Models price as probability waves until "observed" by volume

#### 3.2.3 Agent Orchestrator

```typescript
class AgentOrchestrator {
  private agents: Map<string, ITradeAgent> = new Map();
  private weights: Map<string, number> = new Map();
  private performanceTracker: PerformanceTracker;
  
  async getConsensus(market: MarketData): Promise<TradingDecision> {
    // Parallel agent execution
    const signals = await Promise.all(
      Array.from(this.agents.values()).map(agent => 
        this.executeWithTimeout(agent, market)
      )
    );
    
    // Filter out failed/timeout agents
    const validSignals = signals.filter(s => s !== null);
    
    // Weight-adjusted voting
    const decision = this.calculateWeightedConsensus(validSignals);
    
    // Meta-decision for conflicts
    if (this.hasHighDisagreement(validSignals)) {
      return await this.aiMetaOrchestrator.resolve(validSignals, market);
    }
    
    return decision;
  }
  
  private async executeWithTimeout(
    agent: ITradeAgent, 
    market: MarketData
  ): Promise<AgentSignal | null> {
    try {
      return await Promise.race([
        agent.analyze(market),
        new Promise<null>((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 5000)
        )
      ]);
    } catch (error) {
      this.logger.error(`Agent ${agent.id} failed:`, error);
      return null;
    }
  }
}
```

### 3.3 Order Management

#### 3.3.1 Order Lifecycle Management

```typescript
class OrderLifecycleManager {
  private activeOrders: Map<string, ManagedOrder> = new Map();
  private orderStateMachine: OrderStateMachine;
  
  // Agent signals → Order decision
  async processAgentConsensus(consensus: AgentConsensus): Promise<Order | null> {
    // Minimum confidence threshold
    if (consensus.confidence < 0.6) return null;
    
    // Check if we should create new order or modify existing
    const existingOrder = this.findConflictingOrder(consensus);
    if (existingOrder) {
      return this.modifyOrder(existingOrder, consensus);
    }
    
    // Calculate order size
    const size = await this.calculateOrderSize(consensus);
    if (size < this.minOrderSize) return null;
    
    // Create order with metadata
    return {
      id: generateOrderId(),
      side: consensus.action,
      size,
      type: 'TRAILING',
      trailPercent: consensus.trailDistance,
      metadata: {
        consensus,
        agentVotes: consensus.agentSignals,
        createdBy: consensus.leadAgentId,
        timeConstraints: this.extractTimeConstraints(consensus)
      }
    };
  }
  
  // Dynamic position sizing
  async calculateOrderSize(consensus: AgentConsensus): Promise<number> {
    const available = await this.getAvailableCapital();
    const riskLimit = await this.riskManager.getPositionLimit();
    
    // Kelly Criterion with safety factor
    const kellySize = this.calculateKellySize(
      consensus.expectedWinRate,
      consensus.expectedRiskReward
    );
    
    // Apply constraints
    const baseSize = available * this.baseSizePercent; // 5% default
    const confidenceAdjusted = baseSize * consensus.confidence;
    const kellyAdjusted = Math.min(confidenceAdjusted, kellySize);
    const riskAdjusted = Math.min(kellyAdjusted, riskLimit);
    
    // Round to exchange requirements
    return this.roundToValidSize(riskAdjusted);
  }
  
  // Order monitoring and updates
  async monitorOrder(order: ManagedOrder): Promise<void> {
    const monitor = new OrderMonitor(order);
    
    monitor.on('priceUpdate', async (price) => {
      // Check if we should improve order
      if (this.shouldImproveOrder(order, price)) {
        await this.improveOrder(order, price);
      }
      
      // Check time-based conditions
      if (this.checkTimeConstraints(order)) {
        await this.handleTimeBasedAction(order);
      }
    });
    
    monitor.on('fill', async (fill) => {
      await this.handleOrderFill(order, fill);
    });
    
    monitor.on('partialFill', async (fill) => {
      await this.handlePartialFill(order, fill);
    });
  }
  
  // Order improvement logic
  private shouldImproveOrder(order: ManagedOrder, currentPrice: number): boolean {
    const improvement = order.side === 'BUY' 
      ? order.price - currentPrice 
      : currentPrice - order.price;
      
    // Improve if price moved favorably by >0.1%
    return improvement / order.price > 0.001;
  }
  
  private async improveOrder(order: ManagedOrder, newPrice: number): Promise<void> {
    try {
      await this.exchange.modifyOrder(order.id, { price: newPrice });
      order.price = newPrice;
      order.lastModified = Date.now();
    } catch (error) {
      this.logger.error('Failed to improve order', { orderId: order.id, error });
    }
  }
}

interface TimeConstraints {
  maxDuration?: number;        // Max time to keep order open
  closeBeforeEOD?: boolean;    // Close before market close
  blackoutPeriods?: Period[];  // Don't trade during these times
  expiresAt?: Date;           // Absolute expiration
}
```

#### 3.3.2 Order State Machine

```typescript
enum OrderState {
  PENDING = 'PENDING',
  SUBMITTED = 'SUBMITTED', 
  PARTIAL = 'PARTIAL',
  FILLED = 'FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED'
}

class OrderStateMachine {
  private transitions = {
    PENDING: ['SUBMITTED', 'CANCELLED'],
    SUBMITTED: ['PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'],
    PARTIAL: ['FILLED', 'CANCELLED'],
    FILLED: [],
    CANCELLED: [],
    REJECTED: [],
    EXPIRED: []
  };
  
  canTransition(from: OrderState, to: OrderState): boolean {
    return this.transitions[from]?.includes(to) ?? false;
  }
  
  async transition(order: ManagedOrder, newState: OrderState): Promise<void> {
    if (!this.canTransition(order.state, newState)) {
      throw new Error(`Invalid transition: ${order.state} → ${newState}`);
    }
    
    const oldState = order.state;
    order.state = newState;
    order.stateHistory.push({ state: newState, timestamp: Date.now() });
    
    await this.handleStateChange(order, oldState, newState);
  }
}
```

#### 3.3.3 Dynamic Order Adjustments

```typescript
class DynamicOrderAdjuster {
  // Handle funding changes
  async onFundingChange(event: FundingChangeEvent): Promise<void> {
    const openOrders = await this.getOpenOrders();
    
    for (const order of openOrders) {
      if (event.type === 'DEPOSIT') {
        // Can increase order size
        const newSize = await this.recalculateOrderSize(order);
        if (newSize > order.size * 1.1) { // >10% increase worth updating
          await this.increaseOrderSize(order, newSize);
        }
      } else if (event.type === 'WITHDRAWAL') {
        // May need to reduce exposure
        await this.validateOrderSize(order);
      }
    }
  }
  
  // Handle end-of-day procedures
  async handleEOD(): Promise<void> {
    const orders = await this.getOpenOrders();
    
    for (const order of orders) {
      if (order.metadata.timeConstraints?.closeBeforeEOD) {
        const timeUntilClose = this.getTimeUntilMarketClose();
        
        if (timeUntilClose < 300000) { // 5 minutes
          await this.closeOrder(order, 'EOD_CLOSE');
        }
      }
    }
  }
  
  // Handle partial fills
  async handlePartialFill(order: ManagedOrder, fill: Fill): Promise<void> {
    order.filledSize += fill.size;
    order.avgFillPrice = 
      (order.avgFillPrice * (order.filledSize - fill.size) + 
       fill.price * fill.size) / order.filledSize;
    
    // Decide whether to continue or cancel remainder
    const fillPercent = order.filledSize / order.size;
    
    if (fillPercent > 0.8) {
      // 80% filled, cancel remainder
      await this.cancelRemainder(order);
    } else if (this.hasBeenOpenTooLong(order)) {
      // Taking too long, reassess
      await this.reassessOrder(order);
    }
  }
}
```

#### 3.3.4 Trailing Order Implementation

```typescript
class TrailingOrderManager {
  private activeOrders: Map<string, TrailingOrder> = new Map();
  private orderMonitor: OrderMonitor;
  
  async createTrailingOrder(params: TrailingOrderParams): Promise<Order> {
    const order: TrailingOrder = {
      id: generateOrderId(),
      type: 'TRAILING',
      side: params.side,
      size: params.size,
      trailPercent: params.trailPercent,
      limitPrice: params.limitPrice,
      status: 'PENDING',
      createdAt: Date.now(),
      bestPrice: params.currentPrice,
      triggerPrice: this.calculateInitialTrigger(params)
    };
    
    this.activeOrders.set(order.id, order);
    this.orderMonitor.track(order);
    
    await this.persistOrder(order);
    return order;
  }
  
  private calculateInitialTrigger(params: TrailingOrderParams): number {
    const { currentPrice, side, trailPercent } = params;
    
    if (side === 'BUY') {
      // Trail buy: trigger when price rises X% from lowest point
      return currentPrice * (1 - trailPercent / 100);
    } else {
      // Trail sell: trigger when price falls X% from highest point
      return currentPrice * (1 + trailPercent / 100);
    }
  }
  
  async onPriceUpdate(price: number): Promise<void> {
    for (const order of this.activeOrders.values()) {
      if (order.status !== 'ACTIVE') continue;
      
      // Update best price and trigger
      if (order.side === 'SELL' && price > order.bestPrice) {
        order.bestPrice = price;
        order.triggerPrice = price * (1 - order.trailPercent / 100);
      } else if (order.side === 'BUY' && price < order.bestPrice) {
        order.bestPrice = price;
        order.triggerPrice = price * (1 + order.trailPercent / 100);
      }
      
      // Check trigger condition
      if (this.shouldTrigger(order, price)) {
        await this.executeOrder(order, price);
      }
    }
  }
}
```

#### 3.3.2 Grid Management

```typescript
class GridManager {
  private gridLevels: GridLevel[] = [];
  private activeGrids: Map<number, GridState> = new Map();
  
  async initializeGrid(config: GridConfig): Promise<void> {
    // Self-tuning grid spacing
    const optimalSpacing = await this.calculateOptimalSpacing(config);
    
    const currentPrice = await this.dataFeed.getCurrentPrice();
    const levels = this.generateGridLevels(currentPrice, optimalSpacing);
    
    this.gridLevels = levels;
    
    // Set initial trailing orders at nearby grids
    await this.activateNearbyGrids(currentPrice);
  }
  
  private async calculateOptimalSpacing(config: GridConfig): Promise<number> {
    // Analyze recent price swings
    const history = await this.dataFeed.getHistorical(
      new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days
      new Date()
    );
    
    const swings = this.identifySwings(history);
    const avgSwingSize = this.calculateAverageSwing(swings);
    
    // Grid spacing = fraction of average swing
    return avgSwingSize * 0.3; // 30% of average swing
  }
  
  private activateNearbyGrids(price: number): Promise<void> {
    const nearbyLevels = this.gridLevels.filter(level => 
      Math.abs(level.price - price) / price < 0.05 // Within 5%
    );
    
    return Promise.all(nearbyLevels.map(level => 
      this.activateGridLevel(level)
    ));
  }
}
```

### 3.4 Slippage Minimization Strategy

```typescript
class SlippageMinimizer {
  // Dynamic trail distance based on volatility
  calculateOptimalTrailDistance(
    baseTrail: number,
    volatility: number,
    liquidityScore: number
  ): number {
    // Wider trails in volatile/illiquid markets to avoid false triggers
    const volMultiplier = 1 + (volatility / 0.02); // If vol is 2%, double the trail
    const liquidityMultiplier = 2 - liquidityScore; // 0-1 score, less liquid = wider
    
    return baseTrail * volMultiplier * liquidityMultiplier;
  }
  
  // Smart order placement
  async placeOrderWithMinimalSlippage(order: Order): Promise<OrderResult> {
    const marketConditions = await this.analyzeMarketConditions();
    
    if (marketConditions.spread > 0.002) { // >0.2% spread
      // Use limit order near mid-price
      return this.placeLimitOrder({
        ...order,
        price: this.calculateSmartLimitPrice(marketConditions)
      });
    } else {
      // Tight spread, market order is OK
      return this.placeMarketOrder(order);
    }
  }
  
  // Avoid stop hunting zones
  adjustTrailForStopHunting(
    trailPrice: number,
    priceHistory: Candle[]
  ): number {
    const stopClusters = this.identifyStopClusters(priceHistory);
    
    // Move trail outside common stop zones
    for (const cluster of stopClusters) {
      if (Math.abs(trailPrice - cluster.price) / cluster.price < 0.003) {
        // Within 0.3% of stop cluster, adjust
        return cluster.price * (trailPrice > cluster.price ? 1.005 : 0.995);
      }
    }
    
    return trailPrice;
  }
  
  // Time-based execution
  selectOptimalExecutionTime(): ExecutionStrategy {
    const hour = new Date().getUTCHours();
    
    // Avoid known high-volatility periods
    if (hour === 14 || hour === 15) { // US market open
      return {
        useLimit: true,
        widerTrail: true,
        delayExecution: true
      };
    }
    
    return { useLimit: false, widerTrail: false, delayExecution: false };
  }
}

// Anti-slippage trail configuration
interface TrailConfiguration {
  // Base distances by market condition
  volatile: { min: 0.02, target: 0.03, max: 0.05 },  // 2-5%
  normal: { min: 0.01, target: 0.015, max: 0.03 },   // 1-3%
  calm: { min: 0.005, target: 0.01, max: 0.02 }      // 0.5-2%
}
```

### 3.5 Self-Tuning Parameter System

```typescript
// ONLY 3 user parameters - everything else self-tunes
interface MinimalUserConfig {
  symbol: string;           // What to trade
  capital: number;          // How much money
  riskTolerance: 'conservative' | 'moderate' | 'aggressive'; // Risk level
}

class SelfTuningSystem {
  private derivedParams: Map<string, number> = new Map();
  private learningRate = 0.1;
  
  // Convert minimal config to all internal parameters
  async initializeFromMinimalConfig(config: MinimalUserConfig): Promise<void> {
    // Base everything on risk tolerance
    const riskProfiles = {
      conservative: { drawdown: 0.10, confidence: 0.70, trailBase: 0.02 },
      moderate: { drawdown: 0.15, confidence: 0.60, trailBase: 0.015 },
      aggressive: { drawdown: 0.25, confidence: 0.50, trailBase: 0.01 }
    };
    
    const profile = riskProfiles[config.riskTolerance];
    
    // Self-tune everything else from market data
    const marketStats = await this.analyzeMarket(config.symbol);
    
    this.derivedParams.set('gridSpacing', marketStats.avgSwing * 0.3);
    this.derivedParams.set('positionSize', config.capital * 0.05); // Start with 5%
    this.derivedParams.set('maxConcurrent', Math.floor(profile.drawdown / 0.05));
    this.derivedParams.set('trailDistance', profile.trailBase * marketStats.volatilityRatio);
    
    // Start continuous optimization
    this.startOptimizationLoop();
  }
  
  // Continuous parameter optimization
  private async startOptimizationLoop(): Promise<void> {
    setInterval(async () => {
      const performance = await this.measureRecentPerformance();
      
      // Use gradient-free optimization (evolutionary strategy)
      const variations = this.generateParameterVariations();
      const results = await this.simulateVariations(variations);
      
      // Update parameters based on best performer
      const best = this.selectBest(results);
      this.blendParameters(best, this.learningRate);
      
      this.logger.info('Parameters optimized', {
        changes: this.getParameterChanges(),
        improvement: best.improvement
      });
    }, 24 * 60 * 60 * 1000); // Daily optimization
  }
  
  // Automatic parameter discovery
  private generateParameterVariations(): ParameterSet[] {
    const current = Object.fromEntries(this.derivedParams);
    const variations: ParameterSet[] = [current]; // Include current
    
    // Small random perturbations
    for (let i = 0; i < 10; i++) {
      const variation = { ...current };
      
      // Mutate each parameter slightly
      for (const [key, value] of Object.entries(variation)) {
        const change = (Math.random() - 0.5) * 0.2; // ±10%
        variation[key] = value * (1 + change);
      }
      
      variations.push(variation);
    }
    
    return variations;
  }
}

// Replace manual parameters with learned behaviors
class AdaptiveParameters {
  private history: CircularBuffer<MarketSnapshot>;
  private parameterCache: Map<string, AdaptiveValue> = new Map();
  
  // Get any parameter adaptively
  get(param: string): number {
    if (!this.parameterCache.has(param)) {
      this.parameterCache.set(param, new AdaptiveValue(param));
    }
    
    const adaptive = this.parameterCache.get(param)!;
    return adaptive.getCurrentValue(this.getCurrentContext());
  }
  
  // Learn from outcomes
  feedback(param: string, outcome: 'good' | 'bad'): void {
    const adaptive = this.parameterCache.get(param);
    if (adaptive) {
      adaptive.learn(outcome);
    }
  }
}

// Intelligent defaults that improve over time
class AdaptiveValue {
  private value: number;
  private confidence: number = 0.5;
  private outcomes: RingBuffer<boolean>;
  
  constructor(private paramName: string) {
    this.value = this.getIntelligentDefault(paramName);
    this.outcomes = new RingBuffer(100);
  }
  
  getCurrentValue(context: MarketContext): number {
    // Adjust based on context
    const baseValue = this.value;
    const contextMultiplier = this.getContextMultiplier(context);
    
    return baseValue * contextMultiplier;
  }
  
  learn(outcome: 'good' | 'bad'): void {
    this.outcomes.push(outcome === 'good');
    
    // Adjust value based on success rate
    const successRate = this.outcomes.average();
    
    if (successRate < 0.4) {
      // Poor performance, make bigger adjustment
      this.value *= 0.9;
    } else if (successRate > 0.6) {
      // Good performance, fine-tune
      this.value *= 1.05;
    }
    
    this.confidence = Math.min(0.9, this.confidence + 0.05);
  }
  
  private getIntelligentDefault(param: string): number {
    // Smart defaults based on parameter type
    const defaults: Record<string, number> = {
      gridSpacing: 0.015,      // 1.5%
      trailDistance: 0.02,     // 2%
      positionSize: 0.05,      // 5% of capital
      confidenceThreshold: 0.6, // 60%
      maxSlippage: 0.001,      // 0.1%
      timeoutMs: 5000          // 5 seconds
    };
    
    return defaults[param] ?? 1.0;
  }
}
```

### 3.6 Risk Management

```typescript
class RiskManager {
  private maxDrawdown = 0.15; // 15%
  private maxPositionSize = 0.25; // 25% of capital per position
  private maxConcurrentTrades = 5;
  
  async validateOrder(order: OrderRequest): Promise<RiskValidation> {
    const checks = await Promise.all([
      this.checkDrawdown(),
      this.checkPositionSize(order),
      this.checkConcurrentTrades(),
      this.checkDailyLoss()
    ]);
    
    const failed = checks.filter(c => !c.passed);
    
    return {
      approved: failed.length === 0,
      failedChecks: failed,
      adjustedOrder: this.adjustOrderForRisk(order, checks)
    };
  }
  
  private async checkDrawdown(): Promise<RiskCheck> {
    const equity = await this.portfolio.getCurrentEquity();
    const peak = await this.portfolio.getEquityPeak();
    const drawdown = (peak - equity) / peak;
    
    return {
      name: 'maxDrawdown',
      passed: drawdown < this.maxDrawdown,
      value: drawdown,
      limit: this.maxDrawdown,
      severity: drawdown > this.maxDrawdown * 0.8 ? 'high' : 'medium'
    };
  }
}
```

### 3.5 Core Algorithms

```typescript
// Grid calculation with self-tuning
class GridAlgorithm {
  calculateOptimalSpacing(priceHistory: Candle[]): number {
    const swings = this.findSwingHighsLows(priceHistory);
    const avgSwing = mean(swings.map(s => s.size));
    const volatility = std(priceHistory.map(c => c.close));
    
    // Dynamic spacing: 20-40% of average swing
    const baseSpacing = avgSwing * 0.3;
    const volAdjustment = volatility / avgSwing;
    
    return clamp(baseSpacing * volAdjustment, 0.005, 0.05); // 0.5%-5%
  }
  
  generateGridLevels(centerPrice: number, spacing: number, count: number): number[] {
    const levels: number[] = [];
    for (let i = -count/2; i <= count/2; i++) {
      levels.push(centerPrice * (1 + i * spacing));
    }
    return levels;
  }
}

// Trailing distance optimization
class TrailAlgorithm {
  calculateTrailDistance(agent: AgentSignal, market: MarketCondition): number {
    const baseTrail = 0.02; // 2% base
    
    // Adjust based on market conditions
    const multipliers = {
      trending: 1.5,   // Wider trails in trends
      ranging: 0.7,    // Tighter in ranges
      volatile: 2.0,   // Much wider when volatile
      calm: 0.5        // Very tight when calm
    };
    
    return baseTrail * multipliers[market] * agent.confidence;
  }
}
```

## 4. Data Schema

### 4.1 DuckDB Tables

```sql
-- Price data (optimized for analytics)
CREATE TABLE candles (
  timestamp TIMESTAMP PRIMARY KEY,
  symbol VARCHAR NOT NULL,
  open DECIMAL(20,8),
  high DECIMAL(20,8),
  low DECIMAL(20,8),
  close DECIMAL(20,8),
  volume DECIMAL(20,8),
  trades INTEGER,
  -- Derived columns for fast queries
  returns DECIMAL(10,6) GENERATED ALWAYS AS ((close - open) / open),
  range DECIMAL(10,6) GENERATED ALWAYS AS ((high - low) / low),
  INDEX idx_symbol_time (symbol, timestamp)
);

-- Orders table
CREATE TABLE orders (
  id VARCHAR PRIMARY KEY,
  type VARCHAR NOT NULL, -- 'MARKET', 'LIMIT', 'TRAILING'
  side VARCHAR NOT NULL, -- 'BUY', 'SELL'
  symbol VARCHAR NOT NULL,
  size DECIMAL(20,8),
  price DECIMAL(20,8),
  status VARCHAR NOT NULL,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  filled_at TIMESTAMP,
  fill_price DECIMAL(20,8),
  fees DECIMAL(20,8),
  metadata JSON
);

-- Agent decisions log
CREATE TABLE agent_decisions (
  id INTEGER PRIMARY KEY,
  timestamp TIMESTAMP,
  agent_id VARCHAR,
  market_data JSON,
  signal VARCHAR,
  confidence DECIMAL(5,4),
  reasoning JSON,
  execution_time_ms INTEGER
);

-- Performance tracking
CREATE TABLE trades (
  id INTEGER PRIMARY KEY,
  entry_order_id VARCHAR,
  exit_order_id VARCHAR,
  entry_price DECIMAL(20,8),
  exit_price DECIMAL(20,8),
  size DECIMAL(20,8),
  pnl DECIMAL(20,8),
  fees DECIMAL(20,8),
  duration_seconds INTEGER,
  metadata JSON
);

-- System state for recovery
CREATE TABLE checkpoints (
  id INTEGER PRIMARY KEY,
  timestamp TIMESTAMP,
  type VARCHAR, -- 'auto', 'manual', 'shutdown'
  state JSON,
  mode VARCHAR -- 'live', 'paper', 'backtest'
);
```

## 5. API Specification

### 5.1 REST Endpoints

```typescript
// Health & Status
GET /api/health
GET /api/status

// Trading Control
POST /api/trading/start
POST /api/trading/stop
POST /api/trading/pause
PUT /api/trading/mode // Switch between live/paper

// Configuration
GET /api/config
PUT /api/config
GET /api/agents
PUT /api/agents/:id/weight

// Analytics
GET /api/performance
GET /api/trades
GET /api/agents/performance
GET /api/execution-quality

// Backtesting
POST /api/backtest/run
GET /api/backtest/:id/status
GET /api/backtest/:id/results
```

### 5.2 WebSocket Events

```typescript
// Client -> Server
interface ClientMessages {
  subscribe: { channels: string[] };
  unsubscribe: { channels: string[] };
  command: { action: string; params: any };
}

// Server -> Client
interface ServerEvents {
  price: { symbol: string; price: number; timestamp: number };
  order: { type: 'created' | 'filled' | 'cancelled'; order: Order };
  trade: { trade: Trade; pnl: number };
  agent: { agentId: string; signal: AgentSignal };
  performance: { metrics: PerformanceMetrics };
  alert: { level: 'info' | 'warning' | 'error'; message: string };
}
```

## 6. Monitoring & Observability

### 6.1 Logging Strategy

```typescript
class StructuredLogger {
  private logger: winston.Logger;
  
  constructor() {
    this.logger = winston.createLogger({
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      transports: [
        new winston.transports.File({ 
          filename: 'logs/error.log', 
          level: 'error' 
        }),
        new winston.transports.File({ 
          filename: 'logs/trades.log', 
          level: 'info' 
        }),
        new winston.transports.Console({
          format: winston.format.simple()
        })
      ]
    });
  }
  
  // Debug logging for development
  debug(component: string, message: string, data?: any) {
    if (process.env.DEBUG) {
      this.logger.debug(`[${component}] ${message}`, data);
    }
  }
  
  // Performance logging
  logPerformance(metric: string, value: number, metadata?: any) {
    this.logger.info('performance_metric', {
      metric,
      value,
      timestamp: Date.now(),
      ...metadata
    });
  }
  
  logTrade(trade: Trade) {
    this.logger.info('trade_executed', {
      tradeId: trade.id,
      side: trade.side,
      price: trade.price,
      size: trade.size,
      pnl: trade.pnl,
      fees: trade.fees,
      agentSignals: trade.metadata.signals
    });
  }
}
```

### 6.2 Debug Tools

```typescript
class DebugInspector {
  // Real-time agent decision viewer
  async inspectAgentDecisions(): Promise<AgentDebugInfo[]> {
    return this.agents.map(agent => ({
      id: agent.id,
      lastSignal: agent.getLastSignal(),
      confidence: agent.getConfidence(),
      state: agent.getState(),
      performance: agent.getRecentPerformance()
    }));
  }
  
  // Order flow debugger
  traceOrder(orderId: string): OrderTrace {
    return {
      created: this.getOrderCreation(orderId),
      updates: this.getOrderUpdates(orderId),
      fills: this.getOrderFills(orderId),
      timeline: this.buildOrderTimeline(orderId)
    };
  }
  
  // Performance profiler
  profileAgent(agentId: string): AgentProfile {
    return {
      avgExecutionTime: this.getAvgExecutionTime(agentId),
      memoryUsage: this.getMemoryUsage(agentId),
      successRate: this.getSuccessRate(agentId)
    };
  }
}
```

### 6.3 Metrics Collection

```typescript
interface SystemMetrics {
  // Performance
  totalPnL: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  
  // Execution Quality
  avgSlippage: number;
  avgFillTime: number;
  failedOrders: number;
  
  // System Health
  uptime: number;
  memoryUsage: number;
  cpuUsage: number;
  wsReconnects: number;
  
  // Agent Performance
  agentMetrics: Map<string, AgentMetrics>;
}

class MetricsCollector {
  private metrics: SystemMetrics;
  private prometheus: PrometheusClient;
  
  constructor() {
    this.setupPrometheusMetrics();
    this.startCollection();
  }
  
  private setupPrometheusMetrics() {
    // Custom metrics for trading
    this.pnlGauge = new Gauge({
      name: 'trading_pnl_total',
      help: 'Total profit and loss'
    });
    
    this.tradeCounter = new Counter({
      name: 'trades_total',
      help: 'Total number of trades',
      labelNames: ['side', 'result']
    });
    
    this.agentConfidenceHistogram = new Histogram({
      name: 'agent_confidence',
      help: 'Agent confidence distribution',
      labelNames: ['agent_id'],
      buckets: [0.1, 0.3, 0.5, 0.7, 0.9]
    });
  }
}
```

## 7. Security Considerations

### 7.1 API Key Management

```typescript
class SecureCredentialManager {
  private vault: Map<string, EncryptedCredential> = new Map();
  
  async loadCredentials() {
    // Never store API keys in code or config files
    const encryptedKeys = await this.loadFromEnv();
    
    // Decrypt with master key
    const masterKey = await this.getMasterKey();
    
    for (const [name, encrypted] of encryptedKeys) {
      const decrypted = await this.decrypt(encrypted, masterKey);
      this.vault.set(name, decrypted);
    }
    
    // Clear master key from memory
    this.clearSensitiveData(masterKey);
  }
  
  getApiKey(exchange: string): string {
    const credential = this.vault.get(exchange);
    if (!credential) throw new Error('No credential found');
    
    // Rotate keys periodically
    if (this.shouldRotate(credential)) {
      this.scheduleRotation(exchange);
    }
    
    return credential.key;
  }
}
```

### 7.2 Rate Limiting

```typescript
class RateLimiter {
  private limits = {
    coinbase: {
      public: { calls: 10, window: 1000 }, // 10/second
      private: { calls: 15, window: 1000 }  // 15/second
    }
  };
  
  async executeWithLimit<T>(
    endpoint: string, 
    fn: () => Promise<T>
  ): Promise<T> {
    const limit = this.getLimit(endpoint);
    await this.waitIfNeeded(endpoint, limit);
    
    try {
      return await fn();
    } finally {
      this.recordCall(endpoint);
    }
  }
}
```

## 8. Performance Optimization

### 8.1 Data Processing

```typescript
class PerformantDataProcessor {
  // Use typed arrays for numerical computations
  calculateIndicators(candles: Candle[]): Indicators {
    const prices = new Float64Array(candles.map(c => c.close));
    
    // Batch calculate all indicators
    return {
      sma: this.calculateSMA(prices),
      ema: this.calculateEMA(prices),
      rsi: this.calculateRSI(prices),
      // ... other indicators
    };
  }
  
  // Memoize expensive calculations
  @memoize({ maxAge: 60000 }) // Cache for 1 minute
  async getMarketProfile(symbol: string): Promise<MarketProfile> {
    const data = await this.db.query(`
      SELECT 
        AVG(volume) as avg_volume,
        STDDEV(returns) as volatility,
        COUNT(*) as samples
      FROM candles
      WHERE symbol = ? AND timestamp > NOW() - INTERVAL '30 days'
    `, [symbol]);
    
    return this.processMarketProfile(data);
  }
}
```

### 8.2 Agent Optimization

```typescript
class OptimizedAgentRunner {
  // Run agents in parallel with resource limits
  async runAgents(market: MarketData): Promise<AgentSignal[]> {
    const cpuCount = os.cpus().length;
    const batchSize = Math.max(2, cpuCount - 1); // Leave 1 CPU free
    
    const batches = this.chunk(this.agents, batchSize);
    const results: AgentSignal[] = [];
    
    for (const batch of batches) {
      const batchResults = await Promise.all(
        batch.map(agent => this.runAgent(agent, market))
      );
      results.push(...batchResults);
    }
    
    return results;
  }
  
  // Use worker threads for CPU-intensive agents
  private async runAgent(agent: ITradeAgent, market: MarketData): Promise<AgentSignal> {
    if (agent.isCpuIntensive) {
      return this.runInWorker(agent, market);
    }
    return agent.analyze(market);
  }
}
```

## 9. Testing Strategy

### 9.1 Test Coverage Requirements

- Unit tests: >90% coverage for algorithms
- Integration tests: All API endpoints
- E2E tests: Critical user flows
- Performance tests: Response time < 100ms
- Load tests: Handle 100 concurrent connections

### 9.2 Test Implementation

```typescript
// Using Node.js built-in test runner
import { describe, it, beforeEach, mock } from 'node:test';
import assert from 'node:assert/strict';

describe('TrailingOrderManager', () => {
  let manager: TrailingOrderManager;
  let mockDataFeed: MockDataFeed;
  
  beforeEach(() => {
    mockDataFeed = new MockDataFeed();
    manager = new TrailingOrderManager(mockDataFeed);
  });
  
  describe('Trailing Sell Orders', () => {
    it('should follow price up and trigger on reversal', async (t) => {
      // Create trailing sell at $100 with 2% trail
      const order = await manager.createTrailingOrder({
        side: 'SELL',
        size: 1,
        trailPercent: 2,
        currentPrice: 100
      });
      
      // Price rises to $105
      await mockDataFeed.emitPrice(105);
      assert.equal(order.bestPrice, 105);
      assert.equal(order.triggerPrice, 102.9); // 105 * 0.98
      
      // Price drops to trigger
      await mockDataFeed.emitPrice(102.5);
      
      // Verify order executed
      const fills = await manager.getFills();
      assert.equal(fills.length, 1);
      assert.ok(Math.abs(fills[0].price - 102.5) < 0.01);
    });
  });
  
  describe('Network Resilience', () => {
    it('should retry failed orders with backoff', async (t) => {
      // Mock network failure
      const mockExchange = t.mock.method(exchange, 'createOrder');
      mockExchange.mock.mockImplementation(() => {
        throw new Error('ECONNRESET');
      });
      
      // Should retry
      const order = { side: 'BUY', size: 1, price: 100 };
      await assert.rejects(
        () => manager.submitOrder(order),
        { message: /ECONNRESET/ }
      );
      
      // Verify retry attempts
      assert.equal(mockExchange.mock.calls.length, 6); // 1 + 5 retries
    });
  });
});

// Performance benchmarks
import { test } from 'node:test';

test('Agent execution performance', async (t) => {
  const market = generateLargeMarketData(10000);
  const agent = new VolatilityAgent();
  
  const start = performance.now();
  await agent.analyze(market);
  const duration = performance.now() - start;
  
  assert.ok(duration < 100, `Analysis took ${duration}ms, expected <100ms`);
});
```

## 10. Deployment & Operations

### 10.1 Direct Node.js Deployment

```bash
# Install dependencies
npm ci --production

# Build TypeScript
npm run build

# Start with PM2 for process management
npm install -g pm2
pm2 start dist/index.js --name trdr-bot

# Auto-restart on crash
pm2 startup
pm2 save

# Monitor
pm2 monit
```

### 10.2 Production Checklist

- [ ] Environment variables configured
- [ ] API keys encrypted and stored securely
- [ ] Database backups configured
- [ ] Monitoring alerts set up
- [ ] Rate limits configured
- [ ] Error handling comprehensive
- [ ] Graceful shutdown implemented
- [ ] Recovery procedures tested
- [ ] Performance baselines established
- [ ] Security audit completed

## 11. Success Metrics

### 11.1 Financial Metrics
- **Primary**: Net profit after fees > 15% annually
- **Risk-adjusted**: Sharpe ratio > 1.5
- **Drawdown**: Maximum drawdown < 15%
- **Win rate**: > 45% with 2:1 reward/risk

### 11.2 Operational Metrics
- **Uptime**: > 99.9%
- **Order success rate**: > 99%
- **Average latency**: < 100ms
- **Recovery time**: < 5 minutes

### 11.3 Cost Efficiency
- **Fees as % of gross**: < 20%
- **Slippage**: < 0.1% average
- **Infrastructure cost**: < 5% of profits

## 12. AI & Machine Learning Integration

### 12.1 LLM Integration (Claude/GPT)

```typescript
class AIAnalysisEngine {
  private claudeClient: AnthropicClient;
  private responseCache: LRUCache<string, AIAnalysis>;
  private costTracker: CostTracker;
  
  // Multi-modal chart analysis
  async analyzeChart(marketData: MarketData): Promise<AIAnalysis> {
    // Generate chart image
    const chartImage = await this.generateCandlestickChart(marketData);
    
    // Create structured prompt
    const prompt = this.buildAnalysisPrompt(marketData);
    
    // Check cache first (for backtesting consistency)
    const cacheKey = this.generateCacheKey(marketData);
    if (this.responseCache.has(cacheKey)) {
      return this.responseCache.get(cacheKey)!;
    }
    
    // Call Claude with image + text
    const response = await this.claudeClient.messages.create({
      model: 'claude-3-opus-20240229',
      max_tokens: 500,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: chartImage } }
        ]
      }]
    });
    
    const analysis = this.parseStructuredResponse(response);
    this.responseCache.set(cacheKey, analysis);
    
    return analysis;
  }
  
  private buildAnalysisPrompt(market: MarketData): string {
    return `Analyze this ${market.timeframe} chart for ${market.symbol}.
    
    Recent price sequence: ${this.formatPriceSequence(market)}
    
    Identify:
    1. Chart pattern (if any)
    2. Key support/resistance levels  
    3. Momentum direction
    4. Optimal trailing entry/exit points
    
    Response format:
    PATTERN: [pattern name or NONE]
    SUPPORT: [price level]
    RESISTANCE: [price level]
    MOMENTUM: [BULLISH/BEARISH/NEUTRAL]
    ACTION: [TRAIL_BUY/TRAIL_SELL/WAIT]
    CONFIDENCE: [0-100]
    TRAIL_DISTANCE: [percentage]`;
  }
}

// Cost-aware AI usage
class AIResourceManager {
  private dailyBudget = 10.00; // $10/day
  private currentSpend = 0;
  
  async requestAnalysis(priority: Priority): Promise<boolean> {
    const estimatedCost = this.estimateCost(priority);
    
    if (priority === 'CRITICAL') return true; // Always allow critical
    
    if (this.currentSpend + estimatedCost > this.dailyBudget) {
      this.logger.warn('AI budget exceeded, using cached/approximate analysis');
      return false;
    }
    
    this.currentSpend += estimatedCost;
    return true;
  }
}
```

### 12.2 Evolutionary Algorithms

```typescript
class EvolutionaryOptimizer {
  private population: Strategy[] = [];
  private populationSize = 50;
  private mutationRate = 0.1;
  private crossoverRate = 0.7;
  
  // Evolve trading strategies
  async evolveStrategies(
    historicalData: Candle[],
    generations: number = 100
  ): Promise<Strategy> {
    // Initialize random population
    this.population = this.createInitialPopulation();
    
    for (let gen = 0; gen < generations; gen++) {
      // Evaluate fitness (backtest each strategy)
      const fitness = await this.evaluatePopulation(historicalData);
      
      // Select parents (tournament selection)
      const parents = this.tournamentSelection(fitness);
      
      // Create offspring
      const offspring = this.crossoverAndMutate(parents);
      
      // Replace worst performers
      this.population = this.selectSurvivors(
        [...this.population, ...offspring], 
        fitness
      );
      
      this.logger.info(`Generation ${gen}: Best fitness = ${Math.max(...fitness)}`);
    }
    
    return this.getBestStrategy();
  }
  
  // Genetic representation of strategy
  interface StrategyGenome {
    agentWeights: number[];        // Weight for each agent
    gridSpacing: number;           // 0.005 - 0.05
    trailDistanceBase: number;     // 0.005 - 0.03
    confidenceThreshold: number;   // 0.5 - 0.9
    riskParameters: {
      maxDrawdown: number;         // 0.05 - 0.25
      positionSize: number;        // 0.01 - 0.10
      maxConcurrent: number;       // 1 - 10
    };
  }
  
  private crossover(parent1: StrategyGenome, parent2: StrategyGenome): StrategyGenome {
    // Uniform crossover
    return {
      agentWeights: parent1.agentWeights.map((w, i) => 
        Math.random() < 0.5 ? w : parent2.agentWeights[i]
      ),
      gridSpacing: Math.random() < 0.5 ? parent1.gridSpacing : parent2.gridSpacing,
      // ... mix other parameters
    };
  }
  
  private mutate(genome: StrategyGenome): StrategyGenome {
    const mutated = { ...genome };
    
    // Gaussian mutation
    if (Math.random() < this.mutationRate) {
      const param = this.selectRandomParameter();
      mutated[param] *= (1 + this.gaussian() * 0.2); // ±20% change
    }
    
    return mutated;
  }
}
```

### 12.3 Reinforcement Learning Integration

```typescript
class RLTradingAgent {
  private qTable: Map<StateAction, number> = new Map();
  private epsilon = 0.1; // Exploration rate
  private alpha = 0.001; // Learning rate
  private gamma = 0.95; // Discount factor
  
  // State representation
  private encodeState(market: MarketData): State {
    return {
      priceChange: this.discretize(market.returns, 10),
      volatility: this.discretize(market.volatility, 5),
      position: this.currentPosition,
      gridLevel: this.nearestGridLevel,
      agentConsensus: this.discretizeConsensus()
    };
  }
  
  // Action selection (epsilon-greedy)
  selectAction(state: State): Action {
    if (Math.random() < this.epsilon) {
      // Explore: random action
      return this.randomAction();
    }
    
    // Exploit: best known action
    return this.getBestAction(state);
  }
  
  // Q-learning update
  updateQ(state: State, action: Action, reward: number, nextState: State) {
    const currentQ = this.qTable.get({ state, action }) || 0;
    const maxNextQ = this.getMaxQ(nextState);
    
    const newQ = currentQ + this.alpha * (reward + this.gamma * maxNextQ - currentQ);
    this.qTable.set({ state, action }, newQ);
  }
  
  // Reward function
  calculateReward(trade: CompletedTrade): number {
    const profitReward = trade.pnl / trade.size; // Normalized profit
    const riskPenalty = trade.maxDrawdown * -2; // Penalize drawdown
    const feesPenalty = trade.fees / trade.size * -10; // Penalize high fees
    
    return profitReward + riskPenalty + feesPenalty;
  }
}
```

### 12.4 Ensemble AI Decision Making

```typescript
class AIEnsembleOrchestrator {
  private models = {
    llm: new LLMAnalyzer(),
    rl: new RLTradingAgent(),
    evolution: new EvolutionaryOptimizer(),
    timeSeries: new LSTMPredictor()
  };
  
  async getEnsembleDecision(market: MarketData): Promise<EnsembleDecision> {
    // Get predictions from each model
    const predictions = await Promise.all([
      this.models.llm.predict(market),
      this.models.rl.predict(market),
      this.models.timeSeries.predict(market)
    ]);
    
    // Weight by recent performance
    const weights = this.getAdaptiveWeights();
    
    // Combine predictions
    const ensemble = this.weightedAverage(predictions, weights);
    
    // Use evolutionary algorithm for meta-parameters
    const optimalParams = await this.models.evolution.getCurrentBest();
    
    return {
      action: ensemble.action,
      confidence: ensemble.confidence,
      parameters: optimalParams,
      breakdown: predictions // For transparency
    };
  }
  
  // Adaptive weighting based on recent accuracy
  private getAdaptiveWeights(): Weights {
    const recentPerformance = this.measureRecentAccuracy(100); // Last 100 predictions
    
    // Normalize to sum to 1
    const total = Object.values(recentPerformance).reduce((a, b) => a + b);
    
    return {
      llm: recentPerformance.llm / total,
      rl: recentPerformance.rl / total,
      timeSeries: recentPerformance.timeSeries / total
    };
  }
}
```

### 12.5 AI Cost Optimization

```typescript
class AIOptimizer {
  // Batch multiple decisions for efficiency
  async batchAnalyze(requests: AnalysisRequest[]): Promise<AIAnalysis[]> {
    // Group similar requests
    const batches = this.groupSimilarRequests(requests);
    
    // Single LLM call per batch
    return Promise.all(batches.map(batch => 
      this.analyzeBatch(batch)
    ));
  }
  
  // Use smaller models for simple decisions
  selectModel(complexity: Complexity): ModelChoice {
    switch(complexity) {
      case 'simple':
        return 'claude-instant'; // Faster, cheaper
      case 'complex':
        return 'claude-3-opus'; // More capable
      case 'critical':
        return 'claude-3-opus'; // Best model for important decisions
    }
  }
}
```

## 13. Backtesting Pipeline & Parameter Management

### 13.1 Backtesting Workflow

```typescript
class BacktestingPipeline {
  private sessions: Map<string, BacktestSession> = new Map();
  
  // Create named session with parameters
  async createSession(config: BacktestConfig): Promise<BacktestSession> {
    const session: BacktestSession = {
      id: generateSessionId(),
      name: config.name || `Backtest_${new Date().toISOString()}`,
      description: config.description,
      parameters: config.parameters,
      dateRange: config.dateRange,
      createdAt: new Date(),
      status: 'pending'
    };
    
    // Store session
    await this.db.insert('backtest_sessions', session);
    this.sessions.set(session.id, session);
    
    return session;
  }
  
  // Run backtest with progress tracking
  async runBacktest(sessionId: string): Promise<BacktestResults> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error('Session not found');
    
    session.status = 'running';
    
    // Load historical data
    const data = await this.loadHistoricalData(session.dateRange);
    
    // Initialize engine with session parameters
    const engine = new BacktestEngine(session.parameters);
    
    // Run with progress callbacks
    const results = await engine.run(data, (progress) => {
      this.updateProgress(sessionId, progress);
    });
    
    // Store results
    await this.storeResults(sessionId, results);
    session.status = 'completed';
    
    return results;
  }
  
  // Compare multiple sessions
  async compareSessions(sessionIds: string[]): Promise<ComparisonReport> {
    const results = await Promise.all(
      sessionIds.map(id => this.getResults(id))
    );
    
    return {
      summary: this.generateComparisonSummary(results),
      charts: this.generateComparisonCharts(results),
      recommendations: this.analyzeResults(results)
    };
  }
}
```

### 13.2 Parameter Version Control

```typescript
class ParameterVersionControl {
  // Save parameter set with metadata
  async saveParameterSet(params: ParameterSet): Promise<string> {
    const version = {
      id: generateVersionId(),
      name: params.name,
      description: params.description,
      parameters: params.values,
      parentVersion: params.parentId,
      performance: params.backtestResults,
      createdAt: new Date(),
      tags: params.tags // ['production', 'experimental', 'optimized']
    };
    
    await this.db.insert('parameter_versions', version);
    
    // Git-like commit message
    await this.createCommit(version, params.commitMessage);
    
    return version.id;
  }
  
  // Recall specific parameter set
  async loadParameterSet(identifier: string): Promise<ParameterSet> {
    // Load by ID, name, or tag
    const version = await this.db.query(`
      SELECT * FROM parameter_versions 
      WHERE id = ? OR name = ? OR tags @> ?
      ORDER BY created_at DESC
      LIMIT 1
    `, [identifier, identifier, [identifier]]);
    
    return this.deserializeParameters(version);
  }
  
  // Branch parameters for experimentation
  async branchParameters(
    baseId: string, 
    branchName: string
  ): Promise<ParameterSet> {
    const base = await this.loadParameterSet(baseId);
    
    return {
      ...base,
      id: generateVersionId(),
      name: branchName,
      parentId: baseId,
      tags: [...base.tags, 'branch']
    };
  }
  
  // Merge successful experiments
  async mergeParameters(
    branchId: string,
    targetId: string,
    strategy: 'performance' | 'average' | 'best'
  ): Promise<ParameterSet> {
    const branch = await this.loadParameterSet(branchId);
    const target = await this.loadParameterSet(targetId);
    
    const merged = this.mergeStrategies[strategy](branch, target);
    
    return this.saveParameterSet({
      ...merged,
      commitMessage: `Merged ${branchId} into ${targetId} using ${strategy}`
    });
  }
}
```

### 13.3 Comprehensive Statistics

```typescript
interface BacktestResults {
  // Performance Metrics
  performance: {
    totalReturn: number;         // 45.2%
    annualizedReturn: number;    // 15.1%
    sharpeRatio: number;         // 1.85
    sortinoRatio: number;        // 2.31
    calmarRatio: number;         // 3.2
    maxDrawdown: number;         // -12.3%
    maxDrawdownDuration: number; // 23 days
    winRate: number;             // 58.3%
    profitFactor: number;        // 1.92
    expectancy: number;          // $125 per trade
  };
  
  // Risk Metrics
  risk: {
    valueAtRisk95: number;       // -$2,340
    conditionalVaR: number;      // -$3,120
    downsideDeviation: number;   // 1.2%
    beta: number;                // 0.65
    correlation: number;         // 0.72
  };
  
  // Trade Analysis
  trades: {
    totalTrades: number;         // 156
    avgTradeReturn: number;      // 0.82%
    avgWinningTrade: number;     // 2.15%
    avgLosingTrade: number;      // -0.93%
    largestWin: number;          // $5,230
    largestLoss: number;         // -$1,890
    avgHoldTime: number;         // 3.2 days
    longestHoldTime: number;     // 12 days
  };
  
  // Execution Quality
  execution: {
    avgSlippage: number;         // 0.05%
    totalFees: number;           // $432
    feesAsPercentOfProfit: number; // 8.2%
    avgFillTime: number;         // 230ms
    rejectedOrders: number;      // 2
  };
  
  // Agent Performance
  agentMetrics: Map<string, {
    signalCount: number;
    accuracy: number;
    profitContribution: number;
    avgConfidence: number;
  }>;
  
  // Time Analysis
  timeAnalysis: {
    bestHour: number;           // 14 (2 PM)
    worstHour: number;          // 9 (9 AM)
    bestDayOfWeek: string;      // "Wednesday"
    monthlyReturns: number[];   // [2.1, -0.5, 3.2, ...]
    rollingReturns: {
      '30d': number[];
      '90d': number[];
      '180d': number[];
    };
  };
}
```

### 13.4 Backtest Analysis Tools

```typescript
class BacktestAnalyzer {
  // Monte Carlo simulation
  async runMonteCarloSimulation(
    results: BacktestResults,
    iterations: number = 10000
  ): Promise<MonteCarloResults> {
    const tradeReturns = results.trades.returns;
    const simulations: number[][] = [];
    
    for (let i = 0; i < iterations; i++) {
      // Randomly resample trades
      const simulated = this.bootstrapSample(tradeReturns);
      simulations.push(this.calculateEquityCurve(simulated));
    }
    
    return {
      medianReturn: this.percentile(simulations, 50),
      confidenceInterval95: [
        this.percentile(simulations, 2.5),
        this.percentile(simulations, 97.5)
      ],
      probabilityOfLoss: this.calcProbabilityOfLoss(simulations),
      expectedDrawdown: this.calcExpectedDrawdown(simulations)
    };
  }
  
  // Walk-forward analysis
  async walkForwardAnalysis(
    data: HistoricalData,
    windowSize: number,
    stepSize: number
  ): Promise<WalkForwardResults> {
    const results: PeriodResult[] = [];
    
    for (let start = 0; start < data.length - windowSize; start += stepSize) {
      // Train on window
      const trainData = data.slice(start, start + windowSize * 0.7);
      const testData = data.slice(start + windowSize * 0.7, start + windowSize);
      
      // Optimize parameters
      const optimalParams = await this.optimizeParameters(trainData);
      
      // Test on out-of-sample
      const testResults = await this.runBacktest(testData, optimalParams);
      
      results.push({
        period: { start, end: start + windowSize },
        trainPerformance: optimalParams.performance,
        testPerformance: testResults.performance,
        parameterStability: this.compareParams(optimalParams, this.previousParams)
      });
    }
    
    return {
      periods: results,
      avgOutOfSampleReturn: mean(results.map(r => r.testPerformance.return)),
      robustnessScore: this.calculateRobustness(results)
    };
  }
  
  // Generate report
  async generateReport(sessionId: string): Promise<BacktestReport> {
    const results = await this.getResults(sessionId);
    const session = await this.getSession(sessionId);
    
    return {
      executive: this.generateExecutiveSummary(results),
      performance: this.generatePerformanceReport(results),
      risk: this.generateRiskReport(results),
      trades: this.generateTradeAnalysis(results),
      parameters: this.documentParameters(session.parameters),
      charts: await this.generateCharts(results),
      recommendations: this.generateRecommendations(results)
    };
  }
}
```

### 13.5 Parameter Optimization Tracking

```typescript
class OptimizationTracker {
  // Track optimization history
  async recordOptimization(optimization: OptimizationRun): Promise<void> {
    await this.db.insert('optimizations', {
      id: optimization.id,
      sessionId: optimization.sessionId,
      algorithm: optimization.algorithm, // 'grid_search', 'evolutionary', 'bayesian'
      searchSpace: optimization.searchSpace,
      bestParameters: optimization.bestParameters,
      bestScore: optimization.bestScore,
      iterations: optimization.iterations,
      duration: optimization.duration,
      convergenceHistory: optimization.convergenceHistory
    });
  }
  
  // Visualize optimization landscape
  async generateOptimizationReport(optimizationId: string): Promise<OptimizationReport> {
    const opt = await this.getOptimization(optimizationId);
    
    return {
      parameterImportance: this.analyzeParameterImportance(opt),
      sensitivityAnalysis: this.performSensitivityAnalysis(opt),
      convergencePlot: this.plotConvergence(opt.convergenceHistory),
      parameterCorrelations: this.analyzeParameterCorrelations(opt),
      optimalRegions: this.identifyOptimalRegions(opt)
    };
  }
  
  // Auto-tune based on recent performance
  async suggestParameterUpdates(
    currentParams: ParameterSet,
    recentPerformance: PerformanceMetrics
  ): Promise<ParameterSuggestions> {
    // Analyze what's not working
    const issues = this.identifyPerformanceIssues(recentPerformance);
    
    // Map issues to parameter adjustments
    const suggestions = issues.map(issue => ({
      parameter: issue.relatedParameter,
      currentValue: currentParams[issue.relatedParameter],
      suggestedValue: this.calculateSuggestion(issue),
      reasoning: issue.explanation,
      expectedImprovement: issue.potentialImprovement
    }));
    
    return {
      suggestions,
      confidence: this.calculateSuggestionConfidence(suggestions),
      backtestRecommended: suggestions.length > 2
    };
  }
}
```

## 14. Historical Data Management

### 14.1 Data Collection Strategy

```typescript
class HistoricalDataManager {
  private db: DuckDBConnection;
  private dataCollector: DataCollector;
  
  // Dual approach: Live caching + batch import
  async initializeDataCollection(): Promise<void> {
    // 1. Check existing data coverage
    const coverage = await this.checkDataCoverage();
    
    // 2. Backfill missing historical data
    if (coverage.gaps.length > 0) {
      await this.backfillHistoricalData(coverage.gaps);
    }
    
    // 3. Start live data collection
    await this.startLiveDataCollection();
    
    // 4. Schedule periodic updates
    this.scheduleDataMaintenance();
  }
  
  // Live trading continuously caches data
  async startLiveDataCollection(): Promise<void> {
    this.dataCollector.on('candle', async (candle) => {
      // Store in DuckDB immediately
      await this.db.run(`
        INSERT INTO candles (timestamp, symbol, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (timestamp, symbol) DO UPDATE SET
          high = GREATEST(high, EXCLUDED.high),
          low = LEAST(low, EXCLUDED.low),
          close = EXCLUDED.close,
          volume = volume + EXCLUDED.volume
      `, [candle.timestamp, candle.symbol, candle.open, 
          candle.high, candle.low, candle.close, candle.volume]);
    });
  }
  
  // Batch import historical data from Coinbase
  async backfillHistoricalData(gaps: DataGap[]): Promise<void> {
    for (const gap of gaps) {
      this.logger.info(`Backfilling data for ${gap.symbol} from ${gap.start} to ${gap.end}`);
      
      // Coinbase allows max 300 candles per request
      const chunks = this.createTimeChunks(gap.start, gap.end, 300);
      
      for (const chunk of chunks) {
        try {
          const candles = await this.fetchHistoricalCandles(
            gap.symbol,
            chunk.start,
            chunk.end,
            gap.granularity
          );
          
          await this.batchInsertCandles(candles);
          
          // Rate limit compliance
          await this.sleep(100); // 10 requests per second
          
        } catch (error) {
          this.logger.error(`Failed to fetch chunk: ${error}`);
          // Continue with next chunk
        }
      }
    }
  }
  
  // Fetch from Coinbase REST API
  private async fetchHistoricalCandles(
    symbol: string,
    start: Date,
    end: Date,
    granularity: number
  ): Promise<Candle[]> {
    const response = await this.coinbaseClient.get(
      `/products/${symbol}/candles`,
      {
        start: start.toISOString(),
        end: end.toISOString(),
        granularity // 60, 300, 900, 3600, 21600, 86400
      }
    );
    
    return response.data.map(this.parseCoinbaseCandle);
  }
}
```

### 14.2 Data Storage Architecture

```typescript
// DuckDB schema optimized for time-series queries
const DATA_SCHEMA = `
-- Main candle data partitioned by month
CREATE TABLE candles (
  timestamp TIMESTAMP NOT NULL,
  symbol VARCHAR NOT NULL,
  timeframe INTEGER NOT NULL, -- 60, 300, 900, 3600, etc
  open DECIMAL(20,8) NOT NULL,
  high DECIMAL(20,8) NOT NULL,
  low DECIMAL(20,8) NOT NULL,
  close DECIMAL(20,8) NOT NULL,
  volume DECIMAL(20,8) NOT NULL,
  trades INTEGER,
  -- Computed columns for faster queries
  returns DECIMAL(10,6) GENERATED ALWAYS AS ((close - open) / open),
  log_returns DECIMAL(10,6) GENERATED ALWAYS AS (ln(close / open)),
  true_range DECIMAL(20,8) GENERATED ALWAYS AS (
    GREATEST(high - low, ABS(high - close), ABS(low - close))
  ),
  PRIMARY KEY (symbol, timeframe, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE candles_2024_01 PARTITION OF candles
  FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... more partitions

-- Indexes for common queries
CREATE INDEX idx_symbol_time ON candles (symbol, timestamp);
CREATE INDEX idx_timeframe ON candles (timeframe, timestamp);

-- Aggregated data for faster backtesting
CREATE TABLE candles_daily AS
  SELECT 
    DATE_TRUNC('day', timestamp) as timestamp,
    symbol,
    FIRST(open) as open,
    MAX(high) as high,
    MIN(low) as low,
    LAST(close) as close,
    SUM(volume) as volume
  FROM candles
  WHERE timeframe = 60
  GROUP BY DATE_TRUNC('day', timestamp), symbol;

-- Technical indicators pre-computed
CREATE TABLE indicators AS
  SELECT
    c.*,
    -- SMA
    AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 19 PRECEDING) as sma_20,
    AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp ROWS 49 PRECEDING) as sma_50,
    -- RSI components
    AVG(CASE WHEN returns > 0 THEN returns ELSE 0 END) 
      OVER (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING) as avg_gain_14,
    AVG(CASE WHEN returns < 0 THEN ABS(returns) ELSE 0 END) 
      OVER (PARTITION BY symbol ORDER BY timestamp ROWS 13 PRECEDING) as avg_loss_14
  FROM candles c;
`;

class DataStorageOptimizer {
  // Compress old data
  async compressHistoricalData(): Promise<void> {
    // Convert old minute candles to higher timeframes
    await this.db.run(`
      INSERT INTO candles_compressed
      SELECT 
        DATE_TRUNC('hour', timestamp) as timestamp,
        symbol,
        3600 as timeframe,
        FIRST(open) as open,
        MAX(high) as high,
        MIN(low) as low,
        LAST(close) as close,
        SUM(volume) as volume
      FROM candles
      WHERE timeframe = 60 
        AND timestamp < NOW() - INTERVAL '3 months'
      GROUP BY DATE_TRUNC('hour', timestamp), symbol
    `);
    
    // Delete detailed old data
    await this.db.run(`
      DELETE FROM candles 
      WHERE timeframe = 60 
        AND timestamp < NOW() - INTERVAL '3 months'
    `);
  }
  
  // Ensure data quality
  async validateDataIntegrity(): Promise<DataValidation> {
    const issues = [];
    
    // Check for gaps
    const gaps = await this.db.all(`
      WITH time_series AS (
        SELECT 
          symbol,
          timestamp,
          LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as next_timestamp
        FROM candles
        WHERE timeframe = 3600
      )
      SELECT 
        symbol,
        timestamp as gap_start,
        next_timestamp as gap_end,
        EXTRACT(EPOCH FROM (next_timestamp - timestamp)) / 3600 as hours_missing
      FROM time_series
      WHERE next_timestamp - timestamp > INTERVAL '1 hour'
        AND hours_missing > 1
    `);
    
    if (gaps.length > 0) {
      issues.push({ type: 'gaps', details: gaps });
    }
    
    // Check for outliers
    const outliers = await this.db.all(`
      SELECT *
      FROM candles
      WHERE returns > 0.5 OR returns < -0.5  -- 50% moves suspicious
         OR high / low > 2  -- 100% range suspicious
    `);
    
    if (outliers.length > 0) {
      issues.push({ type: 'outliers', details: outliers });
    }
    
    return { valid: issues.length === 0, issues };
  }
}
```

### 14.3 Data Sources & Quality

```typescript
class DataSourceManager {
  private sources = {
    primary: new CoinbaseDataSource(),
    fallback: new AlternativeDataSource(),
    validator: new DataValidator()
  };
  
  // Multi-source data reconciliation
  async fetchReliableData(
    symbol: string,
    start: Date,
    end: Date
  ): Promise<Candle[]> {
    // Try primary source
    let data = await this.sources.primary.fetch(symbol, start, end);
    
    // Validate data quality
    const validation = await this.sources.validator.validate(data);
    
    if (!validation.isValid) {
      // Try fallback source
      const fallbackData = await this.sources.fallback.fetch(symbol, start, end);
      
      // Reconcile differences
      data = this.reconcileData(data, fallbackData, validation.issues);
    }
    
    // Fill any remaining gaps
    data = await this.interpolateMissingData(data);
    
    return data;
  }
  
  // Handle exchange maintenance windows
  private async interpolateMissingData(data: Candle[]): Promise<Candle[]> {
    const filled: Candle[] = [];
    
    for (let i = 0; i < data.length - 1; i++) {
      filled.push(data[i]);
      
      const gap = (data[i + 1].timestamp - data[i].timestamp) / (60 * 1000);
      
      if (gap > 1) {
        // Linear interpolation for small gaps
        if (gap <= 5) {
          const interpolated = this.linearInterpolate(data[i], data[i + 1], gap - 1);
          filled.push(...interpolated);
        } else {
          // Mark as missing for larger gaps
          filled.push(this.createMissingCandle(data[i], data[i + 1]));
        }
      }
    }
    
    filled.push(data[data.length - 1]);
    return filled;
  }
}
```

### 14.4 Efficient Data Access

```typescript
class BacktestDataProvider {
  private cache: LRUCache<string, Candle[]>;
  
  // Optimized data loading for backtesting
  async loadDataForBacktest(
    symbol: string,
    start: Date,
    end: Date,
    indicators: string[]
  ): Promise<BacktestData> {
    const cacheKey = `${symbol}_${start}_${end}_${indicators.join(',')}`;
    
    // Check memory cache
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }
    
    // Load with pre-computed indicators
    const data = await this.db.all(`
      SELECT 
        c.*,
        ${this.buildIndicatorQuery(indicators)}
      FROM candles c
      LEFT JOIN indicators i ON c.timestamp = i.timestamp AND c.symbol = i.symbol
      WHERE c.symbol = ?
        AND c.timestamp BETWEEN ? AND ?
        AND c.timeframe = ?
      ORDER BY c.timestamp
    `, [symbol, start, end, this.getOptimalTimeframe(start, end)]);
    
    const result = {
      candles: data,
      metadata: {
        actualStart: data[0].timestamp,
        actualEnd: data[data.length - 1].timestamp,
        gaps: this.identifyGaps(data),
        quality: this.assessDataQuality(data)
      }
    };
    
    this.cache.set(cacheKey, result);
    return result;
  }
  
  // Stream data for large backtests
  async *streamData(
    symbol: string,
    start: Date,
    end: Date,
    batchSize: number = 10000
  ): AsyncIterator<Candle[]> {
    let offset = 0;
    
    while (true) {
      const batch = await this.db.all(`
        SELECT * FROM candles
        WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        LIMIT ? OFFSET ?
      `, [symbol, start, end, batchSize, offset]);
      
      if (batch.length === 0) break;
      
      yield batch;
      offset += batchSize;
      
      if (batch.length < batchSize) break;
    }
  }
}
```

## 15. Required Technical Indicators

### 15.1 Minimal Indicator Set

```typescript
interface RequiredIndicators {
  // Price-based (for multiple agents)
  atr: ATR;                    // VolatilityAgent: trail distance calculation
  swingHighsLows: SwingPoints; // GridManager: optimal spacing, MarketStructureAgent
  
  // Momentum (for MomentumAgent)
  rsi: RSI;                    // Divergence detection
  macd: MACD;                  // Momentum shifts
  
  // Volume (for VolumeProfileAgent)
  volumeMA: SMA;               // Volume spike detection (current vs average)
  vwap: VWAP;                  // Volume-weighted average price
  
  // Market Structure (for RegimeAgent)
  sma20: SMA;                  // Short-term trend
  sma50: SMA;                  // Medium-term trend
  
  // Volatility (for multiple agents)
  bollingerBands: BollingerBands; // Volatility expansion/contraction
}

class IndicatorCalculator {
  // ATR - Average True Range (for VolatilityAgent)
  calculateATR(candles: Candle[], period: number = 14): number {
    const trueRanges = candles.slice(1).map((candle, i) => {
      const high = candle.high;
      const low = candle.low;
      const prevClose = candles[i].close;
      
      return Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
    });
    
    return this.sma(trueRanges, period);
  }
  
  // Swing High/Low Detection (for grid spacing & support/resistance)
  findSwingPoints(candles: Candle[], lookback: number = 5): SwingPoint[] {
    const swings: SwingPoint[] = [];
    
    for (let i = lookback; i < candles.length - lookback; i++) {
      const isSwingHigh = candles[i].high === Math.max(
        ...candles.slice(i - lookback, i + lookback + 1).map(c => c.high)
      );
      
      const isSwingLow = candles[i].low === Math.min(
        ...candles.slice(i - lookback, i + lookback + 1).map(c => c.low)
      );
      
      if (isSwingHigh) {
        swings.push({ type: 'high', price: candles[i].high, index: i });
      }
      if (isSwingLow) {
        swings.push({ type: 'low', price: candles[i].low, index: i });
      }
    }
    
    return swings;
  }
  
  // RSI - Relative Strength Index (for MomentumAgent)
  calculateRSI(candles: Candle[], period: number = 14): number[] {
    const changes = candles.slice(1).map((candle, i) => 
      candle.close - candles[i].close
    );
    
    const gains = changes.map(c => c > 0 ? c : 0);
    const losses = changes.map(c => c < 0 ? Math.abs(c) : 0);
    
    const avgGain = this.sma(gains.slice(-period), period);
    const avgLoss = this.sma(losses.slice(-period), period);
    
    const rs = avgGain / avgLoss;
    return candles.map((_, i) => {
      if (i < period) return 50; // Neutral
      return 100 - (100 / (1 + rs));
    });
  }
  
  // MACD - Moving Average Convergence Divergence
  calculateMACD(
    candles: Candle[], 
    fast: number = 12, 
    slow: number = 26, 
    signal: number = 9
  ): MACDResult {
    const closes = candles.map(c => c.close);
    const ema12 = this.ema(closes, fast);
    const ema26 = this.ema(closes, slow);
    
    const macdLine = ema12.map((v, i) => v - ema26[i]);
    const signalLine = this.ema(macdLine, signal);
    const histogram = macdLine.map((v, i) => v - signalLine[i]);
    
    return { macdLine, signalLine, histogram };
  }
  
  // Simple Moving Average (base calculation)
  sma(values: number[], period: number): number {
    if (values.length < period) return values[values.length - 1];
    
    const sum = values.slice(-period).reduce((a, b) => a + b, 0);
    return sum / period;
  }
  
  // Exponential Moving Average
  ema(values: number[], period: number): number[] {
    const multiplier = 2 / (period + 1);
    const ema: number[] = [values[0]];
    
    for (let i = 1; i < values.length; i++) {
      ema.push(
        (values[i] - ema[i - 1]) * multiplier + ema[i - 1]
      );
    }
    
    return ema;
  }
}
```

### 15.2 Indicator Usage by Agent

```typescript
const AGENT_INDICATOR_MAP = {
  // Core Agents
  VolatilityAgent: ['atr', 'bollingerBands'],
  MomentumAgent: ['rsi', 'macd'],
  VolumeProfileAgent: ['volumeMA', 'vwap'],
  MarketStructureAgent: ['swingHighsLows', 'sma20', 'sma50'],
  RegimeAgent: ['atr', 'sma20', 'sma50'],
  
  // Advanced Agents (minimal requirements)
  TimeDecayAgent: [], // Uses time, not indicators
  MicrostructureAgent: [], // Uses bid/ask spread
  CorrelationAgent: [], // Uses external asset prices
  SentimentAgent: ['rsi'], // For extreme readings
  
  // Experimental Agents
  EntropyTransitionAgent: [], // Uses price distribution
  HarmonicResonanceAgent: [], // Uses FFT, not traditional indicators
  // ... other experimental agents use custom calculations
};

// Total unique indicators needed: 8
// ATR, RSI, MACD, SMA(20), SMA(50), Volume MA, VWAP, Swing Detection
```

### 15.3 Optimized Indicator Calculation

```typescript
class IndicatorEngine {
  private cache = new Map<string, any>();
  
  // Calculate all required indicators in one pass
  async calculateAll(candles: Candle[]): Promise<IndicatorSet> {
    // Single pass for efficiency
    const closes = candles.map(c => c.close);
    const volumes = candles.map(c => c.volume);
    
    // Price-based
    const atr = this.calc.calculateATR(candles);
    const swings = this.calc.findSwingPoints(candles);
    
    // Moving averages (shared calculation)
    const sma20 = this.calc.sma(closes, 20);
    const sma50 = this.calc.sma(closes, 50);
    
    // Momentum
    const rsi = this.calc.calculateRSI(candles);
    const macd = this.calc.calculateMACD(candles);
    
    // Volume
    const volumeMA = this.calc.sma(volumes, 20);
    const vwap = this.calc.calculateVWAP(candles);
    
    // Derived
    const bollingerBands = {
      middle: sma20,
      upper: sma20 + (2 * this.stdDev(closes, 20)),
      lower: sma20 - (2 * this.stdDev(closes, 20))
    };
    
    return {
      atr,
      swings,
      sma20,
      sma50,
      rsi,
      macd,
      volumeMA,
      vwap,
      bollingerBands,
      timestamp: candles[candles.length - 1].timestamp
    };
  }
}
```

## 16. Chart Visualization Requirements

### 16.1 Essential Chart Types

```typescript
interface RequiredCharts {
  // Primary: Candlestick (OHLC)
  candlestick: {
    purpose: "Show price action, patterns, support/resistance";
    requiredBy: ["AIPatternAgent", "MarketStructureAgent", "Human monitoring"];
    features: [
      "Swing high/low markers",
      "Grid level overlays", 
      "Trailing order visualization",
      "Volume bars below"
    ];
  };
  
  // Secondary: Grid Heatmap
  gridHeatmap: {
    purpose: "Visualize grid performance and active zones";
    requiredBy: ["GridManager", "Performance monitoring"];
    features: [
      "Profit/loss by grid level",
      "Order density",
      "Time spent at each level",
      "Active trailing orders"
    ];
  };
  
  // Tertiary: Performance Dashboard
  performanceDashboard: {
    purpose: "Real-time P&L, drawdown, agent performance";
    requiredBy: ["Risk monitoring", "Agent optimization"];
    features: [
      "Equity curve",
      "Drawdown chart",
      "Agent contribution pie chart",
      "Win/loss distribution"
    ];
  };
}

class ChartRenderer {
  // Candlestick with overlays for swing trading
  renderTradingChart(data: MarketData, config: ChartConfig): ChartOutput {
    return {
      type: 'candlestick',
      data: {
        ohlc: data.candles,
        overlays: [
          {
            type: 'line',
            data: data.indicators.sma20,
            color: 'blue',
            width: 2
          },
          {
            type: 'line', 
            data: data.indicators.sma50,
            color: 'orange',
            width: 2
          },
          {
            type: 'scatter',
            data: data.swingPoints,
            symbol: 'triangle',
            color: p => p.type === 'high' ? 'red' : 'green'
          },
          {
            type: 'horizontalLines',
            data: this.activeGridLevels,
            style: 'dashed',
            opacity: 0.3
          }
        ],
        panels: [
          {
            height: '20%',
            indicator: 'volume',
            type: 'bar',
            color: c => c.close > c.open ? 'green' : 'red'
          },
          {
            height: '15%',
            indicator: 'rsi',
            type: 'line',
            levels: [30, 70]
          }
        ]
      }
    };
  }
  
  // Grid performance heatmap
  renderGridHeatmap(gridData: GridPerformance): ChartOutput {
    return {
      type: 'heatmap',
      data: {
        x: gridData.timestamps,
        y: gridData.gridLevels,
        z: gridData.profitMatrix, // 2D array of profits
        colorScale: {
          min: -100,
          zero: 0,
          max: 100,
          colors: ['red', 'white', 'green']
        },
        annotations: gridData.activeOrders.map(order => ({
          x: order.timestamp,
          y: order.gridLevel,
          text: order.type === 'buy' ? '↑' : '↓'
        }))
      }
    };
  }
}
```

### 16.2 Chart Requirements by Use Case

```typescript
const CHART_USE_CASES = {
  // 1. AI Analysis (for AIPatternAgent)
  aiAnalysis: {
    chartType: 'candlestick',
    timeframe: '4h',
    requiredElements: [
      'OHLC bars',
      'Volume profile',
      'Recent swing points',
      'No indicators (clean chart for pattern recognition)'
    ],
    exportFormat: 'PNG base64 for Claude API'
  },
  
  // 2. Real-time Monitoring
  monitoring: {
    chartType: 'candlestick',
    timeframe: 'dynamic (1h default)',
    requiredElements: [
      'Live price updates',
      'Active grid levels',
      'Trailing order positions',
      'Entry/exit markers',
      'Current P&L overlay'
    ],
    updateFrequency: 'Every new candle'
  },
  
  // 3. Backtesting Visualization
  backtesting: {
    chartType: 'multi-panel',
    panels: [
      'Candlestick with trades',
      'Equity curve',
      'Drawdown',
      'Agent signals'
    ],
    interactivity: 'Zoom, pan, click for trade details'
  },
  
  // 4. Performance Analysis
  performance: {
    chartTypes: [
      'Equity curve (line)',
      'Monthly returns (heatmap calendar)',
      'Trade distribution (histogram)',
      'Agent performance (radar chart)'
    ]
  }
};
```

### 16.3 Minimal Chart Library Requirements

```typescript
interface ChartLibraryRequirements {
  // Lightweight, performant library needed
  features: [
    'Canvas rendering (not SVG) for performance',
    'Real-time updates without full redraw',
    'Zoom/pan for historical analysis',
    'Export to image for AI analysis',
    'Responsive design'
  ];
  
  // Suggested: Lightweight library or custom canvas
  implementation: {
    option1: 'Custom canvas implementation (most control)',
    option2: 'Lightweight library like uPlot (fast)',
    option3: 'TradingView Lightweight Charts (familiar)'
  };
}

// Custom lightweight implementation
class MinimalChartEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private data: Candle[] = [];
  
  // Optimized rendering for real-time updates
  updateLastCandle(candle: Candle) {
    // Only redraw the last candle area
    const x = this.getX(this.data.length - 1);
    const width = this.candleWidth;
    
    // Clear previous candle
    this.ctx.clearRect(x - 1, 0, width + 2, this.height);
    
    // Draw new candle
    this.drawCandle(candle, x);
    
    // Update overlays for last position only
    this.updateOverlays(x);
  }
  
  // Efficient grid level rendering
  drawGridLevels(levels: number[]) {
    this.ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
    this.ctx.setLineDash([5, 5]);
    
    levels.forEach(price => {
      const y = this.priceToY(price);
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.width, y);
      this.ctx.stroke();
    });
  }
}
```

### 16.4 Chart Data Flow

```typescript
class ChartDataManager {
  // Efficient data structure for charts
  private chartData: {
    candles: CircularBuffer<Candle>;      // Last 1000 candles
    indicators: Map<string, CircularBuffer<number>>;
    trades: Trade[];
    gridLevels: number[];
  };
  
  // Update charts on new data
  async onNewCandle(candle: Candle) {
    // Update data buffers
    this.chartData.candles.push(candle);
    
    // Calculate only needed indicators
    const indicators = await this.indicatorEngine.calculateMinimal(
      this.chartData.candles.toArray()
    );
    
    // Notify chart components
    this.emit('chartUpdate', {
      candle,
      indicators: indicators.latest,
      updateType: 'append'
    });
  }
  
  // Generate chart for AI analysis
  async generateAIChart(symbol: string, timeframe: string): Promise<Buffer> {
    const data = await this.getRecentData(symbol, timeframe, 100);
    
    // Clean candlestick chart, no indicators
    const chart = new MinimalChartEngine({
      width: 800,
      height: 600,
      padding: { top: 20, right: 60, bottom: 40, left: 20 }
    });
    
    chart.drawCandles(data.candles);
    chart.drawVolume(data.volume);
    
    return chart.toBuffer('png');
  }
}
```

## 17. Critical Missing Components

### 12.1 Paper Trading Mode
```typescript
class PaperOrderExecutor {
  executeOrder(order: Order): Promise<Fill> {
    const spread = 0.001; // 0.1% spread
    const fees = 0.0005; // 0.05% fee
    const fillPrice = order.side === 'BUY' 
      ? currentPrice * (1 + spread) 
      : currentPrice * (1 - spread);
    return { price: fillPrice, fees: order.size * fillPrice * fees };
  }
}
```

### 12.2 Event Bus Architecture
```typescript
class UnifiedEventBus {
  emit(event: TradingEvent): void;
  on(type: string, handler: EventHandler): void;
}

interface TimeSource {
  now(): Date;
  subscribe(callback: (time: Date) => void): void;
}
```

### 12.3 Type Definitions
```typescript
interface MarketData {
  symbol: string;
  candles: Candle[];
  currentPrice: number;
  volume: number;
  timestamp: Date;
}

interface TradeResult {
  orderId: string;
  pnl: number;
  fees: number;
}
```

### 12.4 Initial Entry Strategy
- First buy triggers when price drops 2% from session start
- Or when first agent signals BUY with >70% confidence
- Grid activates around entry price

### 12.5 Agent Registry
```typescript
class AgentRegistry {
  loadExternalAgents(): Promise<void>;
  createAgent(id: string): Promise<ITradeAgent>;
}
```

## 13. Future Enhancements

1. **Multi-exchange support**: Binance, Kraken integration
2. **Advanced order types**: Iceberg, TWAP orders  
3. **Portfolio mode**: Trade multiple pairs
4. **Mobile app**: iOS/Android monitoring
5. **Social features**: Strategy sharing marketplace
6. **Advanced AI**: GPT-4/Claude integration for market analysis
7. **DeFi integration**: DEX trading support

## Appendix A: Configuration Example

```json
{
  "trading": {
    "symbol": "BTC-USD",
    "mode": "paper",
    "capital": 10000,
    "maxPositionSize": 0.25,
    "maxConcurrentTrades": 3
  },
  "grid": {
    "autoSpacing": true,
    "minSpacing": 0.005,
    "maxSpacing": 0.02
  },
  "agents": {
    "enabled": [
      "volatility",
      "momentum", 
      "volume",
      "ai_pattern"
    ],
    "aiConfig": {
      "model": "claude-3",
      "maxCallsPerHour": 100
    }
  },
  "risk": {
    "maxDrawdown": 0.15,
    "dailyLossLimit": 0.05,
    "stopLossPercent": 0.02
  }
}
```

## Appendix B: Error Codes

| Code | Description | Action |
|------|-------------|--------|
| E001 | Connection lost | Auto-reconnect with backoff |
| E002 | Insufficient funds | Reduce position size |
| E003 | Rate limit exceeded | Queue and retry |
| E004 | Invalid API response | Log and use fallback |
| E005 | Agent timeout | Skip agent signal |
| E006 | Risk limit exceeded | Reject order |
| E007 | State corruption | Restore from checkpoint |

## Appendix C: Implementation Priority
1. Core engine with paper trading
2. Basic agents (volatility, momentum)
3. Coinbase integration
4. Backtesting framework
5. Web UI
6. Advanced agents
7. AI integration

## Appendix D: Development Plan

### Phase 1: Foundation (Week 1-2)

#### 1.1 Project Setup & Configuration
- Initialize monorepo with npm workspaces
- Set up TypeScript configuration
- Configure ESLint/Prettier
- Set up test framework (node:test)
- Create base package structure

**Deliverable**: Working monorepo with proper tooling

#### 1.2 Core Data Models & Types
- Define shared types (`packages/shared/src/types/`)
  - `market-data.ts` - Candle, Tick, OrderBook interfaces
  - `orders.ts` - Order, Fill, OrderState interfaces
  - `agents.ts` - AgentSignal, AgentState interfaces
  - `config.ts` - Configuration interfaces
- Create domain models with validation

**Deliverable**: Type-safe foundation for entire system

#### 1.3 Event Bus Architecture
- Implement unified event system
- Create event types and handlers
- Build time abstraction layer (live/backtest)
- Add event logging and replay capability

**Deliverable**: Working event-driven architecture

### Phase 2: Data Layer (Week 2-3)

#### 2.1 DuckDB Integration
- Set up DuckDB connection management
- Create schema migrations
- Implement data models (candles, orders, trades)
- Build query builders and repositories

**Deliverable**: Persistent storage layer

#### 2.2 Market Data Pipeline
- Create `MarketDataPipeline` interface
- Implement `BacktestDataFeed` (reads from DuckDB)
- Build data validation and cleaning
- Add data streaming for large datasets

**Deliverable**: Unified data access for backtest mode

#### 2.3 Historical Data Manager
- Build data import from CSV/JSON
- Create data validation pipeline
- Implement gap detection and filling
- Add data compression for old data

**Deliverable**: Ability to load and manage historical data

### Phase 3: Trading Engine Core (Week 3-4)

#### 3.1 Order Management System
- Implement `Order` state machine
- Create `OrderLifecycleManager`
- Build order validation logic
- Add order persistence and recovery

**Deliverable**: Basic order creation and tracking

#### 3.2 Paper Trading Executor
- Implement simulated order execution
- Add realistic spread/slippage simulation
- Create fill generation logic
- Build position tracking

**Deliverable**: Complete paper trading mode

#### 3.3 Risk Management
- Implement position sizing calculator
- Add drawdown monitoring
- Create risk validation for orders
- Build capital allocation system

**Deliverable**: Basic risk controls

### Phase 4: Grid & Trailing Logic (Week 4-5)

#### 4.1 Grid Manager
- Implement grid level calculation
- Create grid activation logic
- Build optimal spacing algorithm
- Add grid state persistence

**Deliverable**: Dynamic grid system

#### 4.2 Trailing Order Implementation
- Create `TrailingOrderManager`
- Implement trail distance calculation
- Build trigger detection logic
- Add order modification system

**Deliverable**: Working trailing orders

#### 4.3 Backtesting Engine
- Create backtest runner
- Implement time simulation
- Add performance tracking
- Build result aggregation

**Deliverable**: Basic backtesting capability

### Phase 5: Agent System (Week 5-6)

#### 5.1 Agent Framework
- Define `ITradeAgent` interface
- Create `AgentOrchestrator`
- Implement consensus mechanism
- Add agent performance tracking

**Deliverable**: Pluggable agent architecture

#### 5.2 Core Agents
- Implement `VolatilityAgent` (uses ATR)
- Build `MomentumAgent` (RSI/MACD)
- Create `VolumeProfileAgent`
- Add `MarketStructureAgent`
- Implement `RegimeAgent`

**Deliverable**: 5 working core agents

#### 5.3 Technical Indicators
- Implement required indicators (ATR, RSI, MACD, SMA)
- Create indicator calculation engine
- Add indicator caching
- Build streaming indicator updates

**Deliverable**: Efficient indicator library

### Phase 6: Live Trading Preparation (Week 6-7)

#### 6.1 Coinbase Integration
- Implement WebSocket connection
- Add REST API client
- Create order submission system
- Build authentication system

**Deliverable**: Coinbase connectivity (paper mode)

#### 6.2 Network Resilience
- Implement retry logic with backoff
- Add connection monitoring
- Create failover mechanisms
- Build order recovery system

**Deliverable**: Robust network layer

#### 6.3 Real-time Data Feed
- Connect to Coinbase WebSocket
- Implement data normalization
- Add data caching layer
- Create fallback REST polling

**Deliverable**: Live market data

### Phase 7: Monitoring & API (Week 7-8)

#### 7.1 REST API
- Set up Express server
- Implement trading control endpoints
- Add configuration endpoints
- Create performance endpoints

**Deliverable**: HTTP API for control

#### 7.2 WebSocket Server
- Implement Socket.io server
- Create real-time event streaming
- Add subscription management
- Build client authentication

**Deliverable**: Real-time updates

#### 7.3 Logging & Metrics
- Set up structured logging
- Implement performance metrics
- Add trade logging
- Create system health monitoring

**Deliverable**: Observability system

### Phase 8: User Interface (Week 8-9)

#### 8.1 React Frontend Setup
- Initialize React with Vite
- Set up routing and state management
- Create component library
- Implement WebSocket client

**Deliverable**: Basic React app

#### 8.2 Trading Dashboard
- Build main trading view
- Create position display
- Add order management UI
- Implement configuration panel

**Deliverable**: Functional trading UI

#### 8.3 Charts & Visualization
- Integrate lightweight charting library
- Create candlestick chart component
- Add grid visualization
- Build performance charts

**Deliverable**: Essential charts

### Phase 9: Advanced Features (Week 9-10)

#### 9.1 Self-Tuning System
- Implement parameter optimization
- Create evolutionary algorithm
- Add performance feedback loop
- Build parameter versioning

**Deliverable**: Self-optimizing parameters

#### 9.2 Advanced Agents
- Implement `TimeDecayAgent`
- Add pattern detection logic
- Create correlation analysis
- Build sentiment indicators

**Deliverable**: Enhanced decision making

#### 9.3 Comprehensive Backtesting
- Add Monte Carlo simulation
- Implement walk-forward analysis
- Create detailed statistics
- Build comparison tools

**Deliverable**: Professional backtesting

### Phase 10: Production Readiness (Week 10-11)

#### 10.1 Testing Suite
- Write unit tests for all components
- Create integration tests
- Add E2E tests for critical paths
- Build performance benchmarks

**Deliverable**: >90% test coverage

#### 10.2 Documentation
- Write API documentation
- Create user guide
- Add architecture diagrams
- Build troubleshooting guide

**Deliverable**: Complete documentation

#### 10.3 Deployment & Operations
- Set up PM2 configuration
- Create deployment scripts
- Add health checks
- Build monitoring alerts

**Deliverable**: Production-ready system

### Phase 11: AI Integration (Week 11-12)

#### 11.1 Claude Integration
- Implement chart generation for AI
- Create structured prompts
- Add response parsing
- Build cost tracking

**Deliverable**: AI pattern recognition

#### 11.2 AI Resource Management
- Implement request batching
- Add caching layer
- Create budget controls
- Build fallback strategies

**Deliverable**: Cost-effective AI usage

### Critical Path & Dependencies

```
1. Event Bus → Data Models → Order System → Paper Trading
2. DuckDB → Historical Data → Backtesting
3. Grid Logic → Trailing Orders → Agent System
4. Paper Trading → Live Integration → Production
```

### Testing Strategy Throughout

Each component must have:
- Unit tests using node:test
- Integration tests with dependencies
- Performance benchmarks
- Error scenario coverage

### File Naming Conventions

All files use kebab-case:
- `market-data-pipeline.ts`
- `trailing-order-manager.ts`
- `volatility-agent.ts`

### Code Style Guidelines

- No semicolons in TypeScript
- Prefer `readonly` for all props
- Use `node:test` for testing
- Functional style where appropriate
- Immutable data structures

### Risk Mitigation

1. **Start with paper trading** - No real money at risk
2. **Extensive backtesting** - Validate strategies first
3. **Gradual feature rollout** - Core features first
4. **Conservative defaults** - Safe parameters initially
5. **Comprehensive logging** - Debug production issues

### Success Metrics

- Phase 1-3: Working backtest with basic grid
- Phase 4-6: Paper trading with live data
- Phase 7-9: Full UI with monitoring
- Phase 10-11: Production-ready with AI

Provides comprehensive blueprint for building production-ready grid trading bot. Architecture designed for maintainability, scalability, and profitability while keeping complexity manageable for retail trading context.