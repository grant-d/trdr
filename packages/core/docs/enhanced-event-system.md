# Enhanced Event System Documentation

## Overview

The Enhanced Event System provides advanced event filtering, priority handling, and comprehensive market data event management for all market data feeds. This system extends the base EventBus with enhanced capabilities while maintaining backward compatibility.

## Architecture

### Core Components

1. **EnhancedEventBus** - Enhanced event bus with filtering and priority handling
2. **EventFilter** - Flexible event filtering system with builder pattern
3. **MarketDataEvents** - Specialized event types for market data
4. **EnhancedMarketDataFeed** - Base class for enhanced market data feeds

### Event Flow

```
Market Data Source → Enhanced Data Feed → Enhanced Event Bus → Filtered Subscribers
                                      ↓
                                  Global Filters → Subscription Filters → Handlers
```

## Event Types

### Enhanced Market Data Events

#### EnhancedTickEvent
Real-time price tick with enhanced metadata and analytics.

```typescript
interface EnhancedTickEvent {
  type: 'market.tick.enhanced'
  timestamp: Date
  symbol: string
  price: number
  volume?: number
  priceChange: number          // Absolute price change from last tick
  priceChangePercent: number   // Percentage change from last tick
  source: string               // Feed source identifier
  feedType: FeedType           // Type of data feed
  sequence: number             // Monotonic sequence number
  priority: EventPriority      // Calculated event priority
  sourceTimestamp?: Date       // Original timestamp from data source
  latency?: number            // Processing latency in milliseconds
}
```

**Priority Classification:**
- `critical`: Price change > 5% or volume > 10x average
- `high`: Price change > 1% or volume > 3x average  
- `medium`: Price change > 0.1% or volume > average
- `low`: All other ticks

#### EnhancedCandleEvent
OHLCV candle data with technical analysis metadata.

```typescript
interface EnhancedCandleEvent {
  type: 'market.candle.enhanced'
  timestamp: Date
  symbol: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  interval: string            // Time interval (1m, 5m, 1h, etc.)
  range: number               // High - Low
  bodySize: number            // |Close - Open|
  upperWick: number           // High - max(Open, Close)
  lowerWick: number           // min(Open, Close) - Low
  typicalPrice: number        // (High + Low + Close) / 3
  candleType: 'bullish' | 'bearish' | 'doji'
  source: string
  feedType: FeedType
  sequence: number
  priority: EventPriority
  sourceTimestamp?: Date
  latency?: number
}
```

#### ConnectionStatusEvent
Data feed connection status changes.

```typescript
interface ConnectionStatusEvent {
  type: 'market.connection'
  timestamp: Date
  source: string
  feedType: FeedType
  status: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting'
  details: {
    reconnectAttempts: number
    uptime: number
    lastError?: string
    subscriptions: string[]
  }
}
```

### Standard Event Types (Backward Compatibility)

The system maintains compatibility with standard events:

- `market.tick` - Basic price tick
- `market.candle` - Basic OHLCV candle
- `system.info` - General system information
- `system.warning` - System warnings
- `system.error` - Error events

## Event Filtering

### Global Filters

Applied to all events before reaching subscribers. Useful for system-wide filtering policies.

```typescript
// Rate limiting filter
enhancedEventBus.addGlobalFilter({
  eventType: 'market.tick.enhanced',
  filter: MarketDataFilters.rateLimit(100), // Max 100 events/second
  priority: 100,
  name: 'tick-rate-limit'
})

// Price change threshold filter
enhancedEventBus.addGlobalFilter({
  eventType: 'market.tick.enhanced', 
  filter: MarketDataFilters.byPriceChangeThreshold(0.001), // 0.1% minimum change
  priority: 90,
  name: 'price-change-threshold'
})
```

### Subscription Filters

Applied to specific subscriptions for targeted filtering.

```typescript
// Subscribe with symbol filter
const subscription = enhancedEventBus.subscribeWithFilter(
  'market.tick.enhanced',
  (event) => console.log('High priority tick:', event),
  {
    filter: MarketDataFilters.bySymbol(['BTC-USD', 'ETH-USD'])
      .and(MarketDataFilters.byPriority(['high', 'critical'])),
    priority: 50
  }
)
```

### Built-in Filters

#### Symbol Filtering
```typescript
MarketDataFilters.bySymbol(['BTC-USD', 'ETH-USD'])
```

#### Price Range Filtering
```typescript
MarketDataFilters.byPriceRange(50000, 70000) // $50k - $70k
```

#### Volume Filtering
```typescript
MarketDataFilters.byMinVolume(1000) // Minimum 1000 volume
```

#### Priority Filtering
```typescript
MarketDataFilters.byPriority(['high', 'critical'])
```

#### Price Change Threshold
```typescript
MarketDataFilters.byPriceChangeThreshold(0.01) // 1% minimum change
```

#### Rate Limiting
```typescript
MarketDataFilters.rateLimit(100) // Max 100 events/second per symbol
```

#### Composite Filtering
```typescript
// Complex filter combining multiple criteria
const filter = MarketDataFilters.bySymbol(['BTC-USD'])
  .and(MarketDataFilters.byPriceRange(50000, 70000))
  .and(MarketDataFilters.byPriority(['high', 'critical']))
  .or(MarketDataFilters.byMinVolume(10000))
```

## Data Feed Integration

### Enhanced Data Feeds

All market data feeds now extend `EnhancedMarketDataFeed` which provides:

1. **Enhanced Event Emission** - Automatic priority calculation and metadata
2. **Global Filter Integration** - Respects system-wide filtering policies  
3. **Performance Metrics** - Latency tracking and event statistics
4. **Connection Status Events** - Automatic status change notifications

### Feed Implementations

#### CoinbaseDataFeed
- Extends `EnhancedMarketDataFeed`
- Uses Coinbase Advanced Trade SDK
- Emits enhanced tick and candle events
- Provides connection status monitoring

#### BacktestDataFeed  
- Extends `EnhancedMarketDataFeed`
- Supports historical data replay with speed control
- Simulates realistic market conditions
- Time-accurate event sequencing

#### PaperTradingFeed
- Extends `EnhancedMarketDataFeed`
- Wraps real data feeds with trading simulation
- Applies market scenarios and execution modeling
- Virtual time support for accelerated testing

### Configuration

Enhanced feeds accept `EnhancedDataFeedConfig`:

```typescript
interface EnhancedDataFeedConfig extends DataFeedConfig {
  enhancedEvents?: boolean              // Enable enhanced features (default: true)
  priceChangeThreshold?: number         // Minimum change threshold (basis points)
  enableCompression?: boolean           // Enable event compression
  compressionWindow?: number            // Compression window (ms)
  maxEventsPerSecond?: number          // Rate limiting
  enableStatistics?: boolean           // Enable detailed statistics
}
```

## Event Serialization

### Serialization Features

- **JSON Serialization** - All events can be serialized to JSON
- **Date Handling** - Proper Date object serialization/deserialization
- **Error Handling** - Error objects preserved with stack traces
- **Compression** - Optional compression for high-frequency data

### Usage

```typescript
// Serialize event
const serialized = EventSerializer.serialize(event)

// Deserialize event  
const event = EventSerializer.deserialize(serialized)

// Export events for analysis
const exportData = await dataFeed.exportEventData(
  ['market.tick.enhanced', 'market.candle.enhanced'],
  { start: new Date('2024-01-01'), end: new Date('2024-01-02') }
)
```

## Performance Considerations

### Event Volume Management

1. **Rate Limiting** - Global and per-symbol rate limits prevent overload
2. **Filtering** - Early filtering reduces downstream processing
3. **Priority Handling** - Critical events bypass rate limits
4. **Batching** - High-frequency events can be batched

### Memory Management

1. **History Limits** - Price history limited to 1000 points per symbol
2. **Event TTL** - Old events automatically cleaned up
3. **Subscription Cleanup** - Automatic cleanup on unsubscribe

### Latency Optimization

1. **Async Processing** - Non-blocking event handling
2. **Priority Queues** - Critical events processed first
3. **Minimal Allocations** - Reuse of event objects where possible

## Consumer Behavior Guidelines

### Subscription Best Practices

1. **Use Specific Filters** - Filter at subscription level to reduce processing
2. **Handle Async Events** - Use async handlers for I/O operations
3. **Monitor Backpressure** - Check `waitForAsyncHandlers()` periodically
4. **Unsubscribe Cleanly** - Always call `unsubscribe()` when done

### Error Handling

1. **Graceful Degradation** - Handle missing or malformed events
2. **Retry Logic** - Implement exponential backoff for transient failures
3. **Circuit Breakers** - Stop processing on repeated failures
4. **Logging** - Log errors but don't crash on individual event failures

### Event Processing Patterns

#### Real-time Processing
```typescript
enhancedEventBus.subscribe('market.tick.enhanced', async (event) => {
  await updatePositions(event)
  await checkStopLosses(event)
})
```

#### Batch Processing
```typescript
const events: EnhancedTickEvent[] = []

enhancedEventBus.subscribe('market.tick.enhanced', (event) => {
  events.push(event)
  
  if (events.length >= 100) {
    processBatch(events)
    events.length = 0
  }
})
```

#### Filtered Processing
```typescript
const subscription = enhancedEventBus.subscribeWithFilter(
  'market.tick.enhanced',
  processHighPriorityTick,
  {
    filter: MarketDataFilters.byPriority(['critical'])
      .and(MarketDataFilters.bySymbol(['BTC-USD'])),
    priority: 100
  }
)
```

## Monitoring and Debugging

### Event Metrics

Access event statistics through enhanced data feeds:

```typescript
const stats = dataFeed.getEnhancedStats()
console.log('Events processed:', stats.events.ticksReceived)
console.log('Average latency:', stats.events.avgLatency)
console.log('Events filtered:', stats.events.eventsFiltered)
```

### Debug Mode

Enable debug logging for troubleshooting:

```typescript
const config: EnhancedDataFeedConfig = {
  feedType: 'coinbase',
  symbol: 'BTC-USD',
  debug: true,  // Enable debug logging
  enhancedEvents: true
}
```

### Filter Analytics

Monitor filter performance:

```typescript
// Check global filter metrics
const metrics = enhancedEventBus.getGlobalMetrics()
console.log('Filters applied:', metrics.filtersApplied)
console.log('Events filtered:', metrics.eventsFiltered)
```

## Migration Guide

### From Basic EventBus

1. **Update Imports**
   ```typescript
   // Old
   import { EventBus } from './events/event-bus'
   
   // New  
   import { enhancedEventBus } from './events/enhanced-event-bus'
   ```

2. **Update Event Types**
   ```typescript
   // Old
   eventBus.subscribe('market.tick', handler)
   
   // New (backward compatible)
   enhancedEventBus.subscribe('market.tick', handler)
   
   // New (enhanced)
   enhancedEventBus.subscribe('market.tick.enhanced', handler)
   ```

3. **Update Data Feed Configuration**
   ```typescript
   // Old
   const config: DataFeedConfig = { feedType: 'coinbase', symbol: 'BTC-USD' }
   
   // New
   const config: EnhancedDataFeedConfig = {
     feedType: 'coinbase',
     symbol: 'BTC-USD',
     enhancedEvents: true,
     maxEventsPerSecond: 100
   }
   ```

### Backward Compatibility

The enhanced system maintains full backward compatibility:

- All existing event types continue to work
- Basic EventBus methods remain available
- No breaking changes to existing APIs
- Standard events are emitted alongside enhanced events

## Security Considerations

1. **Event Validation** - All events validated before processing
2. **Rate Limiting** - Prevents DoS through event flooding
3. **Memory Limits** - Bounded memory usage prevents exhaustion
4. **Error Isolation** - Errors in one handler don't affect others

## Testing

### Unit Testing

Test event filtering:
```typescript
test('should filter by symbol', () => {
  const filter = MarketDataFilters.bySymbol(['BTC-USD'])
  const event = { symbol: 'BTC-USD', price: 50000 }
  assert(filter(event) === true)
})
```

### Integration Testing

Test complete event flow:
```typescript
test('should process enhanced tick events', async () => {
  const events: EnhancedTickEvent[] = []
  
  enhancedEventBus.subscribe('market.tick.enhanced', (event) => {
    events.push(event)
  })
  
  await dataFeed.start()
  await dataFeed.subscribe(['BTC-USD'])
  
  // Wait for events...
  assert(events.length > 0)
  assert(events[0].type === 'market.tick.enhanced')
})
```

This documentation provides comprehensive guidance for consumers of the enhanced event system, covering all aspects from basic usage to advanced filtering and performance optimization.