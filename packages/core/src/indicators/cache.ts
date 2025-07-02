import type { CacheEntry, IIndicatorCache } from './interfaces'

/**
 * LRU cache implementation for indicator results
 */
export class IndicatorCache implements IIndicatorCache {
  private readonly cache = new Map<string, CacheEntry<any>>()
  private readonly maxSize: number
  private hits = 0
  private misses = 0
  private evictions = 0

  constructor(maxSize = 1000) {
    this.maxSize = maxSize
  }

  get<T>(key: string): T | undefined {
    const entry = this.cache.get(key)

    if (!entry) {
      this.misses++
      return undefined
    }

    // Move to end (most recently used)
    this.cache.delete(key)
    this.cache.set(key, entry)

    this.hits++
    return entry.value
  }

  set<T>(key: string, value: T, candleCount: number): void {
    // Remove existing entry if present
    if (this.cache.has(key)) {
      this.cache.delete(key)
    }

    // Evict oldest entry if at capacity
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value
      if (firstKey) {
        this.cache.delete(firstKey)
        this.evictions++
      }
    }

    // Add new entry
    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: Date.now(),
      candleCount,
    }

    this.cache.set(key, entry)
  }

  clear(): void {
    this.cache.clear()
    this.hits = 0
    this.misses = 0
    this.evictions = 0
  }

  invalidate(pattern?: string): void {
    if (!pattern) {
      this.clear()
      return
    }

    // Remove entries matching pattern
    const regex = new RegExp(pattern)
    const keysToDelete: string[] = []

    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        keysToDelete.push(key)
      }
    }

    for (const key of keysToDelete) {
      this.cache.delete(key)
    }
  }

  getSize(): number {
    return this.cache.size
  }

  getStats() {
    return {
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      size: this.cache.size,
    }
  }
}

/**
 * Create a cache key for indicator calculations
 */
export function createCacheKey(
  indicatorName: string,
  params: Record<string, any>,
  candleHash: string
): string {
  const paramStr = Object.entries(params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}:${v}`)
    .join(',')

  return `${indicatorName}:${paramStr}:${candleHash}`
}

/**
 * Create a hash of candle data for cache invalidation
 */
export function createCandleHash(
  candles: readonly { timestamp: number; close: number }[],
  lastN = 10
): string {
  if (candles.length === 0) return 'empty'

  // Use last N candles for hash
  const relevantCandles = candles.slice(-lastN)

  // Create simple hash from timestamps and closes
  let hash = 0
  for (const candle of relevantCandles) {
    const time = candle.timestamp
    const price = Math.round(candle.close * 100)
    hash = (hash << 5) - hash + time + price
    hash = hash & hash // Convert to 32-bit integer
  }

  return `${candles.length}:${hash}`
}
