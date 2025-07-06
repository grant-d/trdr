import { EventEmitter } from 'node:events'
import type { OhlcvDto } from '../models'
import type { Transform } from '../interfaces'
import type { Gap } from './gap-detector'
import type { BackfillRequest } from './backfill-manager'

/**
 * Metric types
 */
export enum MetricType {
  COUNTER = 'counter',
  GAUGE = 'gauge',
  HISTOGRAM = 'histogram',
  SUMMARY = 'summary'
}

/**
 * Metric definition
 */
export interface Metric {
  name: string
  type: MetricType
  description: string
  labels?: Record<string, string>
  value: number
  timestamp: number
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  /** Records per second throughput */
  throughput: number
  /** Average processing latency in ms */
  latency: number
  /** Memory usage in MB */
  memoryUsage: number
  /** CPU usage percentage */
  cpuUsage: number
  /** Error rate per minute */
  errorRate: number
}

/**
 * Transform metrics
 */
export interface TransformMetrics {
  /** Transform name */
  name: string
  /** Records processed */
  recordsProcessed: number
  /** Records output */
  recordsOutput: number
  /** Average processing time in ms */
  avgProcessingTime: number
  /** Error count */
  errors: number
  /** Last execution time */
  lastExecutionTime?: number
}

/**
 * Monitoring configuration
 */
export interface MonitoringConfig {
  /** Enable metrics collection */
  enableMetrics?: boolean
  /** Metrics collection interval in ms */
  metricsInterval?: number
  /** Enable performance profiling */
  enableProfiling?: boolean
  /** Alert thresholds */
  alerts?: {
    maxLatency?: number
    maxMemoryUsage?: number
    maxErrorRate?: number
    minThroughput?: number
  }
}

/**
 * Monitoring events
 */
export interface MonitoringEvents {
  metrics: [metrics: Metric[]]
  performance: [metrics: PerformanceMetrics]
  alert: [alert: Alert]
  transformMetrics: [metrics: TransformMetrics[]]
}

/**
 * Alert definition
 */
export interface Alert {
  level: 'info' | 'warning' | 'error' | 'critical'
  type: string
  message: string
  value?: number
  threshold?: number
  timestamp: number
}

/**
 * Pipeline monitoring system
 */
export class PipelineMonitor extends EventEmitter {
  private readonly config: Required<MonitoringConfig>
  private metrics: Map<string, Metric> = new Map()
  private transformMetrics: Map<string, TransformMetrics> = new Map()
  private metricsTimer?: NodeJS.Timeout
  private performanceData = {
    recordsProcessed: 0,
    startTime: Date.now(),
    errors: 0,
    lastThroughputCheck: Date.now(),
    lastRecordCount: 0
  }

  constructor(config: MonitoringConfig = {}) {
    super()
    this.config = {
      enableMetrics: config.enableMetrics ?? true,
      metricsInterval: config.metricsInterval ?? 10000, // 10 seconds
      enableProfiling: config.enableProfiling ?? false,
      alerts: {
        maxLatency: config.alerts?.maxLatency ?? 1000,
        maxMemoryUsage: config.alerts?.maxMemoryUsage ?? 1000,
        maxErrorRate: config.alerts?.maxErrorRate ?? 10,
        minThroughput: config.alerts?.minThroughput ?? 1
      }
    }
  }

  /**
   * Start monitoring
   */
  start(): void {
    if (!this.config.enableMetrics) return

    this.stop()
    this.performanceData.startTime = Date.now()
    
    this.metricsTimer = setInterval(() => {
      this.collectMetrics()
    }, this.config.metricsInterval)
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.metricsTimer) {
      clearInterval(this.metricsTimer)
      this.metricsTimer = undefined
    }
  }

  /**
   * Record a metric
   */
  recordMetric(name: string, value: number, type: MetricType = MetricType.GAUGE, labels?: Record<string, string>): void {
    const metric: Metric = {
      name,
      type,
      description: '',
      labels,
      value,
      timestamp: Date.now()
    }

    this.metrics.set(name, metric)
  }

  /**
   * Increment a counter
   */
  incrementCounter(name: string, value = 1, labels?: Record<string, string>): void {
    const existing = this.metrics.get(name)
    const newValue = existing ? existing.value + value : value
    this.recordMetric(name, newValue, MetricType.COUNTER, labels)
  }

  /**
   * Record transform execution
   */
  recordTransformExecution(transform: Transform, inputCount: number, outputCount: number, duration: number): void {
    let metrics = this.transformMetrics.get(transform.name)
    
    if (!metrics) {
      metrics = {
        name: transform.name,
        recordsProcessed: 0,
        recordsOutput: 0,
        avgProcessingTime: 0,
        errors: 0
      }
      this.transformMetrics.set(transform.name, metrics)
    }

    // Update metrics
    const totalProcessed = metrics.recordsProcessed + inputCount
    const totalTime = metrics.avgProcessingTime * metrics.recordsProcessed + duration
    
    metrics.recordsProcessed = totalProcessed
    metrics.recordsOutput += outputCount
    metrics.avgProcessingTime = totalTime / totalProcessed
    metrics.lastExecutionTime = Date.now()
  }

  /**
   * Record error
   */
  recordError(source: string, _error: Error): void {
    this.performanceData.errors++
    this.incrementCounter(`errors.${source}`)
    
    // Check error rate alert
    const errorRate = this.calculateErrorRate()
    if (this.config.alerts.maxErrorRate && errorRate > this.config.alerts.maxErrorRate) {
      this.emitAlert({
        level: 'error',
        type: 'error_rate',
        message: `Error rate exceeded threshold: ${errorRate.toFixed(2)}/min`,
        value: errorRate,
        threshold: this.config.alerts.maxErrorRate,
        timestamp: Date.now()
      })
    }
  }

  /**
   * Record processed record
   */
  recordProcessedRecord(_record: OhlcvDto): void {
    this.performanceData.recordsProcessed++
    this.incrementCounter('records.processed')
  }

  /**
   * Record gap detection
   */
  recordGap(gap: Gap): void {
    this.incrementCounter(`gaps.${gap.type}`)
    this.recordMetric(`gaps.severity.${gap.severity}`, 1, MetricType.COUNTER)
  }

  /**
   * Record backfill request
   */
  recordBackfillRequest(request: BackfillRequest): void {
    this.incrementCounter(`backfill.${request.status}`)
    
    if (request.status === 'completed' && request.recordsReceived) {
      this.incrementCounter('backfill.records', request.recordsReceived)
    }
  }

  /**
   * Get current metrics
   */
  getMetrics(): Metric[] {
    return Array.from(this.metrics.values())
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    const throughput = this.calculateThroughput()
    const memoryUsage = process.memoryUsage().heapUsed / 1024 / 1024
    const errorRate = this.calculateErrorRate()

    return {
      throughput,
      latency: this.calculateAverageLatency(),
      memoryUsage,
      cpuUsage: 0, // Would need proper CPU monitoring
      errorRate
    }
  }

  /**
   * Get transform metrics
   */
  getTransformMetrics(): TransformMetrics[] {
    return Array.from(this.transformMetrics.values())
  }

  /**
   * Collect metrics
   */
  private collectMetrics(): void {
    const performanceMetrics = this.getPerformanceMetrics()
    
    // Record performance metrics
    this.recordMetric('throughput', performanceMetrics.throughput)
    this.recordMetric('latency', performanceMetrics.latency)
    this.recordMetric('memory.usage', performanceMetrics.memoryUsage)
    this.recordMetric('error.rate', performanceMetrics.errorRate)

    // Check alerts
    this.checkAlerts(performanceMetrics)

    // Emit events
    this.emit('metrics', this.getMetrics())
    this.emit('performance', performanceMetrics)
    this.emit('transformMetrics', this.getTransformMetrics())
  }

  /**
   * Calculate throughput
   */
  private calculateThroughput(): number {
    const now = Date.now()
    const timeDiff = (now - this.performanceData.lastThroughputCheck) / 1000
    const recordsDiff = this.performanceData.recordsProcessed - this.performanceData.lastRecordCount

    this.performanceData.lastThroughputCheck = now
    this.performanceData.lastRecordCount = this.performanceData.recordsProcessed

    return timeDiff > 0 ? recordsDiff / timeDiff : 0
  }

  /**
   * Calculate average latency
   */
  private calculateAverageLatency(): number {
    // This would need to be implemented based on actual timing data
    const avgProcessingTimes = Array.from(this.transformMetrics.values())
      .map(m => m.avgProcessingTime)
      .filter(t => t > 0)

    return avgProcessingTimes.length > 0
      ? avgProcessingTimes.reduce((a, b) => a + b, 0) / avgProcessingTimes.length
      : 0
  }

  /**
   * Calculate error rate
   */
  private calculateErrorRate(): number {
    const runtime = (Date.now() - this.performanceData.startTime) / 1000 / 60 // minutes
    return runtime > 0 ? this.performanceData.errors / runtime : 0
  }

  /**
   * Check alerts
   */
  private checkAlerts(metrics: PerformanceMetrics): void {
    const { alerts } = this.config

    if (alerts.maxLatency && metrics.latency > alerts.maxLatency) {
      this.emitAlert({
        level: 'warning',
        type: 'latency',
        message: `Latency exceeded threshold: ${metrics.latency.toFixed(2)}ms`,
        value: metrics.latency,
        threshold: alerts.maxLatency,
        timestamp: Date.now()
      })
    }

    if (alerts.maxMemoryUsage && metrics.memoryUsage > alerts.maxMemoryUsage) {
      this.emitAlert({
        level: 'warning',
        type: 'memory',
        message: `Memory usage high: ${metrics.memoryUsage.toFixed(2)}MB`,
        value: metrics.memoryUsage,
        threshold: alerts.maxMemoryUsage,
        timestamp: Date.now()
      })
    }

    if (alerts.minThroughput && metrics.throughput < alerts.minThroughput && this.performanceData.recordsProcessed > 0) {
      this.emitAlert({
        level: 'warning',
        type: 'throughput',
        message: `Throughput below threshold: ${metrics.throughput.toFixed(2)} records/sec`,
        value: metrics.throughput,
        threshold: alerts.minThroughput,
        timestamp: Date.now()
      })
    }
  }

  /**
   * Emit alert
   */
  private emitAlert(alert: Alert): void {
    this.emit('alert', alert)
  }

  /**
   * Type-safe event emitter methods
   */
  emit<K extends keyof MonitoringEvents>(
    event: K,
    ...args: MonitoringEvents[K]
  ): boolean {
    return super.emit(event, ...args)
  }

  on<K extends keyof MonitoringEvents>(
    event: K,
    listener: (...args: MonitoringEvents[K]) => void
  ): this {
    return super.on(event, listener)
  }

  once<K extends keyof MonitoringEvents>(
    event: K,
    listener: (...args: MonitoringEvents[K]) => void
  ): this {
    return super.once(event, listener)
  }

  off<K extends keyof MonitoringEvents>(
    event: K,
    listener: (...args: MonitoringEvents[K]) => void
  ): this {
    return super.off(event, listener)
  }
}

/**
 * Create a monitoring dashboard
 */
export class MonitoringDashboard {
  private monitor: PipelineMonitor
  private updateInterval?: NodeJS.Timeout

  constructor(monitor: PipelineMonitor) {
    this.monitor = monitor
  }

  /**
   * Start dashboard updates
   */
  start(updateInterval = 1000): void {
    this.stop()
    
    this.updateInterval = setInterval(() => {
      this.render()
    }, updateInterval)

    // Initial render
    this.render()
  }

  /**
   * Stop dashboard updates
   */
  stop(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = undefined
    }
  }

  /**
   * Render dashboard
   */
  private render(): void {
    const performance = this.monitor.getPerformanceMetrics()
    const transforms = this.monitor.getTransformMetrics()

    // Clear console and render
    console.clear()
    console.log('=== Pipeline Monitoring Dashboard ===')
    console.log()
    
    // Performance metrics
    console.log('Performance Metrics:')
    console.log(`  Throughput: ${performance.throughput.toFixed(2)} records/sec`)
    console.log(`  Latency: ${performance.latency.toFixed(2)} ms`)
    console.log(`  Memory: ${performance.memoryUsage.toFixed(2)} MB`)
    console.log(`  Error Rate: ${performance.errorRate.toFixed(2)} errors/min`)
    console.log()

    // Transform metrics
    if (transforms.length > 0) {
      console.log('Transform Metrics:')
      for (const transform of transforms) {
        console.log(`  ${transform.name}:`)
        console.log(`    Processed: ${transform.recordsProcessed}`)
        console.log(`    Output: ${transform.recordsOutput}`)
        console.log(`    Avg Time: ${transform.avgProcessingTime.toFixed(2)} ms`)
        console.log(`    Errors: ${transform.errors}`)
      }
    }
  }
}