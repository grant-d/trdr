import { Pipeline, type PipelineConfig } from './pipeline'
import type { DataProvider, RealtimeParams } from '../interfaces'
import type { OhlcvDto } from '../models'
import { EventEmitter } from 'node:events'

/**
 * Server mode configuration options
 */
export interface ServerModeConfig {
  /** Enable auto-restart on failure */
  autoRestart?: boolean
  /** Restart delay in milliseconds */
  restartDelay?: number
  /** Maximum number of restart attempts */
  maxRestartAttempts?: number
  /** Enable graceful shutdown on signals */
  gracefulShutdown?: boolean
  /** Health check interval in milliseconds */
  healthCheckInterval?: number
  /** Callback for health check */
  onHealthCheck?: () => Promise<boolean>
}

/**
 * Server lifecycle events
 */
export interface ServerLifecycleEvents {
  start: []
  stop: []
  error: [error: Error]
  restart: [attempt: number]
  data: [record: OhlcvDto]
  healthCheck: [healthy: boolean]
}

/**
 * Server state
 */
export enum ServerState {
  IDLE = 'idle',
  STARTING = 'starting',
  RUNNING = 'running',
  STOPPING = 'stopping',
  STOPPED = 'stopped',
  ERROR = 'error'
}

/**
 * Pipeline Server for continuous operation
 * Supports both historical backfill and realtime streaming
 */
export class PipelineServer extends EventEmitter {
  private readonly pipeline: Pipeline
  private readonly config: Required<Omit<ServerModeConfig, 'onHealthCheck'>> & Pick<ServerModeConfig, 'onHealthCheck'>
  private state: ServerState = ServerState.IDLE
  private restartAttempts = 0
  private healthCheckTimer?: NodeJS.Timeout
  private realtimeSubscription?: AsyncIterableIterator<OhlcvDto>
  private abortController?: AbortController
  private stats = {
    startTime: 0,
    recordsProcessed: 0,
    errors: 0,
    restarts: 0,
    lastHealthCheck: 0
  }

  constructor(pipelineConfig: PipelineConfig, serverConfig: ServerModeConfig = {}) {
    super()
    this.pipeline = new Pipeline(pipelineConfig)
    this.config = {
      autoRestart: serverConfig.autoRestart ?? true,
      restartDelay: serverConfig.restartDelay ?? 5000,
      maxRestartAttempts: serverConfig.maxRestartAttempts ?? 10,
      gracefulShutdown: serverConfig.gracefulShutdown ?? true,
      healthCheckInterval: serverConfig.healthCheckInterval ?? 60000,
      onHealthCheck: serverConfig.onHealthCheck
    }

    if (this.config.gracefulShutdown) {
      this.setupGracefulShutdown()
    }
  }

  /**
   * Start the server
   */
  async start(): Promise<void> {
    if (this.state !== ServerState.IDLE && this.state !== ServerState.STOPPED) {
      throw new Error(`Cannot start server in state: ${this.state}`)
    }

    this.state = ServerState.STARTING
    this.stats.startTime = Date.now()
    this.abortController = new AbortController()

    try {
      this.emit('start')

      // First run historical backfill if configured
      const provider = this.pipeline.getConfig().provider
      const historicalParams = this.pipeline.getConfig().historicalParams

      if (historicalParams) {
        console.log('Running historical backfill...')
        const result = await this.pipeline.execute()
        console.log(`Backfill complete: ${result.recordsProcessed} records processed`)
        this.stats.recordsProcessed += result.recordsProcessed
      }

      // Switch to realtime mode if provider supports it
      if (this.isRealtimeProvider(provider)) {
        console.log('Starting realtime data stream...')
        await this.startRealtimeStream(provider as DataProvider)
      }

      this.state = ServerState.RUNNING
      this.startHealthCheck()
    } catch (error) {
      this.state = ServerState.ERROR
      this.emit('error', error as Error)
      
      if (this.config.autoRestart && this.restartAttempts < this.config.maxRestartAttempts) {
        await this.scheduleRestart()
      } else {
        throw error
      }
    }
  }

  /**
   * Stop the server
   */
  async stop(): Promise<void> {
    if (this.state !== ServerState.RUNNING) {
      return
    }

    this.state = ServerState.STOPPING
    this.emit('stop')

    // Cancel health checks
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer)
      this.healthCheckTimer = undefined
    }

    // Abort any ongoing operations
    if (this.abortController) {
      this.abortController.abort()
    }

    // Stop realtime stream
    if (this.realtimeSubscription) {
      try {
        await this.realtimeSubscription.return?.()
      } catch (error) {
        console.error('Error stopping realtime stream:', error)
      }
    }

    // Disconnect provider
    const provider = this.pipeline.getConfig().provider
    if (provider.isConnected()) {
      await provider.disconnect()
    }

    // Flush repository
    await this.pipeline.getConfig().repository.flush()

    this.state = ServerState.STOPPED
  }

  /**
   * Get server state
   */
  getState(): ServerState {
    return this.state
  }

  /**
   * Get server statistics
   */
  getStats(): Readonly<typeof this.stats> {
    return { ...this.stats }
  }

  /**
   * Check if provider supports realtime data
   */
  private isRealtimeProvider(provider: any): provider is DataProvider {
    return typeof provider.getRealtimeData === 'function'
  }

  /**
   * Start realtime data stream
   */
  private async startRealtimeStream(provider: DataProvider): Promise<void> {
    if (!provider.isConnected()) {
      await provider.connect()
    }

    const realtimeParams: RealtimeParams = {
      symbols: this.pipeline.getConfig().historicalParams?.symbols || [],
      timeframe: this.pipeline.getConfig().historicalParams?.timeframe || '1m'
    }

    // Get realtime data stream
    this.realtimeSubscription = provider.subscribeRealtime(realtimeParams)

    // Process realtime data through pipeline
    const transform = this.pipeline.getConfig().transform
    const repository = this.pipeline.getConfig().repository

    try {
      let processedStream: AsyncIterableIterator<OhlcvDto>
      
      if (transform && this.realtimeSubscription) {
        const result = await transform.apply(this.realtimeSubscription)
        processedStream = result.data as AsyncIterableIterator<OhlcvDto>
      } else if (this.realtimeSubscription) {
        processedStream = this.realtimeSubscription
      } else {
        throw new Error('No realtime subscription available')
      }

      // Process realtime data
      for await (const record of processedStream) {
        if (this.abortController?.signal.aborted) {
          break
        }

        try {
          await repository.appendBatch([record])
          this.emit('data', record)
          this.stats.recordsProcessed++
        } catch (error) {
          this.stats.errors++
          this.emit('error', error as Error)
          
          if (this.stats.errors > 100) {
            throw new Error('Too many errors in realtime processing')
          }
        }
      }
    } catch (error) {
      if (!this.abortController?.signal.aborted) {
        throw error
      }
    }
  }

  /**
   * Setup graceful shutdown handlers
   */
  private setupGracefulShutdown(): void {
    const shutdown = async (signal: string) => {
      console.log(`Received ${signal}, shutting down gracefully...`)
      try {
        await this.stop()
        process.exit(0)
      } catch (error) {
        console.error('Error during shutdown:', error)
        process.exit(1)
      }
    }

    process.on('SIGINT', () => shutdown('SIGINT'))
    process.on('SIGTERM', () => shutdown('SIGTERM'))
  }

  /**
   * Start health check timer
   */
  private startHealthCheck(): void {
    if (!this.config.healthCheckInterval) {
      return
    }

    this.healthCheckTimer = setInterval(async () => {
      try {
        let healthy = true
        
        if (this.config.onHealthCheck) {
          healthy = await this.config.onHealthCheck()
        } else {
          // Default health check
          const provider = this.pipeline.getConfig().provider
          healthy = this.state === ServerState.RUNNING && provider.isConnected()
        }

        this.stats.lastHealthCheck = Date.now()
        this.emit('healthCheck', healthy)

        if (!healthy && this.config.autoRestart) {
          await this.scheduleRestart()
        }
      } catch (error) {
        console.error('Health check error:', error)
        this.emit('healthCheck', false)
      }
    }, this.config.healthCheckInterval)
  }

  /**
   * Schedule a restart
   */
  private async scheduleRestart(): Promise<void> {
    this.restartAttempts++
    this.stats.restarts++
    
    this.emit('restart', this.restartAttempts)
    console.log(`Scheduling restart attempt ${this.restartAttempts} in ${this.config.restartDelay}ms...`)

    await new Promise(resolve => setTimeout(resolve, this.config.restartDelay))

    try {
      await this.stop()
      await this.start()
      this.restartAttempts = 0 // Reset on successful restart
    } catch (error) {
      console.error('Restart failed:', error)
      
      if (this.restartAttempts >= this.config.maxRestartAttempts) {
        console.error('Max restart attempts reached, giving up')
        this.state = ServerState.ERROR
      }
    }
  }

  /**
   * Type-safe event emitter methods
   */
  emit<K extends keyof ServerLifecycleEvents>(
    event: K,
    ...args: ServerLifecycleEvents[K]
  ): boolean {
    return super.emit(event, ...args)
  }

  on<K extends keyof ServerLifecycleEvents>(
    event: K,
    listener: (...args: ServerLifecycleEvents[K]) => void
  ): this {
    return super.on(event, listener)
  }

  once<K extends keyof ServerLifecycleEvents>(
    event: K,
    listener: (...args: ServerLifecycleEvents[K]) => void
  ): this {
    return super.once(event, listener)
  }

  off<K extends keyof ServerLifecycleEvents>(
    event: K,
    listener: (...args: ServerLifecycleEvents[K]) => void
  ): this {
    return super.off(event, listener)
  }
}