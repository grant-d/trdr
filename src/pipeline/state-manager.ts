import { promises as fs } from 'node:fs'
import { dirname } from 'node:path'
import type { Transform } from '../interfaces'

/**
 * State persistence configuration
 */
export interface StateConfig {
  /** Directory to store state files */
  stateDir?: string
  /** Enable automatic state saving */
  autoSave?: boolean
  /** Auto-save interval in milliseconds */
  autoSaveInterval?: number
  /** Compress state files */
  compress?: boolean
  /** State file format */
  format?: 'json' | 'binary'
}

/**
 * Pipeline state snapshot
 */
export interface PipelineState {
  /** State version for compatibility */
  version: string
  /** Timestamp of state capture */
  timestamp: number
  /** Pipeline metadata */
  metadata?: {
    name?: string
    description?: string
  }
  /** Transform states */
  transforms: TransformState[]
  /** Provider state */
  provider?: {
    type: string
    lastProcessedTime?: number
    lastProcessedId?: string
  }
  /** Repository state */
  repository?: {
    type: string
    recordCount?: number
    lastWriteTime?: number
  }
  /** Server state */
  server?: {
    startTime?: number
    recordsProcessed?: number
    errors?: number
    restarts?: number
  }
}

/**
 * Transform state
 */
export interface TransformState {
  /** Transform type */
  type: string
  /** Transform name */
  name: string
  /** Transform-specific state data */
  state: any
  /** Transform parameters */
  params?: any
}

/**
 * Stateful transform interface
 */
export interface StatefulTransform extends Transform {
  /** Get current state for persistence */
  getState(): any
  /** Restore state from persisted data */
  restoreState(state: any): void
}

/**
 * State manager for pipeline persistence
 */
export class StateManager {
  private readonly config: Required<StateConfig>
  private autoSaveTimer?: NodeJS.Timeout
  private lastSaveTime = 0

  getLastSaveTime(): number {
    return this.lastSaveTime
  }

  constructor(config: StateConfig = {}) {
    this.config = {
      stateDir: config.stateDir ?? './.trdr/state',
      autoSave: config.autoSave ?? true,
      autoSaveInterval: config.autoSaveInterval ?? 60000, // 1 minute
      compress: config.compress ?? false,
      format: config.format ?? 'json'
    }
  }

  /**
   * Start auto-save timer
   */
  startAutoSave(saveCallback: () => Promise<void>): void {
    if (!this.config.autoSave) return

    this.stopAutoSave()
    this.autoSaveTimer = setInterval(async () => {
      try {
        await saveCallback()
        this.lastSaveTime = Date.now()
      } catch (error) {
        console.error('Auto-save failed:', error)
      }
    }, this.config.autoSaveInterval)
  }

  /**
   * Stop auto-save timer
   */
  stopAutoSave(): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer)
      this.autoSaveTimer = undefined
    }
  }

  /**
   * Save pipeline state
   */
  async saveState(stateId: string, state: PipelineState): Promise<void> {
    const filePath = this.getStateFilePath(stateId)
    
    // Ensure directory exists
    await fs.mkdir(dirname(filePath), { recursive: true })

    // Serialize state
    let data: Buffer
    if (this.config.format === 'json') {
      data = Buffer.from(JSON.stringify(state, null, 2))
    } else {
      // Binary format would go here
      throw new Error('Binary format not yet implemented')
    }

    // Compress if configured
    if (this.config.compress) {
      const { gzip } = await import('node:zlib')
      const { promisify } = await import('node:util')
      const gzipAsync = promisify(gzip)
      data = await gzipAsync(data)
    }

    // Write to file
    await fs.writeFile(filePath, data)
  }

  /**
   * Load pipeline state
   */
  async loadState(stateId: string): Promise<PipelineState | null> {
    const filePath = this.getStateFilePath(stateId)

    try {
      // Read file
      let data = await fs.readFile(filePath)

      // Decompress if needed
      if (this.config.compress) {
        const { gunzip } = await import('node:zlib')
        const { promisify } = await import('node:util')
        const gunzipAsync = promisify(gunzip)
        data = await gunzipAsync(data)
      }

      // Deserialize state
      if (this.config.format === 'json') {
        return JSON.parse(data.toString())
      } else {
        throw new Error('Binary format not yet implemented')
      }
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        return null // State file doesn't exist
      }
      throw error
    }
  }

  /**
   * Delete state
   */
  async deleteState(stateId: string): Promise<void> {
    const filePath = this.getStateFilePath(stateId)
    
    try {
      await fs.unlink(filePath)
    } catch (error: any) {
      if (error.code !== 'ENOENT') {
        throw error
      }
    }
  }

  /**
   * List available states
   */
  async listStates(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.config.stateDir)
      const extension = this.config.compress ? '.json.gz' : '.json'
      
      return files
        .filter(file => file.endsWith(extension))
        .map(file => file.replace(extension, ''))
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        return []
      }
      throw error
    }
  }

  /**
   * Get state file path
   */
  private getStateFilePath(stateId: string): string {
    const extension = this.config.format === 'json' ? '.json' : '.bin'
    const fullExtension = this.config.compress ? `${extension}.gz` : extension
    return `${this.config.stateDir}/${stateId}${fullExtension}`
  }

  /**
   * Check if transform is stateful
   */
  static isStateful(transform: Transform): transform is StatefulTransform {
    return (
      typeof (transform as any).getState === 'function' &&
      typeof (transform as any).restoreState === 'function'
    )
  }

  /**
   * Extract state from transform
   */
  static extractTransformState(transform: Transform): TransformState | null {
    if (!this.isStateful(transform)) {
      return null
    }

    return {
      type: transform.type,
      name: transform.name,
      state: transform.getState(),
      params: (transform as any).params
    }
  }

  /**
   * Restore state to transform
   */
  static restoreTransformState(transform: Transform, state: TransformState): boolean {
    if (!this.isStateful(transform)) {
      return false
    }

    if (transform.type !== state.type) {
      console.warn(`Transform type mismatch: ${transform.type} vs ${state.type}`)
      return false
    }

    try {
      transform.restoreState(state.state)
      return true
    } catch (error) {
      console.error(`Failed to restore state for transform ${transform.name}:`, error)
      return false
    }
  }
}

/**
 * Create a state persistence wrapper for transforms
 */
export function createStatefulTransform<T extends Transform>(
  transform: T,
  getState: () => any,
  restoreState: (state: any) => void
): T & StatefulTransform {
  return {
    ...transform,
    getState,
    restoreState
  }
}

/**
 * State checkpoint utility
 */
export class StateCheckpoint {
  private checkpoints: Map<string, PipelineState> = new Map()
  private maxCheckpoints: number

  constructor(maxCheckpoints = 10) {
    this.maxCheckpoints = maxCheckpoints
  }

  /**
   * Create a checkpoint
   */
  create(id: string, state: PipelineState): void {
    this.checkpoints.set(id, structuredClone(state))
    
    // Remove oldest checkpoints if limit exceeded
    if (this.checkpoints.size > this.maxCheckpoints) {
      const oldest = Array.from(this.checkpoints.entries())
        .sort(([, a], [, b]) => a.timestamp - b.timestamp)
        .slice(0, this.checkpoints.size - this.maxCheckpoints)
      
      for (const [oldId] of oldest) {
        this.checkpoints.delete(oldId)
      }
    }
  }

  /**
   * Restore a checkpoint
   */
  restore(id: string): PipelineState | null {
    const state = this.checkpoints.get(id)
    return state ? structuredClone(state) : null
  }

  /**
   * List checkpoints
   */
  list(): Array<{ id: string; timestamp: number }> {
    return Array.from(this.checkpoints.entries())
      .map(([id, state]) => ({ id, timestamp: state.timestamp }))
      .sort((a, b) => b.timestamp - a.timestamp)
  }

  /**
   * Clear all checkpoints
   */
  clear(): void {
    this.checkpoints.clear()
  }
}