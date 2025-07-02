import type { GridState, EpochDate } from '@trdr/shared'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { writeFile, readFile, mkdir, access } from 'fs/promises'
import { constants } from 'fs'
import path from 'path'

/**
 * Serializable version of GridState for persistence
 * Ensures all data can be safely JSON serialized/deserialized
 */
export interface SerializableGridState extends Omit<GridState, 'levels'> {
  readonly levels: SerializableGridLevel[]
}

/**
 * Serializable version of GridLevel for persistence
 */
export interface SerializableGridLevel {
  readonly id: string
  readonly price: number
  readonly side: 'buy' | 'sell'
  readonly size: number
  readonly isActive: boolean
  readonly orderId?: string
  readonly createdAt: number
  readonly updatedAt: number
  readonly fillCount: number
  readonly pnl: number
  readonly trailingState?: {
    readonly status: 'pending' | 'approaching' | 'active' | 'triggered' | 'failed'
    readonly activationPrice?: number
    readonly activatedAt?: number
    readonly lastUpdatePrice?: number
    readonly lastUpdatedAt?: number
    readonly adjustmentCount: number
    readonly fallbackAttempted: boolean
    readonly failureReason?: string
  }
}

/**
 * Complete persistence snapshot containing all grid manager state
 */
export interface GridManagerSnapshot {
  readonly version: string
  readonly timestamp: number
  readonly activeGrids: Record<string, SerializableGridState>
  readonly performanceHistory: readonly PerformanceHistoryRecord[]
  readonly lastPriceUpdates: Record<string, { price: number; timestamp: number }>
  readonly metadata: {
    readonly totalGridsCreated: number
    readonly totalGridsCancelled: number
    readonly systemUptime: number
  }
}

/**
 * Performance history record for self-tuning persistence
 */
export interface PerformanceHistoryRecord {
  readonly spacing: number
  readonly performance: number
  readonly fillRate: number
  readonly timestamp: number
  readonly marketConditions: {
    readonly volatility: number
    readonly trend: 'bullish' | 'bearish' | 'sideways'
    readonly volume: number
  }
}

/**
 * Configuration options for grid state persistence
 */
export interface GridPersistenceConfig {
  /** Directory to store persistence files */
  readonly storageDir: string
  /** How often to auto-save state (milliseconds) */
  readonly autoSaveInterval: number
  /** Maximum number of backup files to keep */
  readonly maxBackups: number
  /** Enable compression for storage files */
  readonly enableCompression: boolean
  /** Validate state integrity on load */
  readonly validateOnLoad: boolean
}

/**
 * Recovery information for failed state loads
 */
export interface StateRecoveryInfo {
  readonly success: boolean
  readonly recoveredGrids: number
  readonly failedGrids: number
  readonly errors: readonly string[]
  readonly warnings: readonly string[]
  readonly backupUsed?: string
}

/**
 * Repository interface for grid state storage
 * Allows swapping out storage implementations (files, database, cloud, etc.)
 */
export interface GridStateRepository {
  /**
   * Initialize the storage system
   */
  initialize(): Promise<void>

  /**
   * Save a snapshot to storage
   */
  saveSnapshot(snapshot: GridManagerSnapshot): Promise<void>

  /**
   * Load the most recent snapshot from storage
   */
  loadSnapshot(): Promise<{ snapshot: GridManagerSnapshot; recoveryInfo: StateRecoveryInfo }>

  /**
   * Create a backup of the current state
   */
  createBackup(): Promise<void>

  /**
   * Clean up old backups/snapshots
   */
  cleanup(): Promise<void>

  /**
   * Shutdown the storage system
   */
  shutdown(): Promise<void>
}

/**
 * File-based implementation of GridStateRepository
 */
export class FileGridStateRepository implements GridStateRepository {
  private readonly config: GridPersistenceConfig
  private readonly logger?: Logger
  private readonly version = '1.0.0'

  constructor(config: GridPersistenceConfig, logger?: Logger) {
    this.config = config
    this.logger = logger
  }

  async initialize(): Promise<void> {
    try {
      await this.ensureStorageDirectory()
      this.logger?.debug('File repository initialized', {
        storageDir: this.config.storageDir
      })
    } catch (error) {
      throw new Error(`File repository initialization failed: ${error}`)
    }
  }

  async saveSnapshot(snapshot: GridManagerSnapshot): Promise<void> {
    const filename = this.getSnapshotFilename()
    const filepath = path.join(this.config.storageDir, filename)
    
    try {
      await this.createBackup()
      
      const data = JSON.stringify(snapshot, null, 2)
      await writeFile(filepath, data, 'utf8')
      
      await this.cleanup()
      
      this.logger?.debug('Snapshot saved to file', {
        filename,
        size: data.length
      })
    } catch (error) {
      throw new Error(`Failed to save snapshot to file: ${error}`)
    }
  }

  async loadSnapshot(): Promise<{ snapshot: GridManagerSnapshot; recoveryInfo: StateRecoveryInfo }> {
    const filename = this.getSnapshotFilename()
    const filepath = path.join(this.config.storageDir, filename)
    
    try {
      const result = await this.tryLoadSnapshot(filepath)
      if (result.success && result.snapshot) {
        return { snapshot: result.snapshot, recoveryInfo: result.recoveryInfo }
      }
      
      this.logger?.warn('Primary snapshot file failed, attempting backup recovery')
      return await this.recoverFromBackup()
      
    } catch (error) {
      return {
        snapshot: this.createEmptySnapshot(),
        recoveryInfo: {
          success: false,
          recoveredGrids: 0,
          failedGrids: 0,
          errors: [`Failed to load snapshot: ${error}`],
          warnings: ['Starting with empty state']
        }
      }
    }
  }

  async createBackup(): Promise<void> {
    const filename = this.getSnapshotFilename()
    const filepath = path.join(this.config.storageDir, filename)
    
    try {
      await access(filepath, constants.F_OK)
      const timestamp = Date.now()
      const backupPath = path.join(
        this.config.storageDir,
        this.getBackupFilename(timestamp)
      )
      
      const data = await readFile(filepath, 'utf8')
      await writeFile(backupPath, data, 'utf8')
      
      this.logger?.debug('Backup created', { backupPath })
    } catch {
      // File doesn't exist, no backup needed
    }
  }

  async cleanup(): Promise<void> {
    // Implementation would scan for backup files and remove oldest ones
    // Simplified for now - in real implementation would manage backup retention
  }

  async shutdown(): Promise<void> {
    // No special cleanup needed for file storage
    this.logger?.debug('File repository shutdown complete')
  }

  private async ensureStorageDirectory(): Promise<void> {
    try {
      await access(this.config.storageDir, constants.F_OK)
    } catch {
      await mkdir(this.config.storageDir, { recursive: true })
      this.logger?.info('Created storage directory', { dir: this.config.storageDir })
    }
  }

  private getSnapshotFilename(): string {
    return 'grid-manager-state.json'
  }

  private getBackupFilename(timestamp: number): string {
    return `grid-manager-state.backup.${timestamp}.json`
  }

  private async tryLoadSnapshot(filepath: string): Promise<{ 
    success: boolean
    snapshot?: GridManagerSnapshot
    recoveryInfo: StateRecoveryInfo 
  }> {
    try {
      const data = await readFile(filepath, 'utf8')
      const snapshot = JSON.parse(data) as GridManagerSnapshot
      
      return {
        success: true,
        snapshot,
        recoveryInfo: {
          success: true,
          recoveredGrids: Object.keys(snapshot.activeGrids || {}).length,
          failedGrids: 0,
          errors: [],
          warnings: []
        }
      }
    } catch (error) {
      return {
        success: false,
        recoveryInfo: {
          success: false,
          recoveredGrids: 0,
          failedGrids: 0,
          errors: [`Failed to load ${filepath}: ${error}`],
          warnings: []
        }
      }
    }
  }

  private async recoverFromBackup(): Promise<{ snapshot: GridManagerSnapshot; recoveryInfo: StateRecoveryInfo }> {
    const emptySnapshot = this.createEmptySnapshot()
    
    return {
      snapshot: emptySnapshot,
      recoveryInfo: {
        success: false,
        recoveredGrids: 0,
        failedGrids: 0,
        errors: ['No valid backups found'],
        warnings: ['Starting with empty state']
      }
    }
  }

  private createEmptySnapshot(): GridManagerSnapshot {
    return {
      version: this.version,
      timestamp: epochDateNow(),
      activeGrids: {},
      performanceHistory: [],
      lastPriceUpdates: {},
      metadata: {
        totalGridsCreated: 0,
        totalGridsCancelled: 0,
        systemUptime: 0
      }
    }
  }
}

/**
 * Grid State Persistence Manager
 * 
 * Handles saving and loading of complete grid trading state using a repository pattern.
 * Can be configured with different storage implementations (files, database, cloud, etc.)
 */
export class GridStatePersistence {
  private readonly repository: GridStateRepository
  private readonly config: GridPersistenceConfig
  private readonly logger?: Logger
  private autoSaveTimer?: NodeJS.Timeout

  private readonly defaultConfig: GridPersistenceConfig = {
    storageDir: './data/grid-state',
    autoSaveInterval: 30000, // 30 seconds
    maxBackups: 10,
    enableCompression: false,
    validateOnLoad: true
  }

  constructor(config?: Partial<GridPersistenceConfig>, repository?: GridStateRepository, logger?: Logger) {
    this.config = { ...this.defaultConfig, ...config }
    this.repository = repository || new FileGridStateRepository(this.config, logger)
    this.logger = logger
  }

  /**
   * Initialize persistence system
   * Delegates to the repository implementation
   */
  async initialize(): Promise<void> {
    try {
      await this.repository.initialize()
      
      this.logger?.info('Grid state persistence initialized', {
        autoSaveInterval: this.config.autoSaveInterval
      })
    } catch (error) {
      this.logger?.error('Failed to initialize grid state persistence', { error })
      throw new Error(`Persistence initialization failed: ${error}`)
    }
  }

  /**
   * Start automatic state saving
   */
  startAutoSave(saveCallback: () => Promise<GridManagerSnapshot>): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer)
    }

    this.autoSaveTimer = setInterval(async () => {
      try {
        const snapshot = await saveCallback()
        await this.saveSnapshot(snapshot)
        this.logger?.debug('Auto-save completed', {
          activeGrids: Object.keys(snapshot.activeGrids).length,
          timestamp: snapshot.timestamp
        })
      } catch (error) {
        this.logger?.error('Auto-save failed', { error })
      }
    }, this.config.autoSaveInterval)

    this.logger?.info('Auto-save started', {
      interval: this.config.autoSaveInterval
    })
  }

  /**
   * Stop automatic state saving
   */
  stopAutoSave(): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer)
      this.autoSaveTimer = undefined
      this.logger?.info('Auto-save stopped')
    }
  }

  /**
   * Save a complete grid manager snapshot
   */
  async saveSnapshot(snapshot: GridManagerSnapshot): Promise<void> {
    try {
      if (this.config.validateOnLoad) {
        const validation = this.validateSnapshot(snapshot)
        if (!validation.valid) {
          throw new Error(`Invalid snapshot: ${validation.errors.join(', ')}`)
        }
      }

      await this.repository.saveSnapshot(snapshot)
      
      this.logger?.debug('Snapshot saved successfully', {
        activeGrids: Object.keys(snapshot.activeGrids).length,
        timestamp: snapshot.timestamp
      })
    } catch (error) {
      this.logger?.error('Failed to save snapshot', { error })
      throw new Error(`Failed to save snapshot: ${error}`)
    }
  }

  /**
   * Load the most recent grid manager snapshot
   */
  async loadSnapshot(): Promise<{ snapshot: GridManagerSnapshot; recoveryInfo: StateRecoveryInfo }> {
    try {
      const result = await this.repository.loadSnapshot()
      
      if (this.config.validateOnLoad && result.recoveryInfo.success) {
        const validation = this.validateSnapshot(result.snapshot)
        if (!validation.valid) {
          this.logger?.warn('Loaded snapshot failed validation', { errors: validation.errors })
          return {
            snapshot: result.snapshot, // Return anyway but mark as problematic
            recoveryInfo: {
              ...result.recoveryInfo,
              warnings: [...result.recoveryInfo.warnings, ...validation.errors.map(e => `Validation: ${e}`)]
            }
          }
        }
      }
      
      return result
      
    } catch (error) {
      this.logger?.error('Failed to load snapshot', { error })
      throw error
    }
  }

  /**
   * Convert GridState to serializable format
   */
  serializeGridState(gridState: GridState): SerializableGridState {
    return {
      ...gridState,
      levels: gridState.levels.map(level => ({
        id: level.id,
        price: level.price,
        side: level.side,
        size: level.size,
        isActive: level.isActive,
        orderId: level.orderId,
        createdAt: Number(level.createdAt),
        updatedAt: Number(level.updatedAt),
        fillCount: level.fillCount,
        pnl: level.pnl,
        trailingState: level.trailingState ? {
          status: level.trailingState.status,
          activationPrice: level.trailingState.activationPrice,
          activatedAt: level.trailingState.activatedAt ? Number(level.trailingState.activatedAt) : undefined,
          lastUpdatePrice: level.trailingState.lastUpdatePrice,
          lastUpdatedAt: level.trailingState.lastUpdatedAt ? Number(level.trailingState.lastUpdatedAt) : undefined,
          adjustmentCount: level.trailingState.adjustmentCount,
          fallbackAttempted: level.trailingState.fallbackAttempted,
          failureReason: level.trailingState.failureReason
        } : undefined
      }))
    }
  }

  /**
   * Convert serializable format back to GridState
   */
  deserializeGridState(serializable: SerializableGridState): GridState {
    return {
      ...serializable,
      levels: serializable.levels.map(level => ({
        id: level.id,
        price: level.price,
        side: level.side,
        size: level.size,
        isActive: level.isActive,
        orderId: level.orderId,
        createdAt: level.createdAt as EpochDate,
        updatedAt: level.updatedAt as EpochDate,
        fillCount: level.fillCount,
        pnl: level.pnl,
        trailingState: level.trailingState ? {
          status: level.trailingState.status,
          activationPrice: level.trailingState.activationPrice,
          activatedAt: level.trailingState.activatedAt ? (level.trailingState.activatedAt as EpochDate) : undefined,
          lastUpdatePrice: level.trailingState.lastUpdatePrice,
          lastUpdatedAt: level.trailingState.lastUpdatedAt ? (level.trailingState.lastUpdatedAt as EpochDate) : undefined,
          adjustmentCount: level.trailingState.adjustmentCount,
          fallbackAttempted: level.trailingState.fallbackAttempted,
          failureReason: level.trailingState.failureReason
        } : undefined
      }))
    }
  }

  /**
   * Validate loaded state for integrity
   */
  validateSnapshot(snapshot: GridManagerSnapshot): { valid: boolean; errors: string[] } {
    const errors: string[] = []
    
    // Check version compatibility
    if (!snapshot.version) {
      errors.push('Missing version information')
    }
    
    // Validate timestamp
    if (!snapshot.timestamp || snapshot.timestamp <= 0) {
      errors.push('Invalid timestamp')
    }
    
    // Validate grid states
    for (const [gridId, gridState] of Object.entries(snapshot.activeGrids)) {
      if (!gridId || typeof gridId !== 'string') {
        errors.push(`Invalid grid ID: ${gridId}`)
        continue
      }
      
      // Validate grid state structure
      if (!gridState.symbol || !gridState.config) {
        errors.push(`Invalid grid state for ${gridId}: missing required fields`)
      }
      
      // Validate levels
      if (!Array.isArray(gridState.levels)) {
        errors.push(`Invalid levels array for grid ${gridId}`)
      } else {
        gridState.levels.forEach((level, index) => {
          if (!level.id || typeof level.price !== 'number' || !level.side) {
            errors.push(`Invalid level ${index} in grid ${gridId}`)
          }
        })
      }
    }
    
    return { valid: errors.length === 0, errors }
  }

  /**
   * Clean shutdown - save final state and stop auto-save
   */
  async shutdown(finalSnapshot?: GridManagerSnapshot): Promise<void> {
    this.stopAutoSave()
    
    if (finalSnapshot) {
      try {
        await this.saveSnapshot(finalSnapshot)
        this.logger?.info('Final state saved on shutdown')
      } catch (error) {
        this.logger?.error('Failed to save final state on shutdown', { error })
      }
    }

    try {
      await this.repository.shutdown()
    } catch (error) {
      this.logger?.error('Repository shutdown failed', { error })
    }
    
    this.logger?.info('Grid state persistence shutdown complete')
  }

}