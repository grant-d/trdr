import type { 
  GridStateRepository, 
  GridManagerSnapshot, 
  StateRecoveryInfo 
} from './grid-state-persistence'
import { epochDateNow } from '@trdr/shared'

/**
 * In-memory mock implementation of GridStateRepository for testing
 * This avoids file system operations that would break CI/CD
 */
export class MockGridStateRepository implements GridStateRepository {
  private snapshots: GridManagerSnapshot[] = []
  private backups: GridManagerSnapshot[] = []
  private initialized = false
  
  // For testing - track method calls
  public saveSnapshotCalls = 0
  public loadSnapshotCalls = 0
  public createBackupCalls = 0
  public cleanupCalls = 0
  
  // For testing - simulate errors
  public shouldFailSave = false
  public shouldFailLoad = false
  public shouldFailInitialize = false
  
  async initialize(): Promise<void> {
    if (this.shouldFailInitialize) {
      throw new Error('Mock initialization failed')
    }
    this.initialized = true
  }
  
  async saveSnapshot(snapshot: GridManagerSnapshot): Promise<void> {
    this.saveSnapshotCalls++
    
    if (!this.initialized) {
      throw new Error('Repository not initialized')
    }
    
    if (this.shouldFailSave) {
      throw new Error('Mock save failed')
    }
    
    // Store the snapshot
    this.snapshots.push(snapshot)
    
    // Also add to backups for recovery testing
    this.backups.push(snapshot)
    
    // Keep only last 10 snapshots in memory
    if (this.snapshots.length > 10) {
      this.snapshots.shift()
    }
    
    // Keep only last 5 backups in memory
    if (this.backups.length > 5) {
      this.backups.shift()
    }
  }
  
  async loadSnapshot(): Promise<{ snapshot: GridManagerSnapshot; recoveryInfo: StateRecoveryInfo }> {
    this.loadSnapshotCalls++
    
    if (!this.initialized) {
      throw new Error('Repository not initialized')
    }
    
    if (this.shouldFailLoad) {
      // Try to recover from backup
      if (this.backups.length > 0) {
        const backup = this.backups[this.backups.length - 1]!
        return {
          snapshot: backup,
          recoveryInfo: {
            success: true,
            recoveredGrids: Object.keys(backup.activeGrids).length,
            failedGrids: 0,
            errors: ['Primary load failed, recovered from backup'],
            warnings: [],
            backupUsed: 'mock-backup'
          }
        }
      }
      
      throw new Error('Mock load failed')
    }
    
    if (this.snapshots.length === 0) {
      // Return empty snapshot
      return {
        snapshot: this.createEmptySnapshot(),
        recoveryInfo: {
          success: false,
          recoveredGrids: 0,
          failedGrids: 0,
          errors: ['No snapshots found'],
          warnings: []
        }
      }
    }
    
    const snapshot = this.snapshots[this.snapshots.length - 1]!
    return {
      snapshot,
      recoveryInfo: {
        success: true,
        recoveredGrids: Object.keys(snapshot.activeGrids).length,
        failedGrids: 0,
        errors: [],
        warnings: []
      }
    }
  }
  
  async createBackup(): Promise<void> {
    this.createBackupCalls++
    
    if (!this.initialized) {
      throw new Error('Repository not initialized')
    }
    
    if (this.snapshots.length > 0) {
      const currentSnapshot = this.snapshots[this.snapshots.length - 1]!
      this.backups.push(currentSnapshot)
      
      // Keep only last 5 backups in memory
      if (this.backups.length > 5) {
        this.backups.shift()
      }
    }
  }
  
  async cleanup(): Promise<void> {
    this.cleanupCalls++
    
    if (!this.initialized) {
      throw new Error('Repository not initialized')
    }
    
    // In mock, just limit the number of snapshots
    while (this.snapshots.length > 10) {
      this.snapshots.shift()
    }
    
    while (this.backups.length > 5) {
      this.backups.shift()
    }
  }
  
  async shutdown(): Promise<void> {
    // Mark as not initialized but keep data for test assertions
    this.initialized = false
    // Don't clear snapshots, backups, or counters - they're needed for test assertions
  }
  
  // Helper methods for testing
  
  /**
   * Reset all call counters
   */
  resetCounters(): void {
    this.saveSnapshotCalls = 0
    this.loadSnapshotCalls = 0
    this.createBackupCalls = 0
    this.cleanupCalls = 0
  }
  
  /**
   * Get the current number of stored snapshots
   */
  getSnapshotCount(): number {
    return this.snapshots.length
  }
  
  /**
   * Get the current number of stored backups
   */
  getBackupCount(): number {
    return this.backups.length
  }
  
  /**
   * Get the most recent snapshot for testing
   */
  getLatestSnapshot(): GridManagerSnapshot | undefined {
    return this.snapshots[this.snapshots.length - 1]
  }
  
  /**
   * Clear all stored data
   */
  clear(): void {
    this.snapshots = []
    this.backups = []
  }
  
  /**
   * Create an empty snapshot
   */
  private createEmptySnapshot(): GridManagerSnapshot {
    return {
      version: '1.0.0',
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