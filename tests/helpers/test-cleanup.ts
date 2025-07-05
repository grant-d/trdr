/**
 * Test cleanup utilities to prevent hanging tests
 * Ensures all async handles are properly cleaned up
 */

// Track active timeouts and intervals for cleanup
const activeTimeouts = new Set<NodeJS.Timeout>()
const activeIntervals = new Set<NodeJS.Timeout>()

// Override setTimeout to track handles
const originalSetTimeout = global.setTimeout
global.setTimeout = function(callback: (...args: any[]) => void, ms?: number, ...args: any[]): NodeJS.Timeout {
  const handle = originalSetTimeout(callback, ms, ...args)
  activeTimeouts.add(handle)
  
  return handle
} as typeof setTimeout

// Override setInterval to track handles
const originalSetInterval = global.setInterval
global.setInterval = function(callback: (...args: any[]) => void, ms?: number, ...args: any[]): NodeJS.Timeout {
  const handle = originalSetInterval(callback, ms, ...args)
  activeIntervals.add(handle)
  return handle
} as typeof setInterval

// Override clearTimeout to remove from tracking
const originalClearTimeout = global.clearTimeout
global.clearTimeout = function(handle?: string | number | NodeJS.Timeout): void {
  if (handle && typeof handle !== 'string' && typeof handle !== 'number') {
    activeTimeouts.delete(handle)
  }
  return originalClearTimeout(handle)
} as typeof clearTimeout

// Override clearInterval to remove from tracking
const originalClearInterval = global.clearInterval
global.clearInterval = function(handle?: string | number | NodeJS.Timeout): void {
  if (handle && typeof handle !== 'string' && typeof handle !== 'number') {
    activeIntervals.delete(handle)
  }
  return originalClearInterval(handle)
} as typeof clearInterval

/**
 * Force cleanup of all active async handles
 * Call this in afterEach() hooks to prevent hanging tests
 */
export function forceCleanupAsyncHandles(): void {
  // Clear all tracked timeouts
  for (const handle of activeTimeouts) {
    try {
      clearTimeout(handle)
    } catch {
      // Ignore errors
    }
  }
  activeTimeouts.clear()

  // Clear all tracked intervals
  for (const handle of activeIntervals) {
    try {
      clearInterval(handle)
    } catch {
      // Ignore errors
    }
  }
  activeIntervals.clear()
}

/**
 * Get count of active async handles for debugging
 */
export function getActiveHandleCount(): { timeouts: number; intervals: number } {
  return {
    timeouts: activeTimeouts.size,
    intervals: activeIntervals.size
  }
}

/**
 * Force process exit if handles are still active after timeout
 * Use as a last resort in test suites
 */
export function forceExitIfHanging(timeoutMs = 5000): NodeJS.Timeout {
  return setTimeout(() => {
    const { timeouts, intervals } = getActiveHandleCount()
    if (timeouts > 0 || intervals > 0) {
      console.warn(`Forcing exit due to ${timeouts} timeouts and ${intervals} intervals still active`)
      process.exit(0)
    }
  }, timeoutMs)
}

/**
 * Safe process exit that cleans up first
 */
export function safeProcessExit(code = 0): void {
  forceCleanupAsyncHandles()
  
  // Give a small delay for cleanup to complete
  setTimeout(() => {
    process.exit(code)
  }, 100)
}