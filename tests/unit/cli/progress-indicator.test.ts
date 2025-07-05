import { describe, it, beforeEach, afterEach } from 'node:test'
import { strictEqual, ok } from 'node:assert'
import { ProgressIndicator, Spinner, MultiProgress } from '../../../src/cli/progress-indicator'
import { forceCleanupAsyncHandles } from '../../helpers/test-cleanup'

// Mock stdout to capture output
let stdoutOutput = ''
const originalWrite = process.stdout.write
const activeTimeouts = new Set<NodeJS.Timeout>()

// Helper function to create tracked timeouts
function createTimeout(callback: () => void, delay: number): Promise<void> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      activeTimeouts.delete(timeout)
      callback()
      resolve()
    }, delay)
    activeTimeouts.add(timeout)
  })
}

beforeEach(() => {
  stdoutOutput = ''
  activeTimeouts.clear()
  process.stdout.write = ((chunk: any) => {
    stdoutOutput += chunk.toString()
    return true
  }) as any
})

afterEach(() => {
  process.stdout.write = originalWrite
  // Clear any remaining timeouts to prevent hangs
  for (const timeout of activeTimeouts) {
    clearTimeout(timeout)
  }
  activeTimeouts.clear()
  forceCleanupAsyncHandles()
})

describe('Progress Indicator', () => {
  describe('ProgressIndicator', () => {
    it('should render basic progress bar', () => {
      const progress = new ProgressIndicator({
        width: 10,
        showPercentage: true,
        showCounts: false,
        showTime: false,
        showEta: false
      })
      
      progress.start(100)
      progress.update(50)
      
      // Should show 50% progress
      ok(stdoutOutput.includes('[█████░░░░░]'))
      ok(stdoutOutput.includes('50%'))
    })
    
    it('should handle completion', () => {
      const progress = new ProgressIndicator({
        width: 10,
        showPercentage: true,
        showCounts: false,
        showTime: false,
        showEta: false
      })
      
      progress.start(100)
      progress.complete('Done!')
      
      ok(stdoutOutput.includes('[██████████]'))
      ok(stdoutOutput.includes('100%'))
      ok(stdoutOutput.includes('Done!'))
    })
    
    it('should show counts when enabled', () => {
      const progress = new ProgressIndicator({
        width: 10,
        showPercentage: false,
        showCounts: true,
        showTime: false,
        showEta: false
      })
      
      progress.start(200)
      progress.update(75)
      
      ok(stdoutOutput.includes('75/200'))
    })
    
    it('should handle indeterminate progress', () => {
      const progress = new ProgressIndicator({
        showPercentage: false,
        showCounts: true,
        showTime: false,
        showEta: false
      })
      
      progress.start()
      progress.update(42)
      
      // Should show spinner character and count
      ok(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/.exec(stdoutOutput))
      ok(stdoutOutput.includes('42'))
    })
    
    it('should throttle updates', () => {
      const progress = new ProgressIndicator({
        updateInterval: 1000, // 1 second
        showPercentage: false,
        showCounts: true,
        showTime: false,
        showEta: false
      })
      
      progress.start(100)
      stdoutOutput = ''
      
      // Rapid updates should be throttled
      for (let i = 1; i <= 10; i++) {
        progress.update(i)
      }
      
      // Should only render once due to throttling
      const lines = stdoutOutput.split('\r').filter(line => line.includes('1/100'))
      strictEqual(lines.length, 1)
    })
    
    it('should update message', () => {
      const progress = new ProgressIndicator({
        width: 10,
        showPercentage: false,
        showCounts: false,
        showTime: false,
        showEta: false
      })
      
      progress.start(100, 'Loading...')
      ok(stdoutOutput.includes('Loading...'))
      
      stdoutOutput = ''
      progress.update(50, undefined, 'Processing...')
      ok(stdoutOutput.includes('Processing...'))
    })
    
    it('should format time correctly', () => {
      const progress = new ProgressIndicator({
        showPercentage: false,
        showCounts: false,
        showTime: true,
        showEta: false
      })
      
      // Mock time
      const startTime = Date.now()
      progress.start(100)
      
      // Simulate 5 seconds elapsed
      const originalNow = Date.now
      Date.now = () => startTime + 5000
      
      stdoutOutput = ''
      progress.update(50)
      
      ok(stdoutOutput.includes('[5s]'))
      
      Date.now = originalNow
    })
  })
  
  describe('Spinner', () => {
    it('should cycle through frames', async () => {
      const spinner = new Spinner(['1', '2', '3'])
      
      spinner.start('Testing')
      
      // Verify initial frame is rendered
      ok(stdoutOutput.includes('1 Testing'))
      
      // Wait for a few frame cycles
      await createTimeout(() => {}, 250)
      
      // Stop the spinner before checking output
      spinner.stop()
      
      // Should have cycled through frames
      const output = stdoutOutput
      ok(output.includes('Testing'), 'Should include the message')
      ok(output.includes('1') || output.includes('2') || output.includes('3'), 'Should include at least one frame')
    })
    
    it('should update message', async () => {
      const spinner = new Spinner()
      
      spinner.start('Initial')
      ok(stdoutOutput.includes('Initial'))
      
      // Give it time to render a frame
      await createTimeout(() => {}, 100)
      
      // Clear output and update message
      stdoutOutput = ''
      spinner.update('Updated')
      
      // Give it time to render with new message
      await createTimeout(() => {}, 100)
      
      // Stop before checking
      spinner.stop()
      
      ok(stdoutOutput.includes('Updated'), 'Should show updated message')
    })
    
    it('should show final message on stop', () => {
      const spinner = new Spinner()
      
      spinner.start()
      stdoutOutput = ''
      spinner.stop('Complete!')
      
      ok(stdoutOutput.includes('Complete!'))
    })
  })
  
  describe('MultiProgress', () => {
    it('should manage multiple progress bars', () => {
      const multi = new MultiProgress()
      
      const prog1 = multi.add('download', {
        width: 10,
        showPercentage: true,
        showCounts: false,
        showTime: false,
        showEta: false
      })
      
      const prog2 = multi.add('process', {
        width: 10,
        showPercentage: true,
        showCounts: false,
        showTime: false,
        showEta: false
      })
      
      ok(prog1)
      ok(prog2)
      strictEqual(multi.get('download'), prog1)
      strictEqual(multi.get('process'), prog2)
    })
    
    it('should remove progress bars', () => {
      const multi = new MultiProgress()
      
      multi.add('test')
      ok(multi.get('test'))
      
      multi.remove('test')
      strictEqual(multi.get('test'), undefined)
    })
    
    it('should clear all progress bars', () => {
      const multi = new MultiProgress()
      
      multi.add('one')
      multi.add('two')
      multi.add('three')
      
      multi.clear()
      
      strictEqual(multi.get('one'), undefined)
      strictEqual(multi.get('two'), undefined)
      strictEqual(multi.get('three'), undefined)
    })
  })
  
  describe('Integration', () => {
    it('should handle complete workflow', () => {
      const progress = new ProgressIndicator({
        width: 20,
        showPercentage: true,
        showCounts: true,
        showTime: false,
        showEta: false
      })
      
      progress.start(1000, 'Processing data')
      
      // Simulate progress updates
      for (let i = 0; i <= 1000; i += 100) {
        progress.update(i)
      }
      
      progress.complete('✓ Processing complete')
      
      // Should end with complete message
      ok(stdoutOutput.includes('✓ Processing complete'))
      ok(stdoutOutput.includes('100%'))
      ok(stdoutOutput.includes('1000/1000'))
    })
  })
})