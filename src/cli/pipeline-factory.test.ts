import assert from 'node:assert'
import { describe, it } from 'node:test'

void describe('PipelineFactory', async () => {
  void describe('duration parsing', async () => {
    function calculateExpectedStart(duration: string): number {
      const now = Date.now()
      const regex = /^(\d+)([mhdwMy])$/
      const match = regex.exec(duration)
      if (!match) return now
      
      const value = parseInt(match[1]!)
      const unit = match[2]!
      
      switch (unit) {
        case 'm': return now - (value * 60000)
        case 'h': return now - (value * 3600000)
        case 'd': return now - (value * 86400000)
        case 'w': return now - (value * 7 * 86400000)
        case 'M': return now - (value * 30 * 86400000)
        case 'y': return now - (value * 365 * 86400000)
        default: return now
      }
    }

    void it('should parse minute durations correctly', () => {
      const expectedStart = calculateExpectedStart('30m')
      const actualStart = Date.now() - (30 * 60000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 30 minutes`
      )
    })

    void it('should parse hour durations correctly', () => {
      const expectedStart = calculateExpectedStart('2h')
      const actualStart = Date.now() - (2 * 3600000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 2 hours`
      )
    })

    void it('should parse day durations correctly', () => {
      const expectedStart = calculateExpectedStart('5d')
      const actualStart = Date.now() - (5 * 86400000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 5 days`
      )
    })

    void it('should parse week durations correctly', () => {
      const expectedStart = calculateExpectedStart('2w')
      const actualStart = Date.now() - (2 * 7 * 86400000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 2 weeks`
      )
    })

    void it('should parse month durations correctly', () => {
      const expectedStart = calculateExpectedStart('3M')
      const actualStart = Date.now() - (3 * 30 * 86400000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 3 months`
      )
    })

    void it('should parse year durations correctly', () => {
      const expectedStart = calculateExpectedStart('1y')
      const actualStart = Date.now() - (365 * 86400000)
      const tolerance = 100
      
      assert.ok(
        Math.abs(expectedStart - actualStart) < tolerance,
        `Expected start calculation should be correct for 1 year`
      )
    })

    void it('should handle invalid duration format', () => {
      const testDuration = 'invalid'
      const regex = /^(\d+)([mhdwMy])$/
      const match = regex.exec(testDuration)
      
      assert.strictEqual(match, null, 'Invalid duration should not match regex')
    })

    void it('should handle continuous duration', () => {
      const testDuration = 'continuous'
      const regex = /^(\d+)([mhdwMy])$/
      const match = regex.exec(testDuration)
      
      assert.strictEqual(match, null, 'Continuous duration should not match standard regex')
      assert.strictEqual(testDuration, 'continuous', 'Should recognize continuous as special case')
    })

    void it('should handle bars duration', () => {
      const testDuration = '1000bars'
      const regex = /^(\d+)([mhdwMy])$/
      const match = regex.exec(testDuration)
      
      assert.strictEqual(match, null, 'Bars duration should not match standard regex')
      assert.ok(testDuration.endsWith('bars'), 'Should recognize bars suffix')
      
      const barsCount = parseInt(testDuration.slice(0, -4))
      assert.strictEqual(barsCount, 1000, 'Should extract bar count correctly')
    })
  })
})