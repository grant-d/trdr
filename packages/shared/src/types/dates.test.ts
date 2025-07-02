import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  type EpochDate,
  epochToIso,
  fromEpochDate,
  fromIsoDate,
  type IsoDate,
  isoToEpoch,
  toEpochDate,
  toIsoDate,
} from './dates'

describe('Date utility functions', () => {
  describe('toIsoDate', () => {
    it('should convert Date object to IsoDate', () => {
      const date = new Date('2024-01-15T12:30:45.123Z')
      const result = toIsoDate(date)
      assert.equal(result, '2024-01-15T12:30:45.123Z')
    })

    it('should convert milliseconds to IsoDate', () => {
      const ms = new Date('2024-01-15T12:30:45.123Z').getTime()
      const result = toIsoDate(ms)
      assert.equal(result, '2024-01-15T12:30:45.123Z')
    })

    it('should convert seconds to IsoDate when precision is "s"', () => {
      const seconds = Math.floor(new Date('2024-01-15T12:30:45.000Z').getTime() / 1000)
      const result = toIsoDate(seconds, 's')
      assert.equal(result, '2024-01-15T12:30:45.000Z')
    })

    it('should handle current date', () => {
      const now = new Date()
      const result = toIsoDate(now)
      assert.ok(result.includes('T'))
      assert.ok(result.endsWith('Z'))
    })
  })

  describe('toEpochDate', () => {
    it('should convert Date object to EpochDate', () => {
      const date = new Date('2024-01-15T12:30:45.123Z')
      const result = toEpochDate(date)
      assert.equal(result, date.getTime())
    })

    it('should return milliseconds as EpochDate', () => {
      const ms = 1705321845123
      const result = toEpochDate(ms)
      assert.equal(result, ms)
    })

    it('should convert seconds to milliseconds when precision is "s"', () => {
      const seconds = 1705321845
      const result = toEpochDate(seconds, 's')
      assert.equal(result, seconds * 1000)
    })

    it('should handle current date', () => {
      const now = new Date()
      const result = toEpochDate(now)
      assert.equal(result, now.getTime())
    })
  })

  describe('fromIsoDate', () => {
    it('should convert IsoDate to Date object', () => {
      const isoDate = '2024-01-15T12:30:45.123Z' as IsoDate
      const result = fromIsoDate(isoDate)
      assert.equal(result.toISOString(), isoDate)
    })

    it('should handle timezone in ISO string', () => {
      const isoDate = '2024-01-15T12:30:45.123+05:00' as IsoDate
      const result = fromIsoDate(isoDate)
      assert.ok(result instanceof Date)
      assert.ok(!isNaN(result.getTime()))
    })
  })

  describe('fromEpochDate', () => {
    it('should convert EpochDate to Date object', () => {
      const epochDate = 1705321845123 as EpochDate
      const result = fromEpochDate(epochDate)
      assert.equal(result.getTime(), epochDate)
    })

    it('should handle zero epoch', () => {
      const epochDate = 0 as EpochDate
      const result = fromEpochDate(epochDate)
      assert.equal(result.toISOString(), '1970-01-01T00:00:00.000Z')
    })

    it('should handle negative epoch (before 1970)', () => {
      const epochDate = -86400000 as EpochDate // -1 day from epoch
      const result = fromEpochDate(epochDate)
      assert.equal(result.toISOString(), '1969-12-31T00:00:00.000Z')
    })
  })

  describe('isoToEpoch', () => {
    it('should convert IsoDate to EpochDate', () => {
      const isoDate = '2024-01-15T12:30:45.123Z' as IsoDate
      const result = isoToEpoch(isoDate)
      const expectedMs = new Date(isoDate).getTime()
      assert.equal(result, expectedMs)
    })

    it('should handle different ISO formats', () => {
      const isoDate = '2024-01-15T12:30:45.123+00:00' as IsoDate
      const result = isoToEpoch(isoDate)
      assert.ok(typeof result === 'number')
      assert.ok(result > 0)
    })
  })

  describe('epochToIso', () => {
    it('should convert EpochDate to IsoDate', () => {
      const epochDate = 1705321845123 as EpochDate
      const result = epochToIso(epochDate)
      assert.equal(result, '2024-01-15T12:30:45.123Z')
    })

    it('should handle zero epoch', () => {
      const epochDate = 0 as EpochDate
      const result = epochToIso(epochDate)
      assert.equal(result, '1970-01-01T00:00:00.000Z')
    })
  })

  describe('Round-trip conversions', () => {
    it('should preserve date through Date -> IsoDate -> Date conversion', () => {
      const original = new Date('2024-01-15T12:30:45.123Z')
      const isoDate = toIsoDate(original)
      const restored = fromIsoDate(isoDate)
      assert.equal(restored.getTime(), original.getTime())
    })

    it('should preserve date through Date -> EpochDate -> Date conversion', () => {
      const original = new Date('2024-01-15T12:30:45.123Z')
      const epochDate = toEpochDate(original)
      const restored = fromEpochDate(epochDate)
      assert.equal(restored.getTime(), original.getTime())
    })

    it('should preserve date through IsoDate -> EpochDate -> IsoDate conversion', () => {
      const original = '2024-01-15T12:30:45.123Z' as IsoDate
      const epochDate = isoToEpoch(original)
      const restored = epochToIso(epochDate)
      assert.equal(restored, original)
    })

    it('should preserve date through EpochDate -> IsoDate -> EpochDate conversion', () => {
      const original = 1705321845123 as EpochDate
      const isoDate = epochToIso(original)
      const restored = isoToEpoch(isoDate)
      assert.equal(restored, original)
    })
  })

  describe('Edge cases', () => {
    it('should handle very large dates', () => {
      const farFuture = new Date('9999-12-31T23:59:59.999Z')
      const isoDate = toIsoDate(farFuture)
      const epochDate = toEpochDate(farFuture)

      assert.ok(isoDate.startsWith('9999'))
      assert.ok(epochDate > 0)

      // Round trip should work
      assert.equal(fromIsoDate(isoDate).getTime(), farFuture.getTime())
      assert.equal(fromEpochDate(epochDate).getTime(), farFuture.getTime())
    })

    it('should handle very old dates', () => {
      const farPast = new Date('0001-01-01T00:00:00.000Z')
      const isoDate = toIsoDate(farPast)
      const epochDate = toEpochDate(farPast)

      assert.ok(isoDate.includes('0001'))
      assert.ok(epochDate < 0) // Before Unix epoch

      // Round trip should work
      assert.equal(fromIsoDate(isoDate).getTime(), farPast.getTime())
      assert.equal(fromEpochDate(epochDate).getTime(), farPast.getTime())
    })

    it('should handle millisecond precision', () => {
      const preciseDate = new Date('2024-01-15T12:30:45.999Z')
      const isoDate = toIsoDate(preciseDate)
      const epochDate = toEpochDate(preciseDate)

      assert.ok(isoDate.includes('.999Z'))
      assert.equal(epochDate % 1000, 999)
    })
  })
})
