import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'

import { createDefaultPipelineConfig } from '../../../src/cli/config-loader'
import { ConfigValidator } from '../../../src/cli/config-validator'
import type { PipelineConfig } from '../../../src/interfaces'

describe('ConfigValidator', () => {
  describe('validate', () => {
    it('should validate a default configuration successfully', () => {
      const config = createDefaultPipelineConfig()
      const validatedConfig = ConfigValidator.validate(config)

      ok(validatedConfig)
      strictEqual(validatedConfig.input.type, 'file')
    })

    it('should validate a complete valid configuration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'file',
          path: './data/input.csv',
          format: 'csv',
          chunkSize: 1000,
          columnMapping: {
            timestamp: 'ts',
            open: 'o',
            high: 'h',
            low: 'l',
            close: 'c',
            volume: 'v'
          }
        },
        output: {
          format: 'csv',
          path: './data/output.csv'
        },
        transformations: [
          {
            type: 'logReturns',
            disabled: false,
            params: {
              tx: { in: 'close', out: 'returns' }
            } as any
          },
          {
            type: 'zScore',
            disabled: false,
            params: {
              tx: { in: 'returns', out: 'returns_zscore' },
              windowSize: 20
            } as any
          }
        ],
        options: {
          chunkSize: 5000
        }
      }

      const validatedConfig = ConfigValidator.validate(config)

      ok(validatedConfig)
      strictEqual(validatedConfig.transformations.length, 2)
    })

    it('should throw on invalid input type', () => {
      const config = createDefaultPipelineConfig();
      (config.input as any).type = 'invalid'

      try {
        ConfigValidator.validate(config)
        ok(false, 'Should have thrown')
      } catch (error: any) {
        ok(error.message.includes('Invalid input type'))
      }
    })

    it('should throw on missing required fields', () => {
      const config: any = {
        output: {
          format: 'csv',
          path: './output.csv'
        },
        transformations: []
      }

      try {
        ConfigValidator.validate(config)
        ok(false, 'Should have thrown')
      } catch (error: any) {
        ok(error.message.includes('input'))
      }
    })
  })

  describe('validateWithDetails', () => {
    it('should return validation details for a valid config', () => {
      const config = createDefaultPipelineConfig()
      const result = ConfigValidator.validateWithDetails(config)

      strictEqual(result.isValid, true)
      ok(result.warnings.length >= 0)
      ok(result.config)
    })

    it('should warn about no enabled transforms', () => {
      const config = createDefaultPipelineConfig()
      config.transformations = []

      const result = ConfigValidator.validateWithDetails(config)

      strictEqual(result.isValid, true)
      ok(result.warnings.some((w) => w.includes('No transformations')))
    })

    it('should warn about disabled transforms', () => {
      const config = createDefaultPipelineConfig()
      config.transformations = [
        {
          type: 'logReturns',
          disabled: true,
          params: {
            tx: { in: 'close', out: 'returns' }
          } as any
        }
      ]

      const result = ConfigValidator.validateWithDetails(config)

      strictEqual(result.isValid, true)
      ok(result.warnings.some((w) => w.includes('disabled')))
    })

    it('should validate file input configurations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'file',
          path: './data.csv',
          format: 'csv'
        },
        output: {
          format: 'csv',
          path: './output.csv'
        },
        transformations: []
      }

      const result = ConfigValidator.validateWithDetails(config)
      strictEqual(result.isValid, true)
    })

    it('should validate provider input configurations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'coinbase',
          symbols: ['BTC-USD'],
          timeframe: '1d'
        },
        output: {
          format: 'csv',
          path: './output.csv'
        },
        transformations: []
      }

      const result = ConfigValidator.validateWithDetails(config)
      strictEqual(result.isValid, true)
    })

    it('should handle validation errors gracefully', () => {
      const config: any = {
        input: {
          type: 'invalid_type'
        },
        output: {
          format: 'csv',
          path: './output.csv'
        },
        transformations: []
      }

      const result = ConfigValidator.validateWithDetails(config)
      strictEqual(result.isValid, false)
      ok(result.warnings.length > 0)
    })
  })
})
