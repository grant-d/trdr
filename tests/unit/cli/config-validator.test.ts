import { ok, strictEqual } from 'node:assert'
import { beforeEach, describe, it, afterEach } from 'node:test'
import { createDefaultPipelineConfig } from '../../../src/cli/config-loader'
import { ConfigValidator, validatePipelineConfig } from '../../../src/cli/config-validator'
import type { PipelineConfig, TransformConfig } from '../../../src/interfaces'
import { forceCleanupAsyncHandles } from '../../helpers/test-cleanup'
import type { LogReturnsParams, ZScoreParams, PriceCalcParams } from '../../../src/transforms'

describe('Config Validator', () => {
  afterEach(() => {
    forceCleanupAsyncHandles()
  })
  describe('validatePipelineConfig function', () => {
    it('should validate a default configuration successfully', () => {
      const config = createDefaultPipelineConfig()
      const result = validatePipelineConfig(config)

      strictEqual(result.isValid, true)
      strictEqual(result.errors.length, 1) // Should have 1 warning about no enabled transforms
      strictEqual(result.errorMessages.length, 0)
      strictEqual(result.warningMessages.length, 1)
      ok(result.warningMessages[0]?.includes('No transformations are enabled'))
    })

    it('should validate a complete valid configuration', () => {
      const logReturnsTx: TransformConfig<LogReturnsParams> = {
        type: 'logReturns',
        disabled: false,
        params: { 
          outputField: 'returns'
        }
      }

      const zScoreTx: TransformConfig<ZScoreParams> = {
        type: 'zScore',
        disabled: false,
        params: { 
          fields: ['returns'],
          windowSize: 20
        }
      }

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
          path: './data/output.jsonl',
          format: 'jsonl',
          overwrite: true
        },
        transformations: [
          logReturnsTx,
          zScoreTx
        ],
        options: {
          chunkSize: 2000,
          continueOnError: true,
          maxErrors: 10,
          showProgress: true
        },
        metadata: {
          name: 'Test Pipeline',
          version: '1.0.0',
          description: 'A test pipeline',
          author: 'Test Suite',
          created: '2024-01-01T00:00:00Z'
        }
      }

      const result = validatePipelineConfig(config)
      strictEqual(result.isValid, true)
      strictEqual(result.errorMessages.length, 0)
    })
  })

  describe('ConfigValidator class', () => {
    let validator: ConfigValidator

    beforeEach(() => {
      validator = new ConfigValidator()
    })

    describe('input validation', () => {
      it('should reject missing input path', () => {
        const config = createDefaultPipelineConfig()
        delete (config.input as any).path

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Input path is required')))
      })

      it('should reject empty input path', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.path = '   '

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Input path cannot be empty')))
      })

      it('should reject invalid input format', () => {
        const config = createDefaultPipelineConfig()
        ;(config.input as any).format = 'xml'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Input format must be')))
      })

      it('should reject invalid chunk size', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.chunkSize = -1

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Chunk size must be a positive integer')))
      })

      it('should warn about very large chunk size', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.chunkSize = 200000

        const result = validator.validate(config)
        strictEqual(result.isValid, true) // Warning, not error
        ok(result.warningMessages.some(msg => msg.includes('Very large chunk size')))
      })

      it('should validate column mapping', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.columnMapping = {
          timestamp: 'ts',
          open: 'o',
          high: '', // Invalid empty string
          low: 'l',
          close: 'c',
          volume: 'v'
        }

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Column mapping for "high"')))
      })

      it('should detect duplicate column mappings', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.columnMapping = {
          timestamp: 'price',
          open: 'price', // Duplicate mapping
          high: 'h',
          low: 'l',
          close: 'c',
          volume: 'v'
        }

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Duplicate column mappings')))
      })

      it('should validate CSV delimiter', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        fileInput.format = 'csv'
        fileInput.delimiter = 'too-long'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('CSV delimiter must be a single character')))
      })
    })

    describe('output validation', () => {
      it('should reject missing output path', () => {
        const config = createDefaultPipelineConfig()
        delete (config.output as any).path

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Output path is required')))
      })

      it('should reject invalid output format', () => {
        const config = createDefaultPipelineConfig()
        ;(config.output as any).format = 'xml'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Output format must be')))
      })

      it('should reject same input and output paths', () => {
        const config = createDefaultPipelineConfig()
        const fileInput = config.input as any
        config.output.path = fileInput.path

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Output path cannot be the same as input path')))
      })
    })

    describe('transformation validation', () => {
      it('should reject invalid transformations array', () => {
        const config = createDefaultPipelineConfig()
        ;(config as any).transformations = 'not-an-array'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Transformations must be an array')))
      })

      it('should reject invalid transform object', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = ['not-an-object' as any]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Transform must be an object')))
      })

      it('should reject missing transform type', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            disabled: false,
            params: {}
          } as any
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Transform type is required')))
      })

      it('should reject unsupported transform type', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'unsupported',
            disabled: false,
            params: {}
          } as any
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Unsupported transform type')))
      })

      it('should allow missing disabled flag (defaults to false)', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'logReturns',
            params: {}
          } as any
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, true)
        strictEqual(result.errors.length, 0)
      })

      it('should reject missing params', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'logReturns',
            disabled: false
          } as any
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Transform params are required')))
      })
    })

    describe('transform-specific parameter validation', () => {
      it('should validate zScore transform parameters', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'zScore',
            disabled: false,
            params: { 
              windowSize: -5 // Invalid window size should fail
            } as any
          }
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Window size must be a positive integer')))
      })

      it('should validate movingAverage transform parameters', () => {
        const config = createDefaultPipelineConfig()
        
        const invalidMovingAverage: TransformConfig = {
          type: 'movingAverage',
          disabled: false,
          params: {
            // @ts-ignore - Intentionally invalid
            field: '',
            windowSize: 'invalid', // Invalid window size
            type: 'invalid' // Invalid type
          }
        }
        
        config.transformations = [invalidMovingAverage]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Field parameter is required')))
        ok(result.errorMessages.some(msg => msg.includes('Window size is required and must be a positive integer')))
        ok(result.errorMessages.some(msg => msg.includes('Moving average type must be')))
      })

      it('should validate priceCalc transform parameters', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'priceCalc',
            disabled: false,
            params: { 
              calculation: 'custom'
              // Missing customFormula
            } as any
          }
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Custom formula is required')))
      })

      it('should validate RSI transform parameters', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'rsi',
            disabled: false,
            params: { 
              period: -1 // Invalid period
            } as any
          }
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('RSI period must be a positive integer')))
      })
    })

    describe('options validation', () => {
      it('should reject invalid options type', () => {
        const config = createDefaultPipelineConfig()
        ;(config as any).options = 'not-an-object'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Options must be an object')))
      })

      it('should validate options fields', () => {
        const config = createDefaultPipelineConfig()
        config.options = {
          chunkSize: -1,
          continueOnError: 'not-boolean',
          maxErrors: 0,
          showProgress: 'not-boolean'
        } as any

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Chunk size must be a positive integer')))
        ok(result.errorMessages.some(msg => msg.includes('continueOnError must be a boolean')))
        ok(result.errorMessages.some(msg => msg.includes('Max errors must be a positive integer')))
        ok(result.errorMessages.some(msg => msg.includes('showProgress must be a boolean')))
      })
    })

    describe('metadata validation', () => {
      it('should reject invalid metadata type', () => {
        const config = createDefaultPipelineConfig()
        ;(config as any).metadata = 'not-an-object'

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('Metadata must be an object')))
      })

      it('should validate metadata string fields', () => {
        const config = createDefaultPipelineConfig()
        config.metadata = {
          name: 123,
          version: true,
          description: [],
          author: {}
        } as any

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('name must be a string')))
        ok(result.errorMessages.some(msg => msg.includes('version must be a string')))
        ok(result.errorMessages.some(msg => msg.includes('description must be a string')))
        ok(result.errorMessages.some(msg => msg.includes('author must be a string')))
      })

      it('should validate date fields', () => {
        const config = createDefaultPipelineConfig()
        config.metadata = {
          created: 'invalid-date',
          modified: 'also-invalid'
        }

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('created must be a valid ISO date string')))
        ok(result.errorMessages.some(msg => msg.includes('modified must be a valid ISO date string')))
      })
    })

    describe('transform chain validation', () => {
      it('should validate transform chain compatibility', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'zScore',
            disabled: false,
            params: { 
              fields: ['non_existent_field'] // This field doesn't exist
            } as any
          }
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('requires field "non_existent_field" which is not available')))
      })

      it('should track fields through transform chain', () => {
        const config = createDefaultPipelineConfig()
        
        const logReturnsTx: TransformConfig<LogReturnsParams> = {
          type: 'logReturns',
          disabled: false,
          params: { 
            outputField: 'returns'
          }
        }

        const zScoreTx: TransformConfig<ZScoreParams> = {
          type: 'zScore',
          disabled: false,
          params: { 
            fields: ['returns'] // This should work - returns is created by logReturns
          }
        }

        config.transformations = [
          logReturnsTx,
          zScoreTx
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, true)
        strictEqual(result.errorMessages.length, 0)
      })

      it('should reject incompatible transform chain', () => {
        const config = createDefaultPipelineConfig()
        config.transformations = [
          {
            type: 'movingAverage',
            disabled: false,
            params: { 
              field: 'returns', // returns field doesn't exist yet
              windowSize: 20
            } as any
          },
          {
            type: 'logReturns',
            disabled: false,
            params: { 
              outputField: 'returns'
            } as any
          }
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('requires field "returns" which is not available')))
      })

      it('should handle disabled transforms in chain', () => {
        const config = createDefaultPipelineConfig()
        
        const logReturnsTx: TransformConfig<LogReturnsParams> = {
          type: 'logReturns',
          disabled: true, // Disabled - won't provide returns field
          params: { 
            outputField: 'returns'
          }
        }

        const zScoreTx: TransformConfig<ZScoreParams> = {
          type: 'zScore',
          disabled: false,
          params: { 
            fields: ['returns'] // This should fail - returns not available
          }
        }

        config.transformations = [
          logReturnsTx,
          zScoreTx
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, false)
        ok(result.errorMessages.some(msg => msg.includes('requires field "returns" which is not available')))
      })

      it('should validate complex transform chains', () => {
        const config = createDefaultPipelineConfig()
        
        const priceCalcTx: TransformConfig<PriceCalcParams> = {
          type: 'priceCalc',
          disabled: false,
          params: { 
            calculation: 'hlc3'
          }
        }

        // movingAverage doesn't seem to exist in the codebase, using any for invalid transform test
        const movingAverageTx = {
          type: 'movingAverage' as any,
          disabled: false,
          params: { 
            field: 'hlc3',
            windowSize: 20,
            type: 'sma'
          } as any
        } as any

        const zScoreTx: TransformConfig<ZScoreParams> = {
          type: 'zScore',
          disabled: false,
          params: { 
            fields: ['hlc3_sma']
          }
        }

        config.transformations = [
          priceCalcTx,
          movingAverageTx,
          zScoreTx
        ]

        const result = validator.validate(config)
        strictEqual(result.isValid, true)
        strictEqual(result.errorMessages.length, 0)
      })
    })

    describe('comprehensive validation', () => {
      it('should provide detailed error report for completely invalid config', () => {
        const invalidConfig = {
          input: {
            type: 'file',
            // Missing path
            format: 'invalid',
            chunkSize: -1
          },
          output: {
            // Missing path and format
          },
          transformations: [
            {
              // Missing type, enabled, params
            },
            'not-an-object'
          ],
          options: {
            chunkSize: 'not-a-number',
            continueOnError: 'not-boolean'
          }
        } as any

        const result = validator.validate(invalidConfig)
        strictEqual(result.isValid, false)
        ok(result.errors.length > 5) // Should have many errors
        
        // Check for specific errors
        ok(result.errorMessages.some(msg => msg.includes('Input path is required')))
        ok(result.errorMessages.some(msg => msg.includes('Output path is required')))
        ok(result.errorMessages.some(msg => msg.includes('Transform type is required')))
        ok(result.errorMessages.some(msg => msg.includes('Transform must be an object')))
        ok(result.errorMessages.some(msg => msg.includes('continueOnError must be a boolean')))
      })
    })
  })
})
