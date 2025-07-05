import assert from 'node:assert'
import { describe, it } from 'node:test'
import { ConfigValidator } from './config-validator'
import type { PipelineConfig } from '../interfaces/pipeline-config.interface'

describe('ConfigValidator', () => {
  describe('duration validation', () => {
    it('should accept valid minute durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '30m'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept valid hour durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '24h'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept valid day durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '7d'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept valid week durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '4w'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept valid month durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '3M'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept valid year durations', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '2y'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept continuous duration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: 'continuous'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.sqlite',
          format: 'sqlite'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should accept bars duration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '1000bars'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, true)
    })

    it('should reject invalid duration formats', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '3x'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, false)
      assert.ok(result.errorMessages[0]?.includes('Duration must be'))
    })

    it('should reject duration without number', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: 'd'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, false)
    })

    it('should reject negative duration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '-5d'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, false)
      assert.ok(result.errorMessages[0]?.includes('Duration must be'))
    })

    it('should reject NaN duration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: 'NaNd'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, false)
      assert.ok(result.errorMessages[0]?.includes('Duration must be'))
    })

    it('should reject zero duration', () => {
      const config: PipelineConfig = {
        input: {
          type: 'provider',
          provider: 'alpaca',
          symbols: ['AAPL'],
          timeframe: '1d',
          duration: '0d'
        },
        transformations: [],
        options: {
          chunkSize: 1000
        },
        output: {
          path: './output.csv',
          format: 'csv'
        }
      }
      
      const validator = new ConfigValidator()
      const result = validator.validate(config)
      assert.strictEqual(result.isValid, false)
      assert.ok(result.errorMessages[0]?.includes('positive number'))
    })
  })
})