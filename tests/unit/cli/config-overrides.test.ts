import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import { createDefaultPipelineConfig } from '../../../src/cli/config-loader'
import type { FileInputConfig } from '../../../src/interfaces/pipeline-config.interface'
import type {
  ConfigOverrideError} from '../../../src/cli/config-overrides'
import {
  applyOverrides,
  cloneConfig,
  getConfigValue
} from '../../../src/cli/config-overrides'

describe('Config Overrides', () => {
  describe('applyOverrides', () => {
    it('should apply simple string override', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, ['input.path=/new/data.csv'])
      
      const fileInput = config.input as FileInputConfig
      strictEqual(fileInput.path, '/new/data.csv')
      // Ensure other properties weren't affected
      strictEqual(fileInput.format, 'csv')
    })

    it('should apply numeric overrides', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.chunkSize=2000',
        'options.chunkSize=5000',
        'options.maxErrors=50'
      ])
      
      const fileInput1 = config.input as FileInputConfig
      strictEqual(fileInput1.chunkSize, 2000)
      strictEqual(config.options?.chunkSize, 5000)
      strictEqual(config.options?.maxErrors, 50)
    })

    it('should apply boolean overrides', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'output.overwrite=false',
        'options.continueOnError=false',
        'options.showProgress=true'
      ])
      
      strictEqual(config.output.overwrite, false)
      strictEqual(config.options?.continueOnError, false)
      strictEqual(config.options?.showProgress, true)
    })

    it('should apply nested object overrides', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.columnMapping.timestamp=ts',
        'input.columnMapping.close=c',
        'input.exchange=coinbase'
      ])
      
      const fileInput2 = config.input as FileInputConfig
      strictEqual(fileInput2.columnMapping?.timestamp, 'ts')
      strictEqual(fileInput2.columnMapping?.close, 'c')
      strictEqual(fileInput2.exchange, 'coinbase')
    })

    it('should apply array value overrides', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.columnMapping={"timestamp":"ts","open":"o","high":"h","low":"l","close":"c","volume":"v"}'
      ])
      
      const fileInput3 = config.input as FileInputConfig
      deepStrictEqual(fileInput3.columnMapping, {
        timestamp: 'ts',
        open: 'o',
        high: 'h',
        low: 'l',
        close: 'c',
        volume: 'v'
      })
    })

    it('should apply transformation overrides', () => {
      const config = createDefaultPipelineConfig()
      config.transformations = [
        {
          type: 'logReturns',
          enabled: true,
          params: { outputField: 'returns' } as any
        },
        {
          type: 'zScore',
          enabled: false,
          params: { fields: ['returns'] } as any
        }
      ]
      
      applyOverrides(config, [
        'transformations.0.enabled=false',
        'transformations.1.enabled=true',
        'transformations.1.params.windowSize=20'
      ])
      
      strictEqual(config.transformations[0]?.enabled, false)
      strictEqual(config.transformations[1]?.enabled, true)
      strictEqual((config.transformations[1]?.params as any).windowSize, 20)
    })

    it('should handle null and undefined values', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.symbol=null',
        'input.exchange=undefined'
      ])
      
      const fileInput4 = config.input as FileInputConfig
      strictEqual(fileInput4.symbol, null)
      strictEqual(fileInput4.exchange, undefined)
    })

    it('should handle scientific notation numbers', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'options.chunkSize=1e3',
        'options.maxErrors=5.5e1'
      ])
      
      strictEqual(config.options?.chunkSize, 1000)
      strictEqual(config.options?.maxErrors, 55)
    })

    it('should handle floating point numbers', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.chunkSize=1000.5'
      ])
      
      const fileInput5 = config.input as FileInputConfig
      strictEqual(fileInput5.chunkSize, 1000.5)
    })

    it('should create intermediate objects when needed', () => {
      const config = createDefaultPipelineConfig()
      delete (config as any).metadata
      
      applyOverrides(config, [
        'metadata.name=New Pipeline',
        'metadata.version=2.0.0'
      ])
      
      strictEqual(config.metadata?.name, 'New Pipeline')
      strictEqual(config.metadata?.version, '2.0.0')
    })

    it('should handle empty override array', () => {
      const config = createDefaultPipelineConfig()
      const originalConfig = cloneConfig(config)
      
      applyOverrides(config, [])
      
      deepStrictEqual(config, originalConfig)
    })

    it('should skip empty override strings', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, ['', '  ', 'input.path=/new/path', '\t'])
      
      const fileInput = config.input as FileInputConfig
      strictEqual(fileInput.path, '/new/path')
    })

    it('should handle multiple overrides for same property', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.path=/first/path',
        'input.path=/second/path',
        'input.path=/final/path'
      ])
      
      const fileInput6 = config.input as FileInputConfig
      strictEqual(fileInput6.path, '/final/path')
    })
  })

  describe('error handling', () => {
    it('should throw error for invalid syntax - missing equals', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['input.path'])
      }, (error: ConfigOverrideError) => {
        strictEqual(error.name, 'ConfigOverrideError')
        ok(error.message.includes('Invalid override syntax'))
        ok(error.message.includes('Expected format: "path.to.property=value"'))
        return true
      })
    })

    it('should throw error for empty property path', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Property path cannot be empty'))
        return true
      })
    })

    it('should throw error for invalid root property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['invalid.property=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid root property "invalid"'))
        return true
      })
    })

    it('should throw error for invalid input property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['input.invalidProperty=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid input property "invalidProperty"'))
        return true
      })
    })

    it('should throw error for invalid output property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['output.invalidProperty=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid output property "invalidProperty"'))
        return true
      })
    })

    it('should throw error for invalid transformation index', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['transformations.notAnIndex.enabled=true'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid transformation index "notAnIndex"'))
        return true
      })
    })

    it('should throw error for invalid transformation property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['transformations.0.invalid=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid transformation property "invalid"'))
        return true
      })
    })

    it('should throw error for invalid options property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['options.invalidOption=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid options property "invalidOption"'))
        return true
      })
    })

    it('should throw error for invalid metadata property', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['metadata.invalidField=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid metadata property "invalidField"'))
        return true
      })
    })

    it('should throw error for invalid column mapping field', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['input.columnMapping.invalidField=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid column mapping field "invalidField"'))
        return true
      })
    })

    it('should throw error when trying to override non-object intermediate value', () => {
      const config = createDefaultPipelineConfig()
      const fileInput7 = config.input as FileInputConfig
      fileInput7.path = 'string-value'
      
      throws(() => {
        applyOverrides(config, ['input.path.nested=value'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('intermediate value is not an object'))
        return true
      })
    })

    it('should throw error for invalid JSON in array value', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['input.columnMapping=[invalid json]'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid array value'))
        return true
      })
    })

    it('should throw error for invalid JSON in object value', () => {
      const config = createDefaultPipelineConfig()
      
      throws(() => {
        applyOverrides(config, ['input.columnMapping={invalid: json}'])
      }, (error: ConfigOverrideError) => {
        ok(error.message.includes('Invalid object value'))
        return true
      })
    })
  })

  describe('type conversion', () => {
    it('should convert string numbers to integers', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.chunkSize=2000',
        'options.maxErrors=50'
      ])
      
      const fileInput8 = config.input as FileInputConfig
      strictEqual(typeof fileInput8.chunkSize, 'number')
      strictEqual(fileInput8.chunkSize, 2000)
      strictEqual(typeof config.options?.maxErrors, 'number')
      strictEqual(config.options?.maxErrors, 50)
    })

    it('should convert string floats to numbers', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, ['input.chunkSize=1000.5'])
      
      const fileInput9 = config.input as FileInputConfig
      strictEqual(typeof fileInput9.chunkSize, 'number')
      strictEqual(fileInput9.chunkSize, 1000.5)
    })

    it('should convert boolean strings', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'output.overwrite=true',
        'options.continueOnError=false'
      ])
      
      strictEqual(typeof config.output.overwrite, 'boolean')
      strictEqual(config.output.overwrite, true)
      strictEqual(typeof config.options?.continueOnError, 'boolean')
      strictEqual(config.options?.continueOnError, false)
    })

    it('should handle case-insensitive boolean values', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'output.overwrite=TRUE',
        'options.continueOnError=False'
      ])
      
      strictEqual(config.output.overwrite, true)
      strictEqual(config.options?.continueOnError, false)
    })

    it('should keep strings as strings when appropriate', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.path=/path/to/file.csv',
        'metadata.name=My Pipeline'
      ])
      
      const fileInput10 = config.input as FileInputConfig
      strictEqual(typeof fileInput10.path, 'string')
      strictEqual(fileInput10.path, '/path/to/file.csv')
      strictEqual(typeof config.metadata?.name, 'string')
      strictEqual(config.metadata?.name, 'My Pipeline')
    })

    it('should handle complex JSON objects', () => {
      const config = createDefaultPipelineConfig()
      
      applyOverrides(config, [
        'input.columnMapping={"timestamp":"ts","open":"o","high":"h","low":"l","close":"c","volume":"v","symbol":"sym"}'
      ])
      
      const fileInput11 = config.input as FileInputConfig
      strictEqual(typeof fileInput11.columnMapping, 'object')
      deepStrictEqual(fileInput11.columnMapping, {
        timestamp: 'ts',
        open: 'o',
        high: 'h',
        low: 'l',
        close: 'c',
        volume: 'v',
        symbol: 'sym'
      })
    })
  })

  describe('cloneConfig', () => {
    it('should create a deep copy of configuration', () => {
      const original = createDefaultPipelineConfig()
      const clone = cloneConfig(original)
      
      // Modify clone
      const cloneFileInput = clone.input as FileInputConfig
      cloneFileInput.path = '/modified/path'
      clone.options.chunkSize = 9999
      
      // Original should be unchanged
      const originalFileInput = original.input as FileInputConfig
      strictEqual(originalFileInput.path, './data/input.csv')
      strictEqual(original.options?.chunkSize, 1000)
      
      // Clone should have modifications
      strictEqual(cloneFileInput.path, '/modified/path')
      strictEqual(clone.options?.chunkSize, 9999)
    })
  })

  describe('getConfigValue', () => {
    it('should retrieve nested configuration values', () => {
      const config = createDefaultPipelineConfig()
      const testInputConfig = config.input as FileInputConfig
      testInputConfig.columnMapping = {
        timestamp: 'ts',
        open: 'o',
        high: 'h',
        low: 'l',
        close: 'c',
        volume: 'v'
      }
      
      strictEqual(getConfigValue(config, 'input.path'), './data/input.csv')
      strictEqual(getConfigValue(config, 'input.format'), 'csv')
      strictEqual(getConfigValue(config, 'input.columnMapping.timestamp'), 'ts')
      strictEqual(getConfigValue(config, 'options.chunkSize'), 1000)
      strictEqual(getConfigValue(config, 'metadata.name'), 'Default Pipeline')
    })

    it('should return undefined for non-existent paths', () => {
      const config = createDefaultPipelineConfig()
      
      strictEqual(getConfigValue(config, 'nonexistent'), undefined)
      strictEqual(getConfigValue(config, 'input.nonexistent'), undefined)
      strictEqual(getConfigValue(config, 'input.columnMapping.nonexistent'), undefined)
    })

    it('should handle null and undefined intermediate values', () => {
      const config = createDefaultPipelineConfig()
      ;(config as any).nullProperty = null
      ;(config as any).undefinedProperty = undefined
      
      strictEqual(getConfigValue(config, 'nullProperty.nested'), undefined)
      strictEqual(getConfigValue(config, 'undefinedProperty.nested'), undefined)
    })
  })

  describe('integration tests', () => {
    it('should apply comprehensive overrides correctly', () => {
      const config = createDefaultPipelineConfig()
      config.transformations = [
        {
          type: 'logReturns',
          enabled: true,
          params: { outputField: 'returns' } as any
        }
      ]
      
      const overrides = [
        'input.path=/new/data.csv',
        'input.format=jsonl',
        'input.chunkSize=5000',
        'input.columnMapping.timestamp=time',
        'output.path=/new/output.sqlite',
        'output.format=sqlite',
        'output.overwrite=false',
        'transformations.0.enabled=false',
        'transformations.0.params.priceField=close',
        'options.chunkSize=10000',
        'options.continueOnError=false',
        'options.maxErrors=25',
        'options.showProgress=true',
        'metadata.name=Production Pipeline',
        'metadata.version=2.1.0'
      ]
      
      applyOverrides(config, overrides)
      
      // Verify all overrides were applied
      const fileInput12 = config.input as FileInputConfig
      strictEqual(fileInput12.path, '/new/data.csv')
      strictEqual(fileInput12.format, 'jsonl')
      strictEqual(fileInput12.chunkSize, 5000)
      strictEqual(fileInput12.columnMapping?.timestamp, 'time')
      strictEqual(config.output.path, '/new/output.sqlite')
      strictEqual(config.output.format, 'sqlite')
      strictEqual(config.output.overwrite, false)
      strictEqual(config.transformations[0]?.enabled, false)
      strictEqual((config.transformations[0]?.params as any).priceField, 'close')
      strictEqual(config.options?.chunkSize, 10000)
      strictEqual(config.options?.continueOnError, false)
      strictEqual(config.options?.maxErrors, 25)
      strictEqual(config.options?.showProgress, true)
      strictEqual(config.metadata?.name, 'Production Pipeline')
      strictEqual(config.metadata?.version, '2.1.0')
    })
  })
})