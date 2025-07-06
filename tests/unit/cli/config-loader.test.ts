import { deepStrictEqual, ok, strictEqual } from 'node:assert'
import { mkdir, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, it } from 'node:test'
import {
  ConfigLoadError,
  ConfigValidationError,
  createDefaultPipelineConfig,
  loadPipelineConfig
} from '../../../src/cli/config-loader'
import type { FileInputConfig } from '../../../src/interfaces/pipeline-config.interface'

describe('Config Loader', () => {
  const testDir = join(process.cwd(), 'test-configs')
  let createdFiles: string[] = []

  beforeEach(async () => {
    // Create test directory
    await mkdir(testDir, { recursive: true })
  })

  afterEach(async () => {
    // Clean up created files
    for (const file of createdFiles) {
      try {
        await unlink(file)
      } catch (error) {
        // File might not exist
        console.warn('Failed to delete test file:', file, error)
      }
    }
    createdFiles = []
  })

  async function createTestConfig(filename: string, content: any): Promise<string> {
    const filePath = join(testDir, filename)
    await writeFile(filePath, JSON.stringify(content, null, 2))
    createdFiles.push(filePath)
    return filePath
  }

  describe('loadPipelineConfig', () => {
    it('should load valid configuration', async () => {
      const validConfig = {
        input: {
          type: 'file',
          path: './data/test.csv',
          format: 'csv',
          chunkSize: 500
        },
        output: {
          path: './output/result.jsonl',
          format: 'jsonl',
          overwrite: true
        },
        transformations: [
          {
            type: 'logReturns',
            disabled: false,
            params: {
              outputField: 'returns'
            }
          }
        ],
        options: {
          chunkSize: 1000,
          continueOnError: true,
          maxErrors: 50,
          showProgress: false
        }
      }

      const configPath = await createTestConfig('valid.json', validConfig)
      const loaded = await loadPipelineConfig(configPath)

      const fileInput = loaded.input as FileInputConfig
      strictEqual(fileInput.path, './data/test.csv')
      strictEqual(fileInput.format, 'csv')
      strictEqual(fileInput.chunkSize, 500)
      strictEqual(loaded.output.path, './output/result.jsonl')
      strictEqual(loaded.output.format, 'jsonl')
      strictEqual(loaded.output.overwrite, true)
      strictEqual(loaded.transformations.length, 1)
      strictEqual(loaded.transformations[0]!.type, 'logReturns')
      strictEqual(loaded.transformations[0]!.disabled || false, false)
      strictEqual((loaded.transformations[0]!.params as any).outputField, 'returns')
      strictEqual(loaded.options?.chunkSize, 1000)
      strictEqual(loaded.options?.continueOnError, true)
      strictEqual(loaded.options?.maxErrors, 50)
      strictEqual(loaded.options?.showProgress, false)
    })

    it('should load minimal valid configuration', async () => {
      const minimalConfig = {
        input: {
          type: 'file',
          path: './input.csv'
        },
        output: {
          path: './output.jsonl',
          format: 'jsonl'
        },
        transformations: []
      }

      const configPath = await createTestConfig('minimal.json', minimalConfig)
      const loaded = await loadPipelineConfig(configPath)

      const fileInput1 = loaded.input as FileInputConfig
      strictEqual(fileInput1.path, './input.csv')
      strictEqual(loaded.output.path, './output.jsonl')
      strictEqual(loaded.output.format, 'jsonl')
      strictEqual(loaded.transformations.length, 0)
    })

    it('should expand environment variables', async () => {
      // Set test environment variables
      process.env.TEST_INPUT_PATH = '/test/input.csv'
      process.env.TEST_OUTPUT_PATH = '/test/output.jsonl'
      process.env.TEST_CHUNK_SIZE = '2000'

      const configWithEnvVars = {
        input: {
          type: 'file',
          path: '${TEST_INPUT_PATH}',
          chunkSize: '$TEST_CHUNK_SIZE'
        },
        output: {
          path: '$TEST_OUTPUT_PATH',
          format: 'jsonl'
        },
        transformations: []
      }

      const configPath = await createTestConfig('with-env.json', configWithEnvVars)
      const loaded = await loadPipelineConfig(configPath)

      const fileInput2 = loaded.input as FileInputConfig
      strictEqual(fileInput2.path, '/test/input.csv')
      strictEqual(fileInput2.chunkSize, '2000') // Note: JSON parsing keeps this as string
      strictEqual(loaded.output.path, '/test/output.jsonl')

      // Clean up environment variables
      delete process.env.TEST_INPUT_PATH
      delete process.env.TEST_OUTPUT_PATH  
      delete process.env.TEST_CHUNK_SIZE
    })

    it('should handle relative and absolute paths', async () => {
      const config = {
        input: { type: 'file', path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: []
      }

      const configPath = await createTestConfig('paths.json', config)
      
      // Test with relative path
      const loadedRelative = await loadPipelineConfig('test-configs/paths.json')
      const fileInputRel = loadedRelative.input as FileInputConfig
      ok(fileInputRel.path)

      // Test with absolute path
      const loadedAbsolute = await loadPipelineConfig(configPath)
      const fileInputAbs = loadedAbsolute.input as FileInputConfig
      ok(fileInputAbs.path)
    })

    it('should throw ConfigLoadError for non-existent file', async () => {
      try {
        await loadPipelineConfig('./non-existent.json')
        throw new Error('Expected ConfigLoadError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigLoadError')
        ok(error.message.includes('Configuration file not found'))
      }
    })

    it('should throw ConfigLoadError for invalid JSON', async () => {
      const invalidJsonPath = join(testDir, 'invalid.json')
      await writeFile(invalidJsonPath, '{ invalid json }')
      createdFiles.push(invalidJsonPath)

      try {
        await loadPipelineConfig(invalidJsonPath)
        throw new Error('Expected ConfigLoadError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigLoadError')
        ok(error.message.includes('Failed to parse JSON'))
      }
    })

    it('should throw ConfigValidationError for missing input section', async () => {
      const invalidConfig = {
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: []
      }

      const configPath = await createTestConfig('no-input.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('Configuration must be a valid JSON object with input, output, and transformations sections'))
      }
    })

    it('should throw ConfigValidationError for missing output section', async () => {
      const invalidConfig = {
        input: { type: 'file', path: './input.csv' },
        transformations: []
      }

      const configPath = await createTestConfig('no-output.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('Configuration must be a valid JSON object with input, output, and transformations sections'))
      }
    })

    it('should throw ConfigValidationError for invalid output format', async () => {
      const invalidConfig = {
        input: { type: 'file', path: './input.csv' },
        output: { path: './output.txt', format: 'txt' },
        transformations: []
      }

      const configPath = await createTestConfig('invalid-format.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('Output format must be one of'))
      }
    })

    it('should throw ConfigValidationError for missing transformations array', async () => {
      const invalidConfig = {
        input: { type: 'file', path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' }
      }

      const configPath = await createTestConfig('no-transforms.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('Configuration must be a valid JSON object with input, output, and transformations sections'))
      }
    })

    it('should throw ConfigValidationError for invalid transformation', async () => {
      const invalidConfig = {
        input: { type: 'file', path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: [
          {
            type: 'logReturns',
            // Missing params
          }
        ]
      }

      const configPath = await createTestConfig('invalid-transform.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('must have a "params" object'))
      }
    })

    it('should throw ConfigValidationError for invalid options', async () => {
      const invalidConfig = {
        input: { type: 'file', path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: [],
        options: {
          chunkSize: -1 // Invalid chunk size
        }
      }

      const configPath = await createTestConfig('invalid-options.json', invalidConfig)

      try {
        await loadPipelineConfig(configPath)
        throw new Error('Expected ConfigValidationError but no error was thrown')
      } catch (error: any) {
        strictEqual(error.name, 'ConfigValidationError')
        ok(error.message.includes('chunkSize must be a positive integer'))
      }
    })

    it('should handle complex configuration with metadata', async () => {
      const complexConfig = {
        input: {
          type: 'file',
          path: './data/complex.csv',
          format: 'csv',
          columnMapping: {
            timestamp: 'ts',
            open: 'o',
            high: 'h',
            low: 'l',
            close: 'c',
            volume: 'v'
          },
          chunkSize: 5000,
          exchange: 'test',
          symbol: 'BTC-USD'
        },
        output: {
          path: './output/result.jsonl',
          format: 'jsonl',
          overwrite: false
        },
        transformations: [
          {
            type: 'logReturns',
            disabled: false,
            params: {
              outputField: 'returns',
              priceField: 'close'
            }
          },
          {
            type: 'zScore',
            disabled: true,
            params: {
              fields: ['returns'],
              windowSize: 20
            }
          }
        ],
        options: {
          chunkSize: 10000,
          continueOnError: false,
          maxErrors: 10,
          showProgress: true
        },
        metadata: {
          name: 'Complex Test Pipeline',
          version: '2.1.0',
          description: 'A complex pipeline for testing',
          author: 'Test Suite',
          created: '2024-01-01T00:00:00Z'
        }
      }

      const configPath = await createTestConfig('complex.json', complexConfig)
      const loaded = await loadPipelineConfig(configPath)

      // Verify complex nested structures
      const fileInput3 = loaded.input as FileInputConfig
      strictEqual(fileInput3.columnMapping?.timestamp, 'ts')
      strictEqual(fileInput3.columnMapping?.close, 'c')
      strictEqual(fileInput3.exchange, 'test')
      strictEqual(fileInput3.symbol, 'BTC-USD')
      
      strictEqual(loaded.output.format, 'jsonl')
      strictEqual(loaded.output.overwrite, false)
      
      strictEqual(loaded.transformations.length, 2)
      strictEqual(loaded.transformations[0]!.disabled, false)
      strictEqual(loaded.transformations[1]!.disabled, true)
      strictEqual((loaded.transformations[1]!.params as any).windowSize, 20)
      
      strictEqual(loaded.options?.continueOnError, false)
      
      strictEqual(loaded.metadata?.name, 'Complex Test Pipeline')
      strictEqual(loaded.metadata?.version, '2.1.0')
    })
  })

  describe('createDefaultPipelineConfig', () => {
    it('should create valid default configuration', () => {
      const defaultConfig = createDefaultPipelineConfig()

      // Check structure
      ok(defaultConfig.input)
      ok(defaultConfig.output)
      ok(Array.isArray(defaultConfig.transformations))
      ok(defaultConfig.options)
      ok(defaultConfig.metadata)

      // Check default values - we know createDefaultPipelineConfig returns file input
      const fileInput = defaultConfig.input as FileInputConfig
      strictEqual(fileInput.path, './data/input.csv')
      strictEqual(fileInput.format, 'csv')
      strictEqual(fileInput.chunkSize, 1000)
      
      strictEqual(defaultConfig.output.path, './data/output.jsonl')
      strictEqual(defaultConfig.output.format, 'jsonl')
      strictEqual(defaultConfig.output.overwrite, true)
      
      strictEqual(defaultConfig.transformations.length, 0)
      
      strictEqual(defaultConfig.options.chunkSize, 1000)
      strictEqual(defaultConfig.options.continueOnError, true)
      strictEqual(defaultConfig.options.maxErrors, 100)
      strictEqual(defaultConfig.options.showProgress, true)
      
      strictEqual(defaultConfig.metadata?.name, 'Default Pipeline')
      strictEqual(defaultConfig.metadata?.version, '1.0.0')
      ok(defaultConfig.metadata?.created)
    })

    it('should create configuration that passes validation', async () => {
      const defaultConfig = createDefaultPipelineConfig()
      const configPath = await createTestConfig('default-test.json', defaultConfig)
      
      // Should load without errors
      const loaded = await loadPipelineConfig(configPath)
      deepStrictEqual(loaded, defaultConfig)
    })
  })

  describe('error handling', () => {
    it('should preserve error chain for file read errors', async () => {
      const nonExistentPath = join(testDir, 'definitely-does-not-exist.json')
      
      try {
        await loadPipelineConfig(nonExistentPath)
      } catch (error) {
        ok(error instanceof ConfigLoadError)
        strictEqual(error.filePath, nonExistentPath)
        ok(error.message.includes('Configuration file not found'))
      }
    })

    it('should include file path in validation errors', async () => {
      const invalidConfig = { not: 'valid' }
      const configPath = await createTestConfig('validation-error.json', invalidConfig)
      
      try {
        await loadPipelineConfig(configPath)
      } catch (error) {
        ok(error instanceof ConfigValidationError)
        strictEqual(error.filePath, configPath)
      }
    })
  })

  describe('environment variable expansion', () => {
    it('should handle missing environment variables gracefully', async () => {
      const configWithMissingEnv = {
        input: {
          type: 'file',
          path: '${MISSING_VAR}/input.csv'
        },
        output: {
          path: './output.jsonl',
          format: 'jsonl'
        },
        transformations: []
      }

      const configPath = await createTestConfig('missing-env.json', configWithMissingEnv)
      const loaded = await loadPipelineConfig(configPath)

      // Missing env var should be replaced with empty string
      const fileInput5 = loaded.input as FileInputConfig
      strictEqual(fileInput5.path, '/input.csv')
    })

    it('should expand multiple variables in same string', async () => {
      process.env.TEST_DIR = '/test'
      process.env.TEST_FILE = 'data.csv'

      const config = {
        input: {
          type: 'file',
          path: '${TEST_DIR}/${TEST_FILE}'
        },
        output: {
          path: './output.jsonl',
          format: 'jsonl'
        },
        transformations: []
      }

      const configPath = await createTestConfig('multi-env.json', config)
      const loaded = await loadPipelineConfig(configPath)

      const fileInput6 = loaded.input as FileInputConfig
      strictEqual(fileInput6.path, '/test/data.csv')

      delete process.env.TEST_DIR
      delete process.env.TEST_FILE
    })
  })
})