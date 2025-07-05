import { deepStrictEqual, ok, rejects, strictEqual } from 'node:assert'
import { mkdir, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, it } from 'node:test'
import {
  ConfigLoadError,
  ConfigValidationError,
  createDefaultPipelineConfig,
  loadPipelineConfig
} from '../../../src/cli/config-loader'

describe('Config Loader', () => {
  const testDir = join(process.cwd(), 'test-configs')
  let createdFiles: string[] = []

  beforeEach(async () => {
    // Create test directory
    try {
      await mkdir(testDir, { recursive: true })
    } catch {
      // Directory might already exist
    }
  })

  afterEach(async () => {
    // Clean up created files
    for (const file of createdFiles) {
      try {
        await unlink(file)
      } catch {
        // File might not exist
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
            enabled: true,
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

      strictEqual(loaded.input.path, './data/test.csv')
      strictEqual(loaded.input.format, 'csv')
      strictEqual(loaded.input.chunkSize, 500)
      strictEqual(loaded.output.path, './output/result.jsonl')
      strictEqual(loaded.output.format, 'jsonl')
      strictEqual(loaded.output.overwrite, true)
      strictEqual(loaded.transformations.length, 1)
      strictEqual(loaded.transformations[0]!.type, 'logReturns')
      strictEqual(loaded.transformations[0]!.enabled, true)
      strictEqual(loaded.transformations[0]!.params.outputField, 'returns')
      strictEqual(loaded.options.chunkSize, 1000)
      strictEqual(loaded.options.continueOnError, true)
      strictEqual(loaded.options.maxErrors, 50)
      strictEqual(loaded.options.showProgress, false)
    })

    it('should load minimal valid configuration', async () => {
      const minimalConfig = {
        input: {
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

      strictEqual(loaded.input.path, './input.csv')
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

      strictEqual(loaded.input.path, '/test/input.csv')
      strictEqual(loaded.input.chunkSize, '2000') // Note: JSON parsing keeps this as string
      strictEqual(loaded.output.path, '/test/output.jsonl')

      // Clean up environment variables
      delete process.env.TEST_INPUT_PATH
      delete process.env.TEST_OUTPUT_PATH  
      delete process.env.TEST_CHUNK_SIZE
    })

    it('should handle relative and absolute paths', async () => {
      const config = {
        input: { path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: []
      }

      const configPath = await createTestConfig('paths.json', config)
      
      // Test with relative path
      const loadedRelative = await loadPipelineConfig('test-configs/paths.json')
      ok(loadedRelative.input.path)

      // Test with absolute path
      const loadedAbsolute = await loadPipelineConfig(configPath)
      ok(loadedAbsolute.input.path)
    })

    it('should throw ConfigLoadError for non-existent file', async () => {
      await rejects(
        loadPipelineConfig('./non-existent.json'),
        (error: ConfigLoadError) => {
          strictEqual(error.name, 'ConfigLoadError')
          ok(error.message.includes('Configuration file not found'))
          return true
        }
      )
    })

    it('should throw ConfigLoadError for invalid JSON', async () => {
      const invalidJsonPath = join(testDir, 'invalid.json')
      await writeFile(invalidJsonPath, '{ invalid json }')
      createdFiles.push(invalidJsonPath)

      await rejects(
        loadPipelineConfig(invalidJsonPath),
        (error: ConfigLoadError) => {
          strictEqual(error.name, 'ConfigLoadError')
          ok(error.message.includes('Failed to parse JSON'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for missing input section', async () => {
      const invalidConfig = {
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: []
      }

      const configPath = await createTestConfig('no-input.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('must have an "input" section'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for missing output section', async () => {
      const invalidConfig = {
        input: { path: './input.csv' },
        transformations: []
      }

      const configPath = await createTestConfig('no-output.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('must have an "output" section'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for invalid output format', async () => {
      const invalidConfig = {
        input: { path: './input.csv' },
        output: { path: './output.txt', format: 'txt' },
        transformations: []
      }

      const configPath = await createTestConfig('invalid-format.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('Output format must be one of'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for missing transformations array', async () => {
      const invalidConfig = {
        input: { path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' }
      }

      const configPath = await createTestConfig('no-transforms.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('must have a "transformations" array'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for invalid transformation', async () => {
      const invalidConfig = {
        input: { path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: [
          {
            type: 'logReturns',
            // Missing enabled and params
          }
        ]
      }

      const configPath = await createTestConfig('invalid-transform.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('must have an "enabled" boolean'))
          return true
        }
      )
    })

    it('should throw ConfigValidationError for invalid options', async () => {
      const invalidConfig = {
        input: { path: './input.csv' },
        output: { path: './output.jsonl', format: 'jsonl' },
        transformations: [],
        options: {
          chunkSize: -1 // Invalid chunk size
        }
      }

      const configPath = await createTestConfig('invalid-options.json', invalidConfig)

      await rejects(
        loadPipelineConfig(configPath),
        (error: ConfigValidationError) => {
          strictEqual(error.name, 'ConfigValidationError')
          ok(error.message.includes('chunkSize must be a positive integer'))
          return true
        }
      )
    })

    it('should handle complex configuration with metadata', async () => {
      const complexConfig = {
        input: {
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
          path: './output/result.sqlite',
          format: 'sqlite',
          overwrite: false
        },
        transformations: [
          {
            type: 'logReturns',
            enabled: true,
            params: {
              outputField: 'returns',
              priceField: 'close'
            }
          },
          {
            type: 'zScore',
            enabled: false,
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
      strictEqual(loaded.input.columnMapping?.timestamp, 'ts')
      strictEqual(loaded.input.columnMapping?.close, 'c')
      strictEqual(loaded.input.exchange, 'test')
      strictEqual(loaded.input.symbol, 'BTC-USD')
      
      strictEqual(loaded.output.format, 'sqlite')
      strictEqual(loaded.output.overwrite, false)
      
      strictEqual(loaded.transformations.length, 2)
      strictEqual(loaded.transformations[0]!.enabled, true)
      strictEqual(loaded.transformations[1]!.enabled, false)
      strictEqual(loaded.transformations[1]!.params.windowSize, 20)
      
      strictEqual(loaded.options.continueOnError, false)
      
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

      // Check default values
      strictEqual(defaultConfig.input.path, './data/input.csv')
      strictEqual(defaultConfig.input.format, 'csv')
      strictEqual(defaultConfig.input.chunkSize, 1000)
      
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
      strictEqual(loaded.input.path, '/input.csv')
    })

    it('should expand multiple variables in same string', async () => {
      process.env.TEST_DIR = '/test'
      process.env.TEST_FILE = 'data.csv'

      const config = {
        input: {
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

      strictEqual(loaded.input.path, '/test/data.csv')

      delete process.env.TEST_DIR
      delete process.env.TEST_FILE
    })
  })
})