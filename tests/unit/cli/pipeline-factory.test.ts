import { deepStrictEqual, ok, rejects, strictEqual } from 'node:assert'
import * as fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { afterEach, beforeEach, describe, it } from 'node:test'
import { PipelineFactory } from '../../../src/cli/pipeline-factory'
import type { PipelineConfig, TransformConfig } from '../../../src/interfaces'
import { forceCleanupAsyncHandles } from '../../helpers/test-cleanup'
import type { LogReturnsParams, MissingValueParams, PriceCalcParams, ZScoreParams, MinMaxParams } from '../../../src/transforms'

describe('Pipeline Factory', () => {
  let tempDir: string

  beforeEach(async () => {
    // Create temp directory for test files
    tempDir = await fs.mkdtemp(path.join(tmpdir(), 'pipeline-factory-test-'))
  })

  afterEach(async () => {
    // Clean up temp directory
    await fs.rm(tempDir, { recursive: true, force: true })
    forceCleanupAsyncHandles()
  })

  describe('createPipeline', () => {
    it('should create a pipeline with CSV input and JSONL output', async () => {
      const inputPath = path.join(tempDir, 'input.csv')
      const outputPath = path.join(tempDir, 'output.jsonl')
      
      // Create test CSV file
      await fs.writeFile(inputPath, 'timestamp,open,high,low,close,volume\n1234567890,100,110,90,105,1000\n')
      
      const config: PipelineConfig = {
        input: {
          type: 'file',
          format: 'csv',
          path: inputPath,
          hasHeader: true
        },
        output: {
          path: outputPath,
          format: 'jsonl',
          overwrite: true
        },
        transformations: [],
        options: {
          chunkSize: 100,
          showProgress: false
        }
      }
      
      const pipeline = await PipelineFactory.createPipeline(config as any)
      
      ok(pipeline)
      ok(pipeline.getConfig())
      strictEqual(pipeline.hasTransforms(), false)
    })

    it('should create a pipeline with transforms', async () => {
      const inputPath = path.join(tempDir, 'input.csv')
      const outputPath = path.join(tempDir, 'output.jsonl')
      
      await fs.writeFile(inputPath, 'timestamp,open,high,low,close,volume\n1234567890,100,110,90,105,1000\n')
      
      const logReturnsTx: TransformConfig<LogReturnsParams> = {
        type: 'logReturns',
        params: {
          in: ['close'],
          out: ['returns']
        }
      }

      const minMaxTx: TransformConfig<MinMaxParams> = {
        type: 'minMax',
        params: {
          in: ['returns'],
          out: ['returns_zscore']
        }
      }

      const config: PipelineConfig = {
        input: {
          type: 'file',
          format: 'csv',
          path: inputPath,
          hasHeader: true
        },
        output: {
          path: outputPath,
          format: 'jsonl'
        },
        transformations: [
          logReturnsTx,
          minMaxTx
        ],
        options: {
          showProgress: false
        }
      }
      
      const pipeline = await PipelineFactory.createPipeline(config as any)
      
      ok(pipeline)
      strictEqual(pipeline.hasTransforms(), true)
      
      const transformInfo = pipeline.getTransformInfo()
      ok(transformInfo)
      strictEqual(transformInfo.type, 'pipeline')
    })

    it('should handle JSONL input and CSV output', async () => {
      const inputPath = path.join(tempDir, 'input.jsonl')
      const outputPath = path.join(tempDir, 'output.csv')
      
      await fs.writeFile(
        inputPath, 
        '{"timestamp":1234567890,"open":100,"high":110,"low":90,"close":105,"volume":1000}\n'
      )

      const tx: TransformConfig<PriceCalcParams> = {
        type: 'priceCalc',
        params: {
          calculation: 'hlc3'
        }
      }

      const config: PipelineConfig = {
        input: {
          type: 'file',
          format: 'jsonl',
          path: inputPath
        },
        output: {
          path: outputPath,
          format: 'csv',
          overwrite: true
        },
        transformations: [
          tx,
        ],
        options: {
          chunkSize: 50
        },
        metadata: {
          name: 'Test Pipeline',
          version: '1.0.0'
        }
      }
      
      const pipeline = await PipelineFactory.createPipeline(config as any)
      
      ok(pipeline)
      const metadata = pipeline.getMetadata()
      ok(metadata)
      strictEqual(metadata.name, 'Test Pipeline')
      strictEqual(metadata.version, '1.0.0')
    })

    it('should throw error for unsupported input type', async () => {
      const config = {
        input: {
          type: 'parquet',
          path: 'test.parquet'
        },
        output: {
          path: 'output.jsonl',
          format: 'jsonl'
        },
        transformations: [],
        options: {}
      }
      
      await rejects(
        PipelineFactory.createPipeline(config as any),
        /Unsupported input type: parquet/
      )
    })

    it('should throw error for unsupported output format', async () => {
      const config = {
        input: {
          type: 'file',
          path: 'test.csv',
          format: 'csv'
        },
        output: {
          path: 'output.xml',
          format: 'xml' as any
        },
        transformations: [],
        options: {}
      }
      
      await rejects(
        PipelineFactory.createPipeline(config as any),
        /Unsupported output format: xml/
      )
    })

    it('should throw error for unknown transform type', async () => {
      const config = {
        input: {
          type: 'file',
          path: 'test.csv',
          format: 'csv'
        },
        output: {
          path: 'output.jsonl',
          format: 'jsonl'
        },
        transformations: [
          {
            type: 'unknownTransform',
            params: {  }
          }
        ],
        options: {}
      }
      
      await rejects(
        PipelineFactory.createPipeline(config as any),
        /Unknown transform type at index 0: unknownTransform/
      )
    })
  })

  describe('createFromConfig', () => {
    it('should validate and create pipeline from raw config', async () => {
      const inputPath = path.join(tempDir, 'input.csv')
      await fs.writeFile(inputPath, 'timestamp,open,high,low,close,volume\n')
      
      const config = {
        input: {
          type: 'file',
          format: 'csv',
          path: inputPath
        },
        output: {
          path: path.join(tempDir, 'output.jsonl'),
          format: 'jsonl',
        },
        transformations: [],
        options: {}
      }
      
      const pipeline = await PipelineFactory.createFromConfig(config)
      ok(pipeline)
    })

    it('should reject invalid configuration', async () => {
      const config = {
        input: {},
        output: {},
        transformations: []
      }
      
      await rejects(
        PipelineFactory.createFromConfig(config),
        /Configuration validation failed/
      )
    })
  })

  describe('createFromFile', () => {
    it('should load and create pipeline from config file', async () => {
      const configPath = path.join(tempDir, 'config.json')
      const inputPath = path.join(tempDir, 'input.csv')
      
      await fs.writeFile(inputPath, 'timestamp,open,high,low,close,volume\n')
      
      const config = {
        input: {
          type: 'file',
          format: 'csv',
          path: inputPath
        },
        output: {
          path: path.join(tempDir, 'output.jsonl'),
          format: 'jsonl'
        },
        transformations: [],
        options: {
          showProgress: false
        }
      }
      
      await fs.writeFile(configPath, JSON.stringify(config, null, 2))
      
      const pipeline = await PipelineFactory.createFromFile(configPath)
      ok(pipeline)
    })
  })

  describe('utility methods', () => {
    it('should get available transforms', () => {
      const transforms = PipelineFactory.getAvailableTransforms()
      
      ok(Array.isArray(transforms))
      ok(transforms.includes('logReturns'))
      ok(transforms.includes('minMax'))
      ok(transforms.includes('zScore'))
      ok(transforms.includes('priceCalc'))
      ok(transforms.includes('missingValues'))
      ok(transforms.includes('timeframeAggregation'))
    })

    it('should get available input types', () => {
      const inputTypes = PipelineFactory.getAvailableInputTypes()
      
      deepStrictEqual(inputTypes, ['csv', 'jsonl'])
    })

    it('should get available output formats', () => {
      const outputFormats = PipelineFactory.getAvailableOutputFormats()
      
      deepStrictEqual(outputFormats, ['csv', 'jsonl'])
    })

    it('should check if transform is supported', () => {
      strictEqual(PipelineFactory.isTransformSupported('logReturns'), true)
      strictEqual(PipelineFactory.isTransformSupported('minMax'), true)
      strictEqual(PipelineFactory.isTransformSupported('unknownTransform'), false)
    })

    it('should get transform parameters', () => {
      const logReturnsParams = PipelineFactory.getTransformParams('logReturns')
      deepStrictEqual(logReturnsParams, ['in', 'out', 'base'])
      
      const minMaxParams = PipelineFactory.getTransformParams('minMax')
      deepStrictEqual(minMaxParams, ['in', 'out', 'windowSize', 'min', 'max'])
      
      const unknownParams = PipelineFactory.getTransformParams('unknownTransform')
      deepStrictEqual(unknownParams, [])
    })
  })

  describe('complex pipeline creation', () => {
    it('should create pipeline with multiple transforms and metadata', async () => {
      const inputPath = path.join(tempDir, 'data.csv')
      const outputPath = path.join(tempDir, 'processed.jsonl')

      await fs.writeFile(
        inputPath,
        'timestamp,open,high,low,close,volume,symbol\n' +
        '1234567890,100,110,90,105,1000,BTC-USD\n' +
        '1234567900,105,115,95,110,1100,BTC-USD\n'
      )

      const imputeTx: TransformConfig<MissingValueParams> = {
        type: 'missingValues',
        params: {
          strategy: 'forward',
          in: ['open', 'high', 'low', 'close', 'volume'],
          out: ['open_minmax', 'high_minmax', 'low_minmax', 'close_minmax', 'volume_minmax']
        }
      }

      const priceCalcTx: TransformConfig<PriceCalcParams> = {
        type: 'priceCalc',
        params: {
          calculation: 'ohlc4'
        }
      }

      const logReturnsTx: TransformConfig<LogReturnsParams> = {
        type: 'logReturns',
        params: {
          in: ['close'],
          out: ['returns']
        }
      }

      const zScoreTx: TransformConfig<ZScoreParams> = {
        type: 'zScore',
        params: {
          in: ['returns'],
          out: ['returns_zscore'],
          windowSize: 20
        }
      }

      const config: PipelineConfig = {
        input: {
          type: 'file',
          format: 'csv',
          path: inputPath,
          hasHeader: true,
          columnMapping: {
            timestamp: 'timestamp',
            open: 'open',
            high: 'high',
            low: 'low',
            close: 'close',
            volume: 'volume',
            symbol: 'symbol'
          }
        },
        output: {
          path: outputPath,
          format: 'jsonl',
          overwrite: true
        },
        transformations: [
          imputeTx,
          priceCalcTx,
          logReturnsTx,
          zScoreTx
        ],
        options: {
          chunkSize: 1000,
          continueOnError: true,
          maxErrors: 10,
          showProgress: false
        },
        metadata: {
          name: 'BTC Price Analysis Pipeline',
          version: '2.0.0',
          description: 'Processes BTC price data with returns and normalization',
          author: 'Test Suite',
          created: new Date().toISOString()
        }
      }
      
      const pipeline = await PipelineFactory.createPipeline(config as any)
      
      ok(pipeline)
      strictEqual(pipeline.hasTransforms(), true)
      
      const metadata = pipeline.getMetadata()
      ok(metadata)
      strictEqual(metadata.name, 'BTC Price Analysis Pipeline')
      strictEqual(metadata.version, '2.0.0')
      strictEqual(metadata.author, 'Test Suite')
      
      const transformInfo = pipeline.getTransformInfo()
      ok(transformInfo)
      strictEqual(transformInfo.type, 'pipeline')
      strictEqual(transformInfo.name, 'Transform Pipeline')
    })
  })
})