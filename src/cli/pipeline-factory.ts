import type { PipelineConfig, Transform, TransformConfig } from '../interfaces'
import { Pipeline } from '../pipeline'
import type { FileProvider } from '../providers'
import { CsvFileProvider, JsonlFileProvider } from '../providers'
import type { OhlcvRepository } from '../repositories'
import { CsvRepository, JsonlRepository, SqliteRepository } from '../repositories'
import {
  LogReturnsNormalizer,
  MinMaxNormalizer,
  MissingValueHandler,
  PriceCalculations,
  TimeframeAggregator,
  TransformPipeline,
  ZScoreNormalizer,
} from '../transforms'
import { ConfigLoader } from './config-loader'
import type { ValidatedConfig } from './config-validator'
import { ConfigValidator } from './config-validator'

/**
 * Transform factory map for creating transform instances
 */
const TRANSFORM_FACTORIES: Record<string, (params: any) => Transform> = {
  logReturns: (params) => new LogReturnsNormalizer(params),
  minMax: (params) => new MinMaxNormalizer(params),
  zScore: (params) => new ZScoreNormalizer(params),
  priceCalc: (params) => new PriceCalculations(params),
  missingValues: (params) => new MissingValueHandler(params),
  timeframeAggregation: (params) => new TimeframeAggregator(params),
}

/**
 * Factory for creating Pipeline instances from validated configurations
 */
export class PipelineFactory {
  /**
   * Create a Pipeline instance from a validated configuration
   */
  public static async createPipeline(config: ValidatedConfig): Promise<Pipeline> {
    // Create input provider
    const provider = await this.createProvider(config.input)

    // Create transforms
    const transforms = this.createTransforms(config.transformations)

    // Create transform pipeline
    const transformPipeline = transforms.length > 0
      ? new TransformPipeline({
        transforms,
        name: config.metadata?.name || 'Data Pipeline',
      })
      : undefined

    // Create output repository
    const repository = await this.createRepository(config.output)

    // Create and return pipeline
    return new Pipeline({
      provider,
      transform: transformPipeline,
      repository,
      options: {
        chunkSize: config.options.chunkSize,
        continueOnError: config.options.continueOnError,
        maxErrors: config.options.maxErrors,
        showProgress: config.options.showProgress,
      },
      metadata: config.metadata,
    })
  }

  /**
   * Create a file provider from configuration
   */
  private static async createProvider(config: PipelineConfig['input']): Promise<FileProvider> {
    switch (config.type) {
      case 'csv':
        return new CsvFileProvider({
          path: config.path,
          columnMapping: config.columnMapping,
          delimiter: config.delimiter,
          hasHeader: config.hasHeader,
          encoding: config.encoding,
        })

      case 'jsonl':
        return new JsonlFileProvider({
          path: config.path,
          columnMapping: config.columnMapping,
          encoding: config.encoding,
        })

      default:
        throw new Error(`Unsupported input type: ${(config as any).type}`)
    }
  }

  /**
   * Create transforms from configuration
   */
  private static createTransforms(configs: TransformConfig[]): Transform[] {
    return configs.map((config, index) => {
      const factory = TRANSFORM_FACTORIES[config.type]

      if (!factory) {
        throw new Error(`Unknown transform type at index ${index}: ${config.type}`)
      }

      try {
        return factory(config.params)
      } catch (error) {
        throw new Error(
          `Failed to create transform '${config.type}' at index ${index}: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
        )
      }
    })
  }

  /**
   * Create a repository from output configuration
   */
  private static async createRepository(config: PipelineConfig['output']): Promise<OhlcvRepository> {
    let repository: OhlcvRepository

    switch (config.format) {
      case 'sqlite':
        repository = new SqliteRepository()
        break

      case 'csv':
        repository = new CsvRepository()
        break

      case 'jsonl':
        repository = new JsonlRepository()
        break

      default:
        throw new Error(`Unsupported output format: ${config.format}`)
    }

    // Initialize repository
    await repository.initialize({
      connectionString: config.path,
      options: {
        overwrite: config.overwrite,
        columnMapping: config.columnMapping,
      },
    })

    return repository
  }

  /**
   * Create a pipeline from a raw configuration object
   * This includes validation
   */
  public static async createFromConfig(config: unknown): Promise<Pipeline> {
    // Validate configuration first
    const validatedConfig = ConfigValidator.validate(config)

    // Create pipeline from validated config
    return this.createPipeline(validatedConfig)
  }

  /**
   * Create a pipeline from a configuration file
   */
  public static async createFromFile(configPath: string): Promise<Pipeline> {
    // Load configuration
    const config = await ConfigLoader.load(configPath)

    // Create pipeline from config
    return this.createFromConfig(config)
  }

  /**
   * Get available transform types
   */
  public static getAvailableTransforms(): string[] {
    return Object.keys(TRANSFORM_FACTORIES)
  }

  /**
   * Get available input types
   */
  public static getAvailableInputTypes(): string[] {
    return ['csv', 'jsonl']
  }

  /**
   * Get available output formats
   */
  public static getAvailableOutputFormats(): string[] {
    return ['sqlite', 'csv', 'jsonl']
  }

  /**
   * Validate that a transform type is supported
   */
  public static isTransformSupported(type: string): boolean {
    return type in TRANSFORM_FACTORIES
  }

  /**
   * Get transform parameter requirements
   * This could be extended to return schema information
   */
  public static getTransformParams(type: string): string[] {
    switch (type) {
      case 'logReturns':
        return ['outputField', 'priceField']
      case 'minMax':
        return ['fields', 'targetMin', 'targetMax']
      case 'zScore':
        return ['fields', 'windowSize', 'suffix', 'addSuffix']
      case 'priceCalc':
        return ['calculation']
      case 'missingValues':
        return ['strategy', 'fields', 'fillValue']
      case 'timeframeAggregation':
        return ['targetTimeframe', 'aggregationMethod']
      default:
        return []
    }
  }
}
