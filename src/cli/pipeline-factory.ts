import type { DataProvider, InputConfig, PipelineConfig, Transform, TransformConfig } from '../interfaces'
import { Pipeline } from '../pipeline'
import type { FileProvider } from '../providers'
import { AlpacaProvider, CoinbaseProvider, CsvFileProvider, JsonlFileProvider } from '../providers'
import type { OhlcvRepository } from '../repositories'
import { CsvRepository, JsonlRepository } from '../repositories'
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
        description: config.metadata?.name || 'Data Pipeline',
      })
      : undefined

    // Create output repository
    const repository = await this.createRepository(config.output)

    // Prepare historical params if using a data provider
    let historicalParams
    if (config.input.type === 'provider') {
      // Parse duration to determine start/end times
      const duration = config.input.duration || '1h'
      const end = Date.now()
      let start = end
      
      if (duration === 'continuous') {
        // For continuous mode, we'll handle this differently in the pipeline
        start = end - 3600000 // Default to last hour for initial data
      } else if (duration.endsWith('m')) {
        const minutes = parseInt(duration.slice(0, -1))
        if (isNaN(minutes) || minutes <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        start = end - (minutes * 60000)
      } else if (duration.endsWith('h')) {
        const hours = parseInt(duration.slice(0, -1))
        if (isNaN(hours) || hours <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        start = end - (hours * 3600000)
      } else if (duration.endsWith('d')) {
        const days = parseInt(duration.slice(0, -1))
        if (isNaN(days) || days <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        start = end - (days * 86400000)
      } else if (duration.endsWith('w')) {
        const weeks = parseInt(duration.slice(0, -1))
        if (isNaN(weeks) || weeks <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        start = end - (weeks * 7 * 86400000)
      } else if (duration.endsWith('M')) {
        const months = parseInt(duration.slice(0, -1))
        if (isNaN(months) || months <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        // Use 30 days as approximate month length
        start = end - (months * 30 * 86400000)
      } else if (duration.endsWith('y')) {
        const years = parseInt(duration.slice(0, -1))
        if (isNaN(years) || years <= 0) {
          throw new Error(`Invalid duration value: ${duration}. Must be a positive number.`)
        }
        // Use 365 days as approximate year length
        start = end - (years * 365 * 86400000)
      } else if (duration.endsWith('bars')) {
        const bars = parseInt(duration.slice(0, -4))
        if (isNaN(bars) || bars <= 0) {
          throw new Error(`Invalid bar count: ${duration}. Must be a positive number.`)
        }
        // For bar count, we'll use a reasonable time window and let the provider handle limiting
        start = end - (7 * 86400000) // Default to 7 days
      } else {
        throw new Error(`Invalid duration format: ${duration}. Supported formats: m (minutes), h (hours), d (days), w (weeks), M (months), y (years), bars, or 'continuous'`)
      }
      
      historicalParams = {
        symbols: config.input.symbols,
        start,
        end,
        timeframe: config.input.timeframe,
      }
    }

    // Create and return pipeline
    return new Pipeline({
      provider,
      transform: transformPipeline,
      repository,
      options: {
        chunkSize: config.options?.chunkSize,
        continueOnError: config.options?.continueOnError,
        maxErrors: config.options?.maxErrors,
        showProgress: config.options?.showProgress,
      },
      metadata: config.metadata,
      historicalParams,
    })
  }

  /**
   * Create a provider from configuration
   */
  private static async createProvider(config: InputConfig): Promise<DataProvider | FileProvider> {
    if (config.type === 'file') {
      // Handle file-based providers
      const fileExt = config.path.toLowerCase().split('.').pop()
      const format = config.format || fileExt
      
      switch (format) {
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
          throw new Error(`Unsupported file format: ${format}`)
      }
    } else if (config.type === 'provider') {
      // Handle API-based providers
      let provider: DataProvider
      
      switch (config.provider) {
        case 'coinbase':
          provider = new CoinbaseProvider()
          break
          
        case 'alpaca':
          provider = new AlpacaProvider()
          break
          
        default:
          throw new Error(`Unsupported provider: ${config.provider}`)
      }
      
      // Validate environment variables
      provider.validateEnvVars()
      
      // Connect to the provider
      await provider.connect()
      
      return provider
    } else {
      throw new Error(`Unsupported input type: ${(config as any).type}`)
    }
  }

  /**
   * Create transforms from configuration
   */
  private static createTransforms(configs: TransformConfig[]): Transform[] {
    return configs
      .filter(config => !config.disabled)
      .map((config, index) => {
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
    return ['csv', 'jsonl']
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
