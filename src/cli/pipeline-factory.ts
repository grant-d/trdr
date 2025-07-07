import type { DataProvider, InputConfig, PipelineConfig, Transform, TransformConfig } from '../interfaces'
import type { Pipeline } from '../pipeline'
import { BufferPipeline } from '../pipeline'
import type { FileProvider } from '../providers'
import { AlpacaProvider, CoinbaseProvider, CsvFileProvider, JsonlFileProvider } from '../providers'
import type { OhlcvRepository } from '../repositories'
import { CsvRepository } from '../repositories'
import {
  AverageTrueRange,
  BollingerBands,
  DollarBarGenerator,
  ExponentialMovingAverage,
  FractionalDiffNormalizer,
  HeikinAshi,
  ImputeTransform,
  LogReturnsNormalizer,
  LorentzianDistanceBarGenerator,
  Macd,
  MapTransform,
  MinMaxNormalizer,
  PriceCalculations,
  RelativeStrengthIndex,
  ShannonInformationBarGenerator,
  SimpleMovingAverage,
  StatisticalRegimeBarGenerator,
  TickBarGenerator,
  TickImbalanceBarGenerator,
  TickRunBarGenerator,
  TimeBarGenerator,
  VolumeBarGenerator,
  VolumeWeightedAveragePrice,
  ZScoreNormalizer
} from '../transforms'
import { DataBuffer, DataSlice } from '../utils'
import { ConfigLoader } from './config-loader'
import { ConfigValidator } from './config-validator'

/**
 * Transform factory map for creating transform instances
 * Some transforms require a buffer, others don't
 */
const TRANSFORM_FACTORIES: Record<
  string,
  (params: any, buffer?: DataBuffer) => Transform
> = {
  logReturns: (params, buffer) => new LogReturnsNormalizer(params, new DataSlice(buffer!, 0, buffer!.length())),
  map: (params, buffer) =>
    new MapTransform(params, new DataSlice(buffer!, 0, buffer!.length())),
  minMax: (params, buffer) => new MinMaxNormalizer(params, new DataSlice(buffer!, 0, buffer!.length())),
  zScore: (params, buffer) => new ZScoreNormalizer(params, new DataSlice(buffer!, 0, buffer!.length())),
  fractionalDiff: (params, buffer) =>
    new FractionalDiffNormalizer(params, new DataSlice(buffer!, 0, buffer!.length())),
  priceCalc: (params, buffer) =>
    new PriceCalculations(params, new DataSlice(buffer!, 0, buffer!.length())),
  missingValues: (params, buffer) =>
    new ImputeTransform(params, new DataSlice(buffer!, 0, buffer!.length())),
  timeBars: (params, buffer) => new TimeBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  sma: (params, buffer) =>
    new SimpleMovingAverage(
      params,
      new DataSlice(buffer!, 0, buffer!.length())
    ),
  ema: (params, buffer) =>
    new ExponentialMovingAverage(
      params,
      new DataSlice(buffer!, 0, buffer!.length())
    ),
  rsi: (params, buffer) =>
    new RelativeStrengthIndex(
      params,
      new DataSlice(buffer!, 0, buffer!.length())
    ),
  bollinger: (params, buffer) =>
    new BollingerBands(params, new DataSlice(buffer!, 0, buffer!.length())),
  macd: (params, buffer) =>
    new Macd(params, new DataSlice(buffer!, 0, buffer!.length())),
  atr: (params, buffer) =>
    new AverageTrueRange(params, new DataSlice(buffer!, 0, buffer!.length())),
  vwap: (params, buffer) =>
    new VolumeWeightedAveragePrice(
      params,
      new DataSlice(buffer!, 0, buffer!.length())
    ),
  tickBars: (params, buffer) => new TickBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  volumeBars: (params, buffer) => new VolumeBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  dollarBars: (params, buffer) => new DollarBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  tickImbalanceBars: (params, buffer) =>
    new TickImbalanceBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  tickRunBars: (params, buffer) => new TickRunBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  regimeBars: (params, buffer) =>
    new StatisticalRegimeBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  lorentzianBars: (params, buffer) =>
    new LorentzianDistanceBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  shannonInfoBars: (params, buffer) =>
    new ShannonInformationBarGenerator(params, new DataSlice(buffer!, 0, buffer!.length())),
  heikinAshi: (params, buffer) =>
    new HeikinAshi(params, new DataSlice(buffer!, 0, buffer!.length()))
}

/**
 * Factory for creating Pipeline instances from validated configurations
 */
export class PipelineFactory {
  /**
   * Create a Pipeline instance from a validated configuration
   */
  public static async createPipeline(
    config: PipelineConfig
  ): Promise<Pipeline> {
    // Create input provider
    const provider = await this.createProvider(config.input)

    // Create initial buffer based on provider's schema
    const initialBuffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        symbol: { index: 1 },
        exchange: { index: 2 },
        open: { index: 3 },
        high: { index: 4 },
        low: { index: 5 },
        close: { index: 6 },
        volume: { index: 7 }
      }
    })

    // Create transforms with buffer chaining
    const { transforms, finalBuffer } = this.createTransformsWithBuffer(
      config.transformations,
      initialBuffer
    )

    // Create output repository
    const repository = await this.createRepository(config.output)

    // Use BufferPipeline for buffer-based transforms only
    const bufferPipeline = new BufferPipeline({
      provider: provider as FileProvider,
      transforms,
      repository,
      batchSize: config.options?.chunkSize,
      initialBuffer,
      finalBuffer
    })

    // TODO: Wrap BufferPipeline in Pipeline interface for compatibility
    return bufferPipeline as any
  }

  /**
   * Create a provider from configuration
   */
  private static async createProvider(
    config: InputConfig
  ): Promise<DataProvider | FileProvider> {
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
            encoding: config.encoding
          })

        case 'jsonl':
          return new JsonlFileProvider({
            path: config.path,
            columnMapping: config.columnMapping,
            encoding: config.encoding
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
   * Create transforms from configuration with buffer flow
   * @param configs Transform configurations
   * @param initialBuffer The initial buffer from the provider
   * @returns Array of transforms and the final buffer
   */
  private static createTransformsWithBuffer(
    configs: TransformConfig[],
    initialBuffer: DataBuffer
  ): { transforms: Transform[]; finalBuffer: DataBuffer } {
    const transforms: Transform[] = []
    let currentBuffer = initialBuffer

    for (const config of configs.filter((c) => !c.disabled)) {
      const factory = TRANSFORM_FACTORIES[config.type]

      if (!factory) {
        throw new Error(`Unknown transform type: ${config.type}`)
      }

      try {
        // Check if this transform type requires a buffer
        const bufferlessTransforms = [
          'timeframeAggregation',
          'tickBars',
          'volumeBars',
          'dollarBars',
          'tickImbalanceBars',
          'tickRunBars',
          'heikinAshi',
          'statisticalRegime',
          'lorentzianDistance',
          'shannonInformation'
        ]

        const transform = bufferlessTransforms.includes(config.type)
          ? factory(config.params)
          : factory(config.params, currentBuffer)

        transforms.push(transform)

        // Get the output buffer from the transform
        // This will be either the same buffer (for in-place transforms)
        // or a new buffer (for reshaping transforms)
        currentBuffer = transform.outputBuffer
      } catch (error) {
        throw new Error(
          `Failed to create transform ${config.type}: ${error instanceof Error ? error.message : String(error)}`
        )
      }
    }

    return { transforms, finalBuffer: currentBuffer }
  }

  /**
   * Create a repository from output configuration
   */
  private static async createRepository(
    config: PipelineConfig['output']
  ): Promise<OhlcvRepository> {
    let repository: OhlcvRepository

    switch (config.format) {
      case 'csv':
        repository = new CsvRepository()
        break

      case 'jsonl':
        // TODO: Implement JSONL repository with buffer support
        throw new Error('JSONL output format not yet supported')

      default:
        throw new Error(`Unsupported output format: ${config.format}`)
    }

    // Initialize repository
    await repository.initialize({
      connectionString: config.path,
      options: {
        overwrite: config.overwrite,
        columnMapping: config.columnMapping
      }
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
        return ['in', 'out', 'base']
      case 'map':
        return ['in', 'out']
      case 'minMax':
        return ['in', 'out', 'windowSize', 'min', 'max']
      case 'zScore':
        return ['in', 'out', 'windowSize']
      case 'fractionalDiff':
        return ['in', 'out', 'd', 'maxWeights', 'minWeight']
      case 'priceCalc':
        return ['calculation']
      case 'missingValues':
        return ['in', 'out', 'strategy', 'fillValue', 'maxFillGap']
      case 'timeframeAggregation':
        return ['targetTimeframe', 'aggregationMethod']
      case 'sma':
        return ['in', 'out', 'period']
      case 'ema':
        return ['in', 'out', 'period']
      case 'rsi':
        return ['in', 'out', 'period']
      case 'bollinger':
        return ['in', 'out', 'period', 'stdDev']
      case 'macd':
        return ['in', 'out', 'fastPeriod', 'slowPeriod', 'signalPeriod']
      case 'atr':
        return ['out', 'period']
      case 'vwap':
        return ['out', 'anchorPeriod']
      default:
        return []
    }
  }
}
