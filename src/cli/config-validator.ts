import type { InputConfig, PipelineConfig, TransformConfig } from '../interfaces'

/**
 * Represents a validation error with details about the issue
 */
export interface ValidationError {
  /** The field or section where the error occurred */
  field: string
  /** Human-readable error message */
  message: string
  /** The invalid value that caused the error */
  value?: any
  /** Severity level of the error */
  severity: 'error' | 'warning'
}

/**
 * Result of configuration validation
 */
export interface ValidationResult {
  /** Whether the configuration is valid */
  isValid: boolean
  /** Array of validation errors and warnings */
  errors: ValidationError[]
  /** Array of just the error messages (for convenience) */
  errorMessages: string[]
  /** Array of just the warning messages (for convenience) */
  warningMessages: string[]
}

/**
 * Validated configuration type
 */
export type ValidatedConfig = PipelineConfig

/**
 * Maps transform types to their required and output fields
 */
const TRANSFORM_FIELD_MAPPINGS: Record<string, {
  requiredFields: string[]
  outputFields: string[]
}> = {
  logReturns: {
    requiredFields: ['close'],
    outputFields: ['returns']
  },
  zScore: {
    requiredFields: [], // Specified dynamically via params.fields
    outputFields: [] // Generated dynamically based on input fields
  },
  minMax: {
    requiredFields: [], // Specified dynamically via params.fields
    outputFields: [] // Generated dynamically based on input fields
  },
  percentChange: {
    requiredFields: ['close'],
    outputFields: ['pct_change']
  },
  priceCalc: {
    requiredFields: ['open', 'high', 'low', 'close'],
    outputFields: ['hlc3', 'ohlc4', 'typical'] // Depends on calculation type
  },
  movingAverage: {
    requiredFields: [], // Specified dynamically via params.fields
    outputFields: [] // Generated dynamically
  },
  rsi: {
    requiredFields: ['close'],
    outputFields: ['rsi']
  },
  bollinger: {
    requiredFields: ['close'],
    outputFields: ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']
  },
  macd: {
    requiredFields: ['close'],
    outputFields: ['macd', 'macd_signal', 'macd_histogram']
  },
  atr: {
    requiredFields: ['high', 'low', 'close'],
    outputFields: ['atr']
  },
  vwap: {
    requiredFields: ['high', 'low', 'close', 'volume'],
    outputFields: ['vwap']
  },
}

/**
 * Base OHLCV fields that are always available from data sources
 */
const BASE_OHLCV_FIELDS = ['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume']

/**
 * Comprehensive configuration validator for pipeline configurations
 */
export class ConfigValidator {
  private errors: ValidationError[] = []

  /**
   * Static method to validate configuration with detailed results
   */
  public static validateWithDetails(config: unknown): ValidationResult {
    const validator = new ConfigValidator()
    return validator.validateConfig(config)
  }

  /**
   * Static method to validate configuration and throw if invalid
   */
  public static validate(config: unknown): PipelineConfig {
    const result = ConfigValidator.validateWithDetails(config)
    if (!result.isValid) {
      throw new Error(`Configuration validation failed:\n${result.errorMessages.join('\n')}`)
    }
    return config as PipelineConfig
  }

  /**
   * Validates a complete pipeline configuration
   */
  public validateConfig(config: unknown): ValidationResult {
    // Basic type check
    if (!config || typeof config !== 'object') {
      this.addError('root', 'Configuration must be an object', config)
      return this.getResult()
    }

    const pipelineConfig = config as PipelineConfig
    return this.validate(pipelineConfig)
  }

  /**
   * Internal validation method
   */
  public validate(config: PipelineConfig): ValidationResult {
    this.errors = []

    // Validate each section
    this.validateInput(config.input)
    this.validateOutput(config.output)
    this.validateTransformations(config.transformations)
    this.validateOptions(config.options)
    this.validateMetadata(config.metadata)

    // Validate cross-section compatibility
    this.validateTransformChain(config.transformations)
    this.validateInputOutputCompatibility(config.input, config.output)

    // Separate errors and warnings
    const errors = this.errors.filter(e => e.severity === 'error')
    const warnings = this.errors.filter(e => e.severity === 'warning')

    return {
      isValid: errors.length === 0,
      errors: this.errors,
      errorMessages: errors.map(e => e.message),
      warningMessages: warnings.map(e => e.message),
    }
  }

  /**
   * Get the current validation result
   */
  private getResult(): ValidationResult {
    const errors = this.errors.filter(e => e.severity === 'error')
    const warnings = this.errors.filter(e => e.severity === 'warning')

    return {
      isValid: errors.length === 0,
      errors: this.errors,
      errorMessages: errors.map(e => e.message),
      warningMessages: warnings.map(e => e.message),
    }
  }

  /**
   * Validates input configuration
   */
  private validateInput(input: InputConfig): void {
    if (!input || typeof input !== 'object') {
      this.addError('input', 'Input configuration must be an object', input)
      return
    }

    const inputObj = input as any
    if (!inputObj.type || typeof inputObj.type !== 'string') {
      this.addError('input.type', 'Input type is required and must be a string', inputObj.type)
      return
    }

    if (input.type === 'file') {
      // Validate file-based input
      if (!input.path || typeof input.path !== 'string') {
        this.addError('input.path', 'Input path is required and must be a string', input.path)
        return
      }

      if (input.path.trim().length === 0) {
        this.addError('input.path', 'Input path cannot be empty', input.path)
      }

      // Validate format if specified
      if (input.format && !['csv', 'jsonl'].includes(input.format)) {
        this.addError('input.format', 'Input format must be "csv" or "jsonl"', input.format)
      }

      // Validate chunk size
      if (input.chunkSize !== undefined) {
        if (!Number.isInteger(input.chunkSize) || input.chunkSize <= 0) {
          this.addError('input.chunkSize', 'Chunk size must be a positive integer', input.chunkSize)
        } else if (input.chunkSize > 100000) {
          this.addWarning('input.chunkSize', 'Very large chunk size may impact memory usage', input.chunkSize)
        }
      }

      // Validate column mapping
      if (input.columnMapping) {
        this.validateColumnMapping(input.columnMapping)
      }

      // Validate delimiter for CSV
      if (input.format === 'csv' && input.delimiter) {
        if (input.delimiter.length !== 1) {
          this.addError('input.delimiter', 'CSV delimiter must be a single character', input.delimiter)
        }
      }
    } else if (input.type === 'provider') {
      // Validate provider-based input
      if (!input.provider || typeof input.provider !== 'string') {
        this.addError('input.provider', 'Provider name is required and must be a string', input.provider)
        return
      }

      const supportedProviders = ['coinbase', 'alpaca']
      if (!supportedProviders.includes(input.provider)) {
        this.addError('input.provider', `Provider must be one of: ${supportedProviders.join(', ')}`, input.provider)
      }

      // Validate symbols
      if (!input.symbols || !Array.isArray(input.symbols)) {
        this.addError('input.symbols', 'Symbols must be an array', input.symbols)
      } else if (input.symbols.length === 0) {
        this.addError('input.symbols', 'Symbols array cannot be empty', input.symbols)
      } else {
        input.symbols.forEach((symbol, index) => {
          if (typeof symbol !== 'string' || symbol.trim().length === 0) {
            this.addError(`input.symbols[${index}]`, 'Symbol must be a non-empty string', symbol)
          }
        })
      }

      // Validate timeframe
      if (!input.timeframe || typeof input.timeframe !== 'string') {
        this.addError('input.timeframe', 'Timeframe is required and must be a string', input.timeframe)
      } else {
        // Basic validation for timeframe format
        const timeframeRegex = /^\d+[smhd]$/
        if (!timeframeRegex.test(input.timeframe)) {
          this.addError('input.timeframe', 'Timeframe must be in format like "1m", "5m", "1h", "1d"', input.timeframe)
        }
      }

      // Validate duration if provided
      if (input.duration !== undefined) {
        if (typeof input.duration !== 'string') {
          this.addError('input.duration', 'Duration must be a string', input.duration)
        } else if (input.duration !== 'continuous') {
          const durationMatch = input.duration.match(/^(\d+)([mhdwMy]|bars)$/)
          if (!durationMatch) {
            this.addError('input.duration', 'Duration must be "continuous", or format like "5m", "1h", "7d", "1w", "3M", "1y", "1000bars"', input.duration)
          } else {
            const value = parseInt(durationMatch[1]!)
            if (value <= 0 || isNaN(value)) {
              this.addError('input.duration', 'Duration value must be a positive number', input.duration)
            }
          }
        }
      }
    } else {
      this.addError('input.type', `Invalid input type: ${inputObj.type}. Must be "file" or "provider"`, inputObj.type)
    }
  }

  /**
   * Validates column mapping configuration
   */
  private validateColumnMapping(mapping: any): void {
    const requiredFields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    for (const field of requiredFields) {
      if (!mapping[field] || typeof mapping[field] !== 'string') {
        this.addError(
          `input.columnMapping.${field}`,
          `Column mapping for "${field}" is required and must be a string`,
          mapping[field],
        )
      }
    }

    // Check for duplicate mappings
    const values = Object.values(mapping).filter(v => typeof v === 'string')
    const duplicates = values.filter((value, index) => values.indexOf(value) !== index)
    if (duplicates.length > 0) {
      this.addError(
        'input.columnMapping',
        `Duplicate column mappings found: ${duplicates.join(', ')}`,
        mapping,
      )
    }
  }

  /**
   * Validates output configuration
   */
  private validateOutput(output: any): void {
    if (!output.path || typeof output.path !== 'string') {
      this.addError('output.path', 'Output path is required and must be a string', output.path)
      return
    }

    if (output.path.trim().length === 0) {
      this.addError('output.path', 'Output path cannot be empty', output.path)
    }

    if (!output.format || !['csv', 'jsonl', 'sqlite'].includes(output.format)) {
      this.addError('output.format', 'Output format must be "csv", "jsonl", or "sqlite"', output.format)
    }

    // Validate overwrite flag
    if (output.overwrite !== undefined && typeof output.overwrite !== 'boolean') {
      this.addError('output.overwrite', 'Overwrite flag must be a boolean', output.overwrite)
    }

    // Validate column mapping for output
    if (output.columnMapping && typeof output.columnMapping !== 'object') {
      this.addError('output.columnMapping', 'Output column mapping must be an object', output.columnMapping)
    }
  }

  /**
   * Validates transformations array
   */
  private validateTransformations(transformations: TransformConfig[]): void {
    if (!Array.isArray(transformations)) {
      this.addError('transformations', 'Transformations must be an array', transformations)
      return
    }

    transformations.forEach((transform, index) => {
      this.validateSingleTransform(transform, index)
    })

    // Check for enabled transforms
    const enabledTransforms = transformations.filter(t => t.enabled)
    if (enabledTransforms.length === 0) {
      this.addWarning('transformations', 'No transformations are enabled - pipeline will only copy input to output')
    }
  }

  /**
   * Validates a single transformation
   */
  private validateSingleTransform(transform: TransformConfig, index: number): void {
    const prefix = `transformations[${index}]`

    if (!transform || typeof transform !== 'object') {
      this.addError(prefix, 'Transform must be an object', transform)
      return
    }

    // Validate type
    if (!transform.type || typeof transform.type !== 'string') {
      this.addError(`${prefix}.type`, 'Transform type is required and must be a string', transform.type)
      return
    }

    // Check if transform type is supported
    const supportedTypes = Object.keys(TRANSFORM_FIELD_MAPPINGS)
    if (!supportedTypes.includes(transform.type)) {
      this.addError(
        `${prefix}.type`,
        `Unsupported transform type "${transform.type}". Supported types: ${supportedTypes.join(', ')}`,
        transform.type,
      )
    }

    // Validate enabled flag
    if (typeof transform.enabled !== 'boolean') {
      this.addError(`${prefix}.enabled`, 'Transform enabled flag must be a boolean', transform.enabled)
    }

    // Validate params
    if (!transform.params || typeof transform.params !== 'object') {
      this.addError(`${prefix}.params`, 'Transform params are required and must be an object', transform.params)
      return
    }

    // Validate transform-specific parameters
    this.validateTransformParams(transform.type, transform.params, `${prefix}.params`)
  }

  /**
   * Validates transform-specific parameters
   */
  private validateTransformParams(type: string, params: any, prefix: string): void {
    switch (type) {
      case 'logReturns':
        if (params.outputField && typeof params.outputField !== 'string') {
          this.addError(`${prefix}.outputField`, 'Output field must be a string', params.outputField)
        }
        if (params.priceField && typeof params.priceField !== 'string') {
          this.addError(`${prefix}.priceField`, 'Price field must be a string', params.priceField)
        }
        break

      case 'zScore':
      case 'minMax':
        if (!params.fields || !Array.isArray(params.fields)) {
          this.addError(`${prefix}.fields`, 'Fields parameter must be an array', params.fields)
        } else if (params.fields.length === 0) {
          this.addError(`${prefix}.fields`, 'Fields array cannot be empty', params.fields)
        } else {
          params.fields.forEach((field: any, idx: number) => {
            if (typeof field !== 'string') {
              this.addError(`${prefix}.fields[${idx}]`, 'Field name must be a string', field)
            }
          })
        }

        if (params.windowSize !== undefined) {
          if (!Number.isInteger(params.windowSize) || params.windowSize <= 0) {
            this.addError(`${prefix}.windowSize`, 'Window size must be a positive integer', params.windowSize)
          }
        }
        break

      case 'movingAverage':
        if (!params.field || typeof params.field !== 'string') {
          this.addError(`${prefix}.field`, 'Field parameter is required and must be a string', params.field)
        }
        if (!Number.isInteger(params.windowSize) || params.windowSize <= 0) {
          this.addError(`${prefix}.windowSize`, 'Window size is required and must be a positive integer', params.windowSize)
        }
        if (params.type && !['sma', 'ema', 'wma'].includes(params.type)) {
          this.addError(`${prefix}.type`, 'Moving average type must be "sma", "ema", or "wma"', params.type)
        }
        break

      case 'rsi':
        if (params.period !== undefined && (!Number.isInteger(params.period) || params.period <= 0)) {
          this.addError(`${prefix}.period`, 'RSI period must be a positive integer', params.period)
        }
        break

      case 'priceCalc':
        if (!params.calculation || typeof params.calculation !== 'string') {
          this.addError(`${prefix}.calculation`, 'Calculation type is required', params.calculation)
        } else if (!['hlc3', 'ohlc4', 'typical', 'custom'].includes(params.calculation)) {
          this.addError(`${prefix}.calculation`, 'Invalid calculation type', params.calculation)
        }

        if (params.calculation === 'custom' && (!params.customFormula || typeof params.customFormula !== 'string')) {
          this.addError(`${prefix}.customFormula`, 'Custom formula is required for custom calculation', params.customFormula)
        }
        break
    }
  }

  /**
   * Validates processing options
   */
  private validateOptions(options?: any): void {
    if (!options) return

    if (typeof options !== 'object') {
      this.addError('options', 'Options must be an object', options)
      return
    }

    // Validate chunk size
    if (options.chunkSize !== undefined) {
      if (!Number.isInteger(options.chunkSize) || options.chunkSize <= 0) {
        this.addError('options.chunkSize', 'Chunk size must be a positive integer', options.chunkSize)
      }
    }

    // Validate boolean flags
    const booleanFields = ['continueOnError', 'showProgress']
    for (const field of booleanFields) {
      if (options[field] !== undefined && typeof options[field] !== 'boolean') {
        this.addError(`options.${field}`, `${field} must be a boolean`, options[field])
      }
    }

    // Validate max errors
    if (options.maxErrors !== undefined) {
      if (!Number.isInteger(options.maxErrors) || options.maxErrors <= 0) {
        this.addError('options.maxErrors', 'Max errors must be a positive integer', options.maxErrors)
      }
    }
  }

  /**
   * Validates metadata section
   */
  private validateMetadata(metadata?: any): void {
    if (!metadata) return

    if (typeof metadata !== 'object') {
      this.addError('metadata', 'Metadata must be an object', metadata)
      return
    }

    // Validate string fields
    const stringFields = ['name', 'version', 'description', 'author', 'created', 'modified']
    for (const field of stringFields) {
      if (metadata[field] !== undefined && typeof metadata[field] !== 'string') {
        this.addError(`metadata.${field}`, `${field} must be a string`, metadata[field])
      }
    }

    // Validate date strings
    const dateFields = ['created', 'modified']
    for (const field of dateFields) {
      if (metadata[field] && typeof metadata[field] === 'string') {
        const date = new Date(metadata[field])
        if (isNaN(date.getTime())) {
          this.addError(`metadata.${field}`, `${field} must be a valid ISO date string`, metadata[field])
        }
      }
    }
  }

  /**
   * Validates that transforms in the chain are compatible
   */
  private validateTransformChain(transformations: TransformConfig[]): void {
    // Skip validation if transformations is not a valid array
    if (!Array.isArray(transformations)) return

    const enabledTransforms = transformations.filter(t => t && t.enabled)
    if (enabledTransforms.length === 0) return

    const availableFields = new Set(BASE_OHLCV_FIELDS)

    for (let i = 0; i < enabledTransforms.length; i++) {
      const transform = enabledTransforms[i]!
      const mapping = TRANSFORM_FIELD_MAPPINGS[transform.type]

      if (!mapping) continue

      // Get required fields for this transform
      const requiredFields = this.getTransformRequiredFields(transform)

      // Check if all required fields are available
      for (const field of requiredFields) {
        if (!availableFields.has(field)) {
          this.addError(
            `transformations[${transformations.indexOf(transform)}]`,
            `Transform "${transform.type}" requires field "${field}" which is not available. Available fields: ${Array.from(availableFields).join(', ')}`,
            transform,
          )
        }
      }

      // Add output fields from this transform to available fields
      const outputFields = this.getTransformOutputFields(transform)
      for (const field of outputFields) {
        availableFields.add(field)
      }
    }
  }

  /**
   * Gets the required fields for a specific transform instance
   */
  private getTransformRequiredFields(transform: TransformConfig): string[] {
    const mapping = TRANSFORM_FIELD_MAPPINGS[transform.type]
    if (!mapping) return []

    // For transforms with dynamic field requirements
    const params = transform.params as any
    switch (transform.type) {
      case 'zScore':
      case 'minMax':
        return params.fields || []
      case 'movingAverage':
        return params.field ? [params.field] : []
      default:
        return mapping.requiredFields
    }
  }

  /**
   * Gets the output fields for a specific transform instance
   */
  private getTransformOutputFields(transform: TransformConfig): string[] {
    const mapping = TRANSFORM_FIELD_MAPPINGS[transform.type]
    if (!mapping) return []

    // For transforms with dynamic output fields
    const params = transform.params as any
    switch (transform.type) {
      case 'zScore':
        const fields = params.fields || []
        return fields.map((field: string) => `${field}_zscore`)
      case 'minMax':
        const minMaxFields = params.fields || []
        return minMaxFields.map((field: string) => `${field}_norm`)
      case 'movingAverage':
        const field = params.field
        const maType = params.type || 'sma'
        return field ? [`${field}_${maType}`] : []
      case 'priceCalc':
        const calc = params.calculation
        if (calc === 'hlc3') return ['hlc3']
        if (calc === 'ohlc4') return ['ohlc4']
        if (calc === 'typical') return ['typical']
        return ['custom_price']
      default:
        return mapping.outputFields
    }
  }

  /**
   * Validates compatibility between input and output configurations
   */
  private validateInputOutputCompatibility(input: InputConfig, output: any): void {
    // Check if output path conflicts with input path (only for file inputs)
    if (input.type === 'file' && input.path === output.path) {
      this.addError(
        'output.path',
        'Output path cannot be the same as input path',
        output.path,
      )
    }

    // Check format compatibility
    if (input.type === 'file' && input.format === 'csv' && output.format === 'sqlite') {
      this.addWarning(
        'output.format',
        'Converting from CSV to SQLite - ensure sufficient memory for large files',
      )
    }

    // Warn about continuous mode with file output
    if (input.type === 'provider' && input.duration === 'continuous' && output.format !== 'sqlite') {
      this.addWarning(
        'output.format',
        'Continuous data collection is better suited for SQLite output format',
      )
    }
  }

  /**
   * Adds an error to the validation result
   */
  private addError(field: string, message: string, value?: any): void {
    this.errors.push({
      field,
      message,
      value,
      severity: 'error',
    })
  }

  /**
   * Adds a warning to the validation result
   */
  private addWarning(field: string, message: string, value?: any): void {
    this.errors.push({
      field,
      message,
      value,
      severity: 'warning',
    })
  }
}

/**
 * Validates a pipeline configuration and returns the result
 * Convenience function for one-off validation
 */
export function validatePipelineConfig(config: PipelineConfig): ValidationResult {
  const validator = new ConfigValidator()
  return validator.validate(config)
}
