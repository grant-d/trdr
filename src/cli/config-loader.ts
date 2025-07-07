import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { isAbsolute, resolve } from 'node:path'
import { isPipelineConfig, type PipelineConfig } from '../interfaces'

/**
 * Error thrown when configuration loading fails
 */
export class ConfigLoadError extends Error {
  constructor(
    message: string,
    public readonly filePath: string,
    public readonly cause?: Error
  ) {
    super(message)
    this.name = 'ConfigLoadError'
  }
}

/**
 * Error thrown when configuration validation fails
 */
export class ConfigValidationError extends Error {
  constructor(
    message: string,
    public readonly filePath: string
  ) {
    super(message)
    this.name = 'ConfigValidationError'
  }
}

/**
 * Expands environment variables in a string
 * Supports ${VAR_NAME} and $VAR_NAME syntax
 */
function expandEnvironmentVariables(str: string): string {
  return str
    .replace(/\$\{([^}]+)\}/g, (_, varName) => {
      return process.env[varName] || ''
    })
    .replace(/\$([A-Z_][A-Z0-9_]*)/gi, (_, varName) => {
      return process.env[varName] || ''
    })
}

/**
 * Recursively expands environment variables in an object
 */
function expandObjectEnvironmentVariables(obj: unknown): unknown {
  if (typeof obj === 'string') {
    return expandEnvironmentVariables(obj)
  }

  if (Array.isArray(obj)) {
    return obj.map(expandObjectEnvironmentVariables)
  }

  if (obj && typeof obj === 'object') {
    const expanded: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(obj)) {
      expanded[key] = expandObjectEnvironmentVariables(value)
    }
    return expanded
  }

  return obj
}

/**
 * Validates a pipeline configuration object
 */
function validatePipelineConfig(
  config: unknown,
  filePath: string
): asserts config is PipelineConfig {
  if (!isPipelineConfig(config)) {
    throw new ConfigValidationError(
      'Configuration must be a valid JSON object with input, output, and transformations sections',
      filePath
    )
  }

  // Validate input section
  if (!config.input) {
    throw new ConfigValidationError(
      'Configuration must have an "input" section',
      filePath
    )
  }

  // Check input type
  if (!config.input.type || typeof config.input.type !== 'string') {
    throw new ConfigValidationError(
      'Input configuration must have a valid "type" string',
      filePath
    )
  }

  // Validate based on input type
  if (config.input.type === 'file') {
    if (!config.input.path || typeof config.input.path !== 'string') {
      throw new ConfigValidationError(
        'File input configuration must have a valid "path" string',
        filePath
      )
    }
  } else if (config.input.type === 'provider') {
    if (!config.input.provider || typeof config.input.provider !== 'string') {
      throw new ConfigValidationError(
        'Provider input configuration must have a valid "provider" string',
        filePath
      )
    }
    if (!config.input.symbols || !Array.isArray(config.input.symbols)) {
      throw new ConfigValidationError(
        'Provider input configuration must have a "symbols" array',
        filePath
      )
    }
    if (!config.input.timeframe || typeof config.input.timeframe !== 'string') {
      throw new ConfigValidationError(
        'Provider input configuration must have a valid "timeframe" string',
        filePath
      )
    }
  } else {
    throw new ConfigValidationError(
      `Invalid input type: ${(config.input as any).type}. Must be "file" or "provider"`,
      filePath
    )
  }

  // Validate output section
  if (!config.output) {
    throw new ConfigValidationError(
      'Configuration must have an "output" section',
      filePath
    )
  }

  if (!config.output.path || typeof config.output.path !== 'string') {
    throw new ConfigValidationError(
      'Output configuration must have a valid "path" string',
      filePath
    )
  }

  if (
    !config.output.format ||
    !['csv', 'jsonl'].includes(config.output.format)
  ) {
    throw new ConfigValidationError(
      'Output format must be one of: csv, jsonl',
      filePath
    )
  }

  // Validate transformations section
  if (!Array.isArray(config.transformations)) {
    throw new ConfigValidationError(
      'Configuration must have a "transformations" array',
      filePath
    )
  }

  // Validate each transformation
  for (let i = 0; i < config.transformations.length; i++) {
    const transform = config.transformations[i]
    if (!transform || typeof transform !== 'object') {
      throw new ConfigValidationError(
        `Transformation ${i} must be a valid object`,
        filePath
      )
    }

    if (!transform.type || typeof transform.type !== 'string') {
      throw new ConfigValidationError(
        `Transformation ${i} must have a valid "type" string`,
        filePath
      )
    }

    // disabled is optional - if not present, defaults to false
    if (
      transform.hasOwnProperty('disabled') &&
      typeof transform.disabled !== 'boolean'
    ) {
      throw new ConfigValidationError(
        `Transformation ${i} disabled field must be a boolean if present`,
        filePath
      )
    }

    if (!transform.params || typeof transform.params !== 'object') {
      throw new ConfigValidationError(
        `Transformation ${i} must have a "params" object`,
        filePath
      )
    }
  }

  // Validate options section (optional but if present must be valid)
  if (config.options) {
    if (typeof config.options !== 'object') {
      throw new ConfigValidationError(
        'Options must be a valid object',
        filePath
      )
    }

    if (
      config.options.chunkSize !== undefined &&
      (!Number.isInteger(config.options.chunkSize) ||
        config.options.chunkSize <= 0)
    ) {
      throw new ConfigValidationError(
        'Options chunkSize must be a positive integer',
        filePath
      )
    }

    if (
      config.options.maxErrors !== undefined &&
      (!Number.isInteger(config.options.maxErrors) ||
        config.options.maxErrors <= 0)
    ) {
      throw new ConfigValidationError(
        'Options maxErrors must be a positive integer',
        filePath
      )
    }
  }
}

/**
 * Loads and parses a pipeline configuration file
 * @param configPath Path to the configuration file (absolute or relative)
 * @returns Parsed and validated pipeline configuration
 * @throws ConfigLoadError if file cannot be read
 * @throws ConfigValidationError if configuration is invalid
 */
export async function loadPipelineConfig(
  configPath: string
): Promise<PipelineConfig> {
  // Resolve path (convert relative to absolute)
  const resolvedPath = isAbsolute(configPath)
    ? configPath
    : resolve(process.cwd(), configPath)

  // Check if file exists
  if (!existsSync(resolvedPath)) {
    throw new ConfigLoadError(
      `Configuration file not found: ${resolvedPath}`,
      resolvedPath
    )
  }

  let rawContent: string
  try {
    rawContent = await readFile(resolvedPath, 'utf-8')
  } catch (error) {
    throw new ConfigLoadError(
      `Failed to read configuration file: ${error instanceof Error ? error.message : 'Unknown error'}`,
      resolvedPath,
      error instanceof Error ? error : undefined
    )
  }

  // Parse JSON
  let parsedConfig: any
  try {
    parsedConfig = JSON.parse(rawContent)
  } catch (error) {
    throw new ConfigLoadError(
      `Failed to parse JSON configuration: ${error instanceof Error ? error.message : 'Invalid JSON'}`,
      resolvedPath,
      error instanceof Error ? error : undefined
    )
  }

  // Expand environment variables
  const expandedConfig = expandObjectEnvironmentVariables(parsedConfig)

  // Validate configuration structure
  validatePipelineConfig(expandedConfig, resolvedPath)

  return expandedConfig
}

/**
 * Creates a default pipeline configuration
 * Useful for initialization or as a template
 */
export function createDefaultPipelineConfig(): PipelineConfig {
  return {
    input: {
      type: 'file',
      path: './data/input.csv',
      format: 'csv',
      chunkSize: 1000
    },
    output: {
      path: './data/output.jsonl',
      format: 'jsonl',
      overwrite: true
    },
    transformations: [],
    options: {
      chunkSize: 1000,
      continueOnError: true,
      maxErrors: 100,
      showProgress: true
    },
    metadata: {
      name: 'Default Pipeline',
      version: '1.0.0',
      description: 'Default pipeline configuration template',
      created: new Date().toISOString()
    }
  }
}

/**
 * ConfigLoader class for compatibility
 */
export class ConfigLoader {
  /**
   * Load pipeline configuration from file
   */
  public static async load(configPath: string): Promise<PipelineConfig> {
    return loadPipelineConfig(configPath)
  }

  /**
   * Create default configuration
   */
  public static createDefault(): PipelineConfig {
    return createDefaultPipelineConfig()
  }
}
