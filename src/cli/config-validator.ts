import type { PipelineConfig } from '../interfaces'

/**
 * Validated configuration is just the PipelineConfig type
 * The actual validation happens in each transform's constructor
 */
export type ValidatedConfig = PipelineConfig;

/**
 * Config validator that leverages transform-specific validation
 */
export class ConfigValidator {
  /**
   * Validate a pipeline configuration
   * @param config Raw configuration object
   * @returns Validated configuration
   * @throws Error if validation fails
   */
  public static validate(config: unknown): ValidatedConfig {
    // Basic structure validation
    if (!config || typeof config !== 'object') {
      throw new Error('Configuration must be an object')
    }

    const cfg = config as any

    // Validate required fields
    if (!cfg.input || typeof cfg.input !== 'object') {
      throw new Error('Configuration must have an input section')
    }

    if (!cfg.output || typeof cfg.output !== 'object') {
      throw new Error('Configuration must have an output section')
    }

    if (!Array.isArray(cfg.transformations)) {
      throw new Error('Configuration must have a transformations array')
    }

    // Validate input
    if (cfg.input.type === 'file') {
      if (!cfg.input.path) {
        throw new Error('File input requires a path')
      }
    } else if (cfg.input.type === 'provider') {
      if (!cfg.input.provider) {
        throw new Error('Provider input requires a provider name')
      }
      if (!Array.isArray(cfg.input.symbols) || cfg.input.symbols.length === 0) {
        throw new Error('Provider input requires symbols array')
      }
      if (!cfg.input.timeframe) {
        throw new Error('Provider input requires a timeframe')
      }
    } else {
      throw new Error(`Invalid input type: ${cfg.input.type}`)
    }

    // Validate output
    if (!cfg.output.path) {
      throw new Error('Output requires a path')
    }
    if (!cfg.output.format || !['csv', 'jsonl'].includes(cfg.output.format)) {
      throw new Error('Output format must be csv or jsonl')
    }

    // Return as ValidatedConfig - transform validation happens during instantiation
    return cfg as ValidatedConfig
  }

  /**
   * Validate with details (for backward compatibility)
   * @param config Raw configuration object
   * @returns Object with validated config and any warnings
   */
  public static validateWithDetails(config: unknown): {
    config: ValidatedConfig;
    warnings: string[];
    isValid: boolean;
  } {
    try {
      const validated = this.validate(config)
      const warnings: string[] = []

      // Add warnings
      if (validated.transformations.length === 0) {
        warnings.push('No transformations configured')
      }

      return {
        config: validated,
        warnings,
        isValid: true
      }
    } catch (error) {
      throw error
    }
  }

  /**
   * Load and validate a configuration file
   * @param filePath Path to configuration file
   * @returns Validated configuration
   */
  public static async loadAndValidate(
    filePath: string
  ): Promise<ValidatedConfig> {
    const { ConfigLoader } = await import('./config-loader')
    const rawConfig = await ConfigLoader.load(filePath)
    return this.validate(rawConfig)
  }
}
