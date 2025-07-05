import type { PipelineConfig } from '../interfaces'

/**
 * Error thrown when configuration override fails
 */
export class ConfigOverrideError extends Error {
  constructor(message: string, public readonly override: string) {
    super(message)
    this.name = 'ConfigOverrideError'
  }
}

/**
 * Represents a parsed override with its path and value
 */
interface ParsedOverride {
  /** Dot-notation path to the property */
  path: string[]
  /** Raw string value from command line */
  value: string
  /** Original override string for error reporting */
  original: string
}

/**
 * Parses a single override string into path components and value
 * Supports dot notation like "input.path=/new/path" or "options.chunkSize=1000"
 */
function parseOverride(override: string): ParsedOverride {
  const equalIndex = override.indexOf('=')
  if (equalIndex === -1) {
    throw new ConfigOverrideError(
      `Invalid override syntax: "${override}". Expected format: "path.to.property=value"`,
      override
    )
  }

  const pathString = override.substring(0, equalIndex).trim()
  const value = override.substring(equalIndex + 1)

  if (!pathString) {
    throw new ConfigOverrideError(
      `Invalid override syntax: "${override}". Property path cannot be empty`,
      override
    )
  }

  const path = pathString.split('.').map(segment => segment.trim()).filter(segment => segment.length > 0)

  if (path.length === 0) {
    throw new ConfigOverrideError(
      `Invalid override syntax: "${override}". Property path cannot be empty`,
      override
    )
  }

  return {
    path,
    value,
    original: override
  }
}

/**
 * Converts a string value to the appropriate type based on context
 * Handles common data types: boolean, number, string, arrays
 */
function convertValue(value: string, _path: string[], original: string): unknown {
  // Handle null/undefined
  if (value.toLowerCase() === 'null') return null
  if (value.toLowerCase() === 'undefined') return undefined

  // Handle boolean values
  if (value.toLowerCase() === 'true') return true
  if (value.toLowerCase() === 'false') return false

  // Handle array values (JSON-like syntax)
  if (value.startsWith('[') && value.endsWith(']')) {
    try {
      return JSON.parse(value)
    } catch (error) {
      throw new ConfigOverrideError(
        `Invalid array value in override "${original}": ${error instanceof Error ? error.message : 'Invalid JSON'}`,
        original
      )
    }
  }

  // Handle object values (JSON-like syntax)
  if (value.startsWith('{') && value.endsWith('}')) {
    try {
      return JSON.parse(value)
    } catch (error) {
      throw new ConfigOverrideError(
        `Invalid object value in override "${original}": ${error instanceof Error ? error.message : 'Invalid JSON'}`,
        original
      )
    }
  }

  // Handle numeric values
  if (/^-?\d+$/.test(value)) {
    return parseInt(value, 10)
  }
  if (/^-?\d*\.\d+$/.test(value)) {
    return parseFloat(value)
  }

  // Handle scientific notation
  if (/^-?\d*\.?\d+[eE][+-]?\d+$/.test(value)) {
    return parseFloat(value)
  }

  // Return as string for everything else
  return value
}

/**
 * Sets a nested property value using dot notation path
 * Creates intermediate objects as needed
 */
function setNestedProperty(obj: PipelineConfig, path: string[], value: unknown, original: string): void {
  if (path.length === 0) {
    throw new ConfigOverrideError(`Empty property path in override "${original}"`, original)
  }

  let current = obj as any

  // Navigate to the parent of the target property
  for (let i = 0; i < path.length - 1; i++) {
    const segment = path[i]!

    if (current[segment] === undefined || current[segment] === null) {
      // Create intermediate object
      current[segment] = {}
    } else if (typeof current[segment] !== 'object') {
      throw new ConfigOverrideError(
        `Cannot override property "${path.slice(0, i + 1).join('.')}" in "${original}": intermediate value is not an object`,
        original,
      )
    }

    current = current[segment]
  }

  // Set the final property
  const finalKey = path[path.length - 1]!
  current[finalKey] = value
}

/**
 * Validates that an override path points to a valid configuration property
 * This helps catch typos and invalid property names early
 */
function validateOverridePath(path: string[], original: string): void {
  const validRootProperties = ['input', 'output', 'transformations', 'options', 'metadata']

  if (path.length === 0) {
    throw new ConfigOverrideError(`Empty property path in override "${original}"`, original)
  }

  const rootProperty = path[0]!
  if (!validRootProperties.includes(rootProperty)) {
    throw new ConfigOverrideError(
      `Invalid root property "${rootProperty}" in override "${original}". Valid root properties: ${validRootProperties.join(', ')}`,
      original
    )
  }

  // Validate specific property paths
  switch (rootProperty) {
    case 'input':
      validateInputOverridePath(path.slice(1), original)
      break
    case 'output':
      validateOutputOverridePath(path.slice(1), original)
      break
    case 'transformations':
      validateTransformationOverridePath(path.slice(1), original)
      break
    case 'options':
      validateOptionsOverridePath(path.slice(1), original)
      break
    case 'metadata':
      validateMetadataOverridePath(path.slice(1), original)
      break
  }
}

/**
 * Validates input-specific override paths
 */
function validateInputOverridePath(path: string[], original: string): void {
  if (path.length === 0) return // Allow overriding entire input object

  const validInputProperties = [
    'path', 'format', 'columnMapping', 'chunkSize', 'exchange', 'symbol', 'delimiter',
  ]

  const property = path[0]!
  if (!validInputProperties.includes(property)) {
    throw new ConfigOverrideError(
      `Invalid input property "${property}" in override "${original}". Valid input properties: ${validInputProperties.join(', ')}`,
      original
    )
  }

  // Validate column mapping paths
  if (property === 'columnMapping' && path.length > 1) {
    const validColumnFields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'exchange']
    const field = path[1]!
    if (!validColumnFields.includes(field)) {
      throw new ConfigOverrideError(
        `Invalid column mapping field "${field}" in override "${original}". Valid fields: ${validColumnFields.join(', ')}`,
        original
      )
    }
  }
}

/**
 * Validates output-specific override paths
 */
function validateOutputOverridePath(path: string[], original: string): void {
  if (path.length === 0) return // Allow overriding entire output object

  const validOutputProperties = ['path', 'format', 'overwrite', 'columnMapping']

  const property = path[0]!
  if (!validOutputProperties.includes(property)) {
    throw new ConfigOverrideError(
      `Invalid output property "${property}" in override "${original}". Valid output properties: ${validOutputProperties.join(', ')}`,
      original
    )
  }
}

/**
 * Validates transformation-specific override paths
 */
function validateTransformationOverridePath(path: string[], original: string): void {
  if (path.length === 0) return // Allow overriding entire transformations array

  // First segment should be array index
  const indexStr = path[0]!
  if (!/^\d+$/.test(indexStr)) {
    throw new ConfigOverrideError(
      `Invalid transformation index "${indexStr}" in override "${original}". Expected numeric index`,
      original
    )
  }

  if (path.length > 1) {
    const validTransformProperties = ['type', 'disabled', 'params']
    const property = path[1]!
    if (!validTransformProperties.includes(property)) {
      throw new ConfigOverrideError(
        `Invalid transformation property "${property}" in override "${original}". Valid properties: ${validTransformProperties.join(', ')}`,
        original
      )
    }
  }
}

/**
 * Validates options-specific override paths
 */
function validateOptionsOverridePath(path: string[], original: string): void {
  if (path.length === 0) return // Allow overriding entire options object

  const validOptionsProperties = [
    'chunkSize', 'continueOnError', 'maxErrors', 'showProgress'
  ]

  const property = path[0]!
  if (!validOptionsProperties.includes(property)) {
    throw new ConfigOverrideError(
      `Invalid options property "${property}" in override "${original}". Valid options properties: ${validOptionsProperties.join(', ')}`,
      original
    )
  }
}

/**
 * Validates metadata-specific override paths
 */
function validateMetadataOverridePath(path: string[], original: string): void {
  if (path.length === 0) return // Allow overriding entire metadata object

  const validMetadataProperties = [
    'name', 'version', 'description', 'author', 'created', 'modified'
  ]

  const property = path[0]!
  if (!validMetadataProperties.includes(property)) {
    throw new ConfigOverrideError(
      `Invalid metadata property "${property}" in override "${original}". Valid metadata properties: ${validMetadataProperties.join(', ')}`,
      original,
    )
  }
}

/**
 * Applies an array of override strings to a pipeline configuration
 * Modifies the configuration object in place
 *
 * @param config The pipeline configuration to modify
 * @param overrides Array of override strings in dot notation format
 * @throws ConfigOverrideError if any override is invalid
 *
 * @example
 * ```typescript
 * const config = loadPipelineConfig('pipeline.json')
 * applyOverrides(config, [
 *   'input.path=/new/data.csv',
 *   'options.chunkSize=2000',
 *   'transformations.0.enabled=false'
 * ])
 * ```
 */
export function applyOverrides(config: PipelineConfig, overrides: string[]): void {
  if (!overrides || overrides.length === 0) {
    return // Nothing to override
  }

  // Parse all overrides first to catch syntax errors early
  const parsedOverrides: ParsedOverride[] = []
  for (const override of overrides) {
    if (!override || override.trim().length === 0) {
      continue // Skip empty overrides
    }

    try {
      parsedOverrides.push(parseOverride(override.trim()))
    } catch (error) {
      if (error instanceof ConfigOverrideError) {
        throw error
      }
      throw new ConfigOverrideError(
        `Failed to parse override "${override}": ${error instanceof Error ? error.message : 'Unknown error'}`,
        override
      )
    }
  }

  // Validate all override paths
  for (const parsed of parsedOverrides) {
    validateOverridePath(parsed.path, parsed.original)
  }

  // Apply overrides
  for (const parsed of parsedOverrides) {
    try {
      const convertedValue = convertValue(parsed.value, parsed.path, parsed.original)
      setNestedProperty(config, parsed.path, convertedValue, parsed.original)
    } catch (error) {
      if (error instanceof ConfigOverrideError) {
        throw error
      }
      throw new ConfigOverrideError(
        `Failed to apply override "${parsed.original}": ${error instanceof Error ? error.message : 'Unknown error'}`,
        parsed.original
      )
    }
  }
}

/**
 * Creates a deep copy of a configuration object
 * Useful for testing overrides without modifying the original
 */
export function cloneConfig(config: PipelineConfig): PipelineConfig {
  return JSON.parse(JSON.stringify(config))
}

/**
 * Gets the current value at a given path in the configuration
 * Useful for debugging and testing
 */
export function getConfigValue(config: PipelineConfig, path: string): unknown {
  const pathArray = path.split('.').filter(segment => segment.length > 0)
  let current = config as any

  for (const segment of pathArray) {
    if (current === null || current === undefined || typeof current !== 'object') {
      return undefined
    }
    current = current[segment]
  }

  return current
}

/**
 * ConfigOverrides class for compatibility
 */
export class ConfigOverrides {
  /**
   * Apply overrides to configuration
   */
  public static apply(config: PipelineConfig, overrides: string[]): PipelineConfig {
    const cloned = cloneConfig(config)
    applyOverrides(cloned, overrides)
    return cloned
  }

  /**
   * Get value at path
   */
  public static getValue(config: PipelineConfig, path: string): unknown {
    return getConfigValue(config, path)
  }

  /**
   * Clone configuration
   */
  public static clone(config: PipelineConfig): PipelineConfig {
    return cloneConfig(config)
  }
}
