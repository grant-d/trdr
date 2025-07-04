import * as path from 'node:path'
import { CsvFileProvider } from './csv-file-provider'
import type { FileProvider } from './file-provider.base'
import { JsonlFileProvider } from './jsonl-file-provider'
import type { FileProviderConfig } from './types'

/**
 * Factory for creating file providers based on format
 */
export function createFileProvider(config: FileProviderConfig): FileProvider {
  const format = config.format || path.extname(config.path).toLowerCase().slice(1)

  switch (format) {
    case 'csv':
      return new CsvFileProvider(config)
    case 'jsonl':
    case 'pq':
      return new JsonlFileProvider(config)
    default:
      throw new Error(`Unsupported file format: ${format}`)
  }
}
