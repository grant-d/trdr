import * as path from 'node:path'
import { CsvFileProvider } from './csv-file-provider'
import type { FileProvider } from './file-provider.base'
import { ParquetFileProvider } from './parquet-file-provider'
import type { FileProviderConfig } from './types'

/**
 * Factory for creating file providers based on format
 */
export async function createFileProvider(config: FileProviderConfig): Promise<FileProvider> {
  const format = config.format || path.extname(config.path).toLowerCase().slice(1)
  
  switch (format) {
    case 'csv':
      return new CsvFileProvider(config)
    case 'parquet':
    case 'pq':
      return new ParquetFileProvider(config)
    default:
      throw new Error(`Unsupported file format: ${format}`)
  }
}
