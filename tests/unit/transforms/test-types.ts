import { OhlcvDto } from '../../../src/models'

/**
 * Type for OHLCV data with additional transform fields
 * Used in tests to access transform output fields without TypeScript errors
 */
export type TransformedOhlcvDto = OhlcvDto & {
  // Log returns fields
  close_log_return?: number
  open_log_return?: number
  high_log_return?: number
  low_log_return?: number
  high_return?: number
  
  // Z-score fields
  open_zscore?: number
  high_zscore?: number
  low_zscore?: number
  close_zscore?: number
  volume_zscore?: number
  close_z?: number
  
  // Min-max normalized fields
  open_norm?: number
  high_norm?: number
  low_norm?: number
  close_norm?: number
  volume_norm?: number
  close_scaled?: number
  close_01?: number
}