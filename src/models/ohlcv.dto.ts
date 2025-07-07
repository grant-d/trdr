/**
 * OHLCV (Open, High, Low, Close, Volume) data transfer object
 * Represents a single candlestick/bar of market data
 */
export interface OhlcvDto {
  /** Exchange where the data originates from (e.g., 'coinbase', 'binance') */
  // exchange: string

  /** Trading pair symbol (e.g., 'BTC-USD', 'ETH-USD') */
  // symbol: string

  /** Unix timestamp in milliseconds (UTC) */
  timestamp: number

  /** Opening price for the time period */
  open: number

  /** Highest price during the time period */
  high: number

  /** Lowest price during the time period */
  low: number

  /** Closing price for the time period */
  close: number

  /** Volume traded during the time period */
  volume: number

  /** Additional columns that can be added by transforms */
  [key: string]: number | string
}

/**
 * Validates that an OHLCV data object has valid values
 * @param data The OHLCV data to validate
 * @returns true if valid, false otherwise
 */
export function isValidOhlcv(data: Partial<OhlcvDto>): boolean {
  // Check required fields exist
  if (!data.timestamp) {
    return false
  }

  if (!data.open || !data.high || !data.low || !data.close || !data.volume) {
    return false
  }

  // Check numeric fields are valid numbers
  if (
    isNaN(data.timestamp) ||
    isNaN(data.open) ||
    isNaN(data.high) ||
    isNaN(data.low) ||
    isNaN(data.close) ||
    isNaN(data.volume)
  ) {
    return false
  }

  // Check OHLC relationships
  if (data.high < data.low) {
    return false
  }

  if (data.high < data.open || data.high < data.close) {
    return false
  }

  if (data.low > data.open || data.low > data.close) {
    return false
  }

  // Check non-negative values
  if (
    data.open < 0 ||
    data.high < 0 ||
    data.low < 0 ||
    data.close < 0 ||
    data.volume < 0
  ) {
    return false
  }

  // Check timestamp is reasonable (between year 2000 and 2100)
  const year2000 = 946684800000
  const year2100 = 4102444800000
  return !(data.timestamp < year2000 || data.timestamp > year2100)
}

/**
 * Formats an OHLCV data object as a string for logging
 * @param data The OHLCV data to format
 * @returns Formatted string representation
 */
export function formatOhlcv(data: OhlcvDto): string {
  const date = new Date(data.timestamp).toISOString()
  return `${date} O:${data.open} H:${data.high} L:${data.low} C:${data.close} V:${data.volume}`
}

/**
 * Creates a copy of OHLCV data with only the standard fields
 * Removes any additional fields added by transforms
 * @param data The OHLCV data to clean
 * @returns Clean OHLCV data with only standard fields
 */
export function cleanOhlcv(data: OhlcvDto): OhlcvDto {
  return {
    timestamp: data.timestamp,
    open: data.open,
    high: data.high,
    low: data.low,
    close: data.close,
    volume: data.volume
  }
}
