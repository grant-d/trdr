/**
 * Branded string type for ISO 8601 date strings.
 * 
 * This type ensures type safety when working with ISO date strings
 * throughout the application. The format is: YYYY-MM-DDTHH:mm:ss.sssZ
 * 
 * @example
 * const date: IsoDate = '2024-01-15T12:30:45.123Z' as IsoDate
 */
export type IsoDate = string & { readonly __brand: 'IsoDate' }

/**
 * Branded number type for epoch timestamps in milliseconds since 1970-01-01.
 * 
 * This type ensures type safety when working with epoch timestamps
 * throughout the application. Values are stored as numbers representing
 * milliseconds since the Unix epoch (January 1, 1970 00:00:00 UTC).
 * 
 * @example
 * const timestamp: EpochDate = 1705321845123 as EpochDate
 */
export type EpochDate = number & { readonly __brand: 'EpochDate' }

/**
 * Converts a Date object or timestamp to an ISO 8601 formatted date string.
 * 
 * @param value - A Date object or numeric timestamp to convert
 * @param precision - Optional precision for numeric timestamps: 'ms' (milliseconds, default) or 's' (seconds)
 * @returns An ISO 8601 formatted date string as IsoDate
 * 
 * @example
 * // From Date object
 * const date = new Date('2024-01-15T12:30:45.123Z')
 * const isoDate = toIsoDate(date) // '2024-01-15T12:30:45.123Z'
 * 
 * @example
 * // From milliseconds timestamp
 * const isoDate = toIsoDate(1705321845123) // '2024-01-15T12:30:45.123Z'
 * 
 * @example
 * // From seconds timestamp
 * const isoDate = toIsoDate(1705321845, 's') // '2024-01-15T12:30:45.000Z'
 */
export function toIsoDate(value: Date): IsoDate
export function toIsoDate(value: number, precision?: 'ms' | 's'): IsoDate
export function toIsoDate(value: Date | number, precision?: 'ms' | 's'): IsoDate {
  if (typeof value === 'number') {
    value = new Date(precision === 's' ? value * 1000 : value)
  }
  return value.toISOString() as IsoDate
}

/**
 * Converts a Date object or timestamp to an epoch timestamp in milliseconds.
 * 
 * @param value - A Date object or numeric timestamp to convert
 * @param precision - Optional precision for numeric timestamps: 'ms' (milliseconds, default) or 's' (seconds)
 * @returns A numeric timestamp in milliseconds since Unix epoch as EpochDate
 * 
 * @example
 * // From Date object
 * const date = new Date('2024-01-15T12:30:45.123Z')
 * const epoch = toEpochDate(date) // 1705321845123
 * 
 * @example
 * // From milliseconds (pass-through)
 * const epoch = toEpochDate(1705321845123) // 1705321845123
 * 
 * @example
 * // From seconds timestamp
 * const epoch = toEpochDate(1705321845, 's') // 1705321845000
 */
export function toEpochDate(value: Date): EpochDate
export function toEpochDate(value: number, precision?: 'ms' | 's'): EpochDate
export function toEpochDate(value: Date | number, precision?: 'ms' | 's'): EpochDate {
  if (typeof value === 'number') {
    return (precision === 's' ? value * 1000 : value) as EpochDate
  }
  return value.getTime() as EpochDate
}

/**
 * Converts an ISO 8601 date string to a JavaScript Date object.
 * 
 * @param isoDate - An ISO 8601 formatted date string
 * @returns A JavaScript Date object
 * 
 * @example
 * const isoDate = '2024-01-15T12:30:45.123Z' as IsoDate
 * const date = fromIsoDate(isoDate) // Date object
 * console.log(date.getFullYear()) // 2024
 */
export function fromIsoDate(isoDate: IsoDate): Date {
  return new Date(isoDate)
}

/**
 * Converts an epoch timestamp to a JavaScript Date object.
 * 
 * @param epochDate - A numeric timestamp in milliseconds since Unix epoch
 * @returns A JavaScript Date object
 * 
 * @example
 * const epoch = 1705321845123 as EpochDate
 * const date = fromEpochDate(epoch) // Date object
 * console.log(date.toISOString()) // '2024-01-15T12:30:45.123Z'
 */
export function fromEpochDate(epochDate: EpochDate): Date {
  return new Date(epochDate)
}

/**
 * Converts an ISO 8601 date string to an epoch timestamp.
 * 
 * This is a convenience function that combines fromIsoDate and toEpochDate.
 * 
 * @param isoDate - An ISO 8601 formatted date string
 * @returns A numeric timestamp in milliseconds since Unix epoch
 * 
 * @example
 * const isoDate = '2024-01-15T12:30:45.123Z' as IsoDate
 * const epoch = isoToEpoch(isoDate) // 1705321845123
 */
export function isoToEpoch(isoDate: IsoDate): EpochDate {
  return toEpochDate(new Date(isoDate))
}

/**
 * Converts an epoch timestamp to an ISO 8601 date string.
 * 
 * This is a convenience function that combines fromEpochDate and toIsoDate.
 * 
 * @param epochDate - A numeric timestamp in milliseconds since Unix epoch
 * @returns An ISO 8601 formatted date string
 * 
 * @example
 * const epoch = 1705321845123 as EpochDate
 * const isoDate = epochToIso(epoch) // '2024-01-15T12:30:45.123Z'
 */
export function epochToIso(epochDate: EpochDate): IsoDate {
  return toIsoDate(new Date(epochDate))
}
