/**
 * Simple logger interface for the trading system
 */
export interface Logger {
  debug(message: string, context?: Record<string, any>): void
  info(message: string, context?: Record<string, any>): void
  warn(message: string, context?: Record<string, any>): void
  error(message: string, context?: Record<string, any>): void
}

/**
 * Console-based logger implementation
 */
export class ConsoleLogger implements Logger {
  constructor(private readonly name: string) {}

  debug(message: string, context?: Record<string, any>): void {
    console.debug(`[${this.name}] ${message}`, context)
  }

  info(message: string, context?: Record<string, any>): void {
    console.info(`[${this.name}] ${message}`, context)
  }

  warn(message: string, context?: Record<string, any>): void {
    console.warn(`[${this.name}] ${message}`, context)
  }

  error(message: string, context?: Record<string, any>): void {
    console.error(`[${this.name}] ${message}`, context)
  }
}

/**
 * No-op logger for testing
 */
export class NoopLogger implements Logger {
  debug(_message: string, _context?: Record<string, any>): void {}
  info(_message: string, _context?: Record<string, any>): void {}
  warn(_message: string, _context?: Record<string, any>): void {}
  error(_message: string, _context?: Record<string, any>): void {}
}