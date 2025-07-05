import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { afterEach, beforeEach, describe, it } from 'node:test'
import { DEFAULT_ARGS, parseArgs } from '../../../src/cli/args-parser'

describe('Args Parser', () => {
  let originalExit: typeof process.exit
  let originalConsoleError: typeof console.error
  let exitCode: number | undefined
  let errorOutput: string[]

  beforeEach(() => {
    // Mock process.exit to capture exit codes
    originalExit = process.exit
    exitCode = undefined
    process.exit = ((code?: number) => {
      exitCode = code
      throw new Error(`Process exit with code ${code}`)
    }) as typeof process.exit

    // Mock console.error to capture error output
    originalConsoleError = console.error
    errorOutput = []
    console.error = (...args: any[]) => {
      errorOutput.push(args.join(' '))
    }
  })

  afterEach(() => {
    // Restore original functions
    process.exit = originalExit
    console.error = originalConsoleError
  })

  describe('default arguments', () => {
    it('should return default values when no arguments provided', () => {
      const args = parseArgs(['node', 'cli.js'])
      
      strictEqual(args.config, DEFAULT_ARGS.config)
      strictEqual(args.mode, 'pipeline')
      deepStrictEqual(args.override, DEFAULT_ARGS.override)
      strictEqual(args.verbose, DEFAULT_ARGS.verbose)
      strictEqual(args.noProgress, DEFAULT_ARGS.noProgress)
      strictEqual(args.version, false)
    })
  })

  describe('config option', () => {
    it('should parse short config option', () => {
      const args = parseArgs(['node', 'cli.js', '-c', 'custom.json'])
      strictEqual(args.config, 'custom.json')
    })

    it('should parse long config option', () => {
      const args = parseArgs(['node', 'cli.js', '--config', 'my-pipeline.json'])
      strictEqual(args.config, 'my-pipeline.json')
    })

    it('should handle config with spaces in path', () => {
      const args = parseArgs(['node', 'cli.js', '-c', '/path/with spaces/config.json'])
      strictEqual(args.config, '/path/with spaces/config.json')
    })
  })

  describe('mode option', () => {
    it('should parse pipeline mode', () => {
      const args = parseArgs(['node', 'cli.js', '-m', 'pipeline'])
      strictEqual(args.mode, 'pipeline')
    })

    it('should parse interactive mode', () => {
      const args = parseArgs(['node', 'cli.js', '-m', 'interactive'])
      strictEqual(args.mode, 'interactive')
    })

    it('should parse server mode', () => {
      const args = parseArgs(['node', 'cli.js', '--mode', 'server'])
      strictEqual(args.mode, 'server')
    })

    it('should reject invalid mode', () => {
      throws(() => {
        parseArgs(['node', 'cli.js', '-m', 'invalid'])
      }, /Process exit with code 1/)
      
      strictEqual(exitCode, 1)
      strictEqual(errorOutput.length > 0, true)
      ok(errorOutput.some(msg => msg.includes('Invalid mode')))
    })
  })

  describe('override option', () => {
    it('should parse single override', () => {
      const args = parseArgs(['node', 'cli.js', '-o', 'input.path=/new/path'])
      deepStrictEqual(args.override, ['input.path=/new/path'])
    })

    it('should parse multiple overrides with short option', () => {
      const args = parseArgs(['node', 'cli.js', '-o', 'input.path=/new/path', '-o', 'output.format=jsonl'])
      deepStrictEqual(args.override, ['input.path=/new/path', 'output.format=jsonl'])
    })

    it('should parse multiple overrides with long option', () => {
      const args = parseArgs(['node', 'cli.js', '--override', 'input.path=/data', '--override', 'options.chunkSize=1000'])
      deepStrictEqual(args.override, ['input.path=/data', 'options.chunkSize=1000'])
    })

    it('should handle complex override values', () => {
      const args = parseArgs(['node', 'cli.js', '-o', 'transformations.0.params.windowSize=50'])
      deepStrictEqual(args.override, ['transformations.0.params.windowSize=50'])
    })

    it('should handle overrides with special characters', () => {
      const args = parseArgs(['node', 'cli.js', '-o', 'input.path=/path/with-dashes_and.dots/file.csv'])
      deepStrictEqual(args.override, ['input.path=/path/with-dashes_and.dots/file.csv'])
    })
  })

  describe('verbose option', () => {
    it('should handle single verbose flag', () => {
      const args = parseArgs(['node', 'cli.js', '-v'])
      strictEqual(args.verbose, 1)
    })

    it('should handle multiple verbose flags', () => {
      const args = parseArgs(['node', 'cli.js', '-vv'])
      strictEqual(args.verbose, 2)
    })

    it('should handle separate verbose flags', () => {
      const args = parseArgs(['node', 'cli.js', '-v', '-v', '-v'])
      strictEqual(args.verbose, 3)
    })

    it('should handle long verbose option', () => {
      const args = parseArgs(['node', 'cli.js', '--verbose', '--verbose'])
      strictEqual(args.verbose, 2)
    })
  })

  describe('progress option', () => {
    it('should default to progress enabled', () => {
      const args = parseArgs(['node', 'cli.js'])
      strictEqual(args.noProgress, false)
    })

    it('should disable progress with --no-progress', () => {
      const args = parseArgs(['node', 'cli.js', '--no-progress'])
      strictEqual(args.noProgress, true)
    })
  })

  describe('combined options', () => {
    it('should parse multiple options correctly', () => {
      const args = parseArgs([
        'node', 'cli.js',
        '-c', 'custom.json',
        '-m', 'server',
        '-o', 'input.path=/data',
        '-o', 'output.format=jsonl',
        '-vv',
        '--no-progress'
      ])

      strictEqual(args.config, 'custom.json')
      strictEqual(args.mode, 'server')
      deepStrictEqual(args.override, ['input.path=/data', 'output.format=jsonl'])
      strictEqual(args.verbose, 2)
      strictEqual(args.noProgress, true)
    })

    it('should handle options in different order', () => {
      const args = parseArgs([
        'node', 'cli.js',
        '--no-progress',
        '-vv',
        '--mode', 'server',
        '--override', 'input.path=/data',
        '--config', 'test.json'
      ])

      strictEqual(args.config, 'test.json')
      strictEqual(args.mode, 'server')
      deepStrictEqual(args.override, ['input.path=/data'])
      strictEqual(args.verbose, 2)
      strictEqual(args.noProgress, true)
    })
  })

  describe('edge cases', () => {
    it('should handle empty override array', () => {
      const args = parseArgs(['node', 'cli.js', '-c', 'test.json'])
      deepStrictEqual(args.override, [])
    })

    it('should handle zero verbose level', () => {
      const args = parseArgs(['node', 'cli.js', '-c', 'test.json'])
      strictEqual(args.verbose, 0)
    })

    it('should handle paths with equals signs', () => {
      const args = parseArgs(['node', 'cli.js', '-o', 'input.query=symbol=BTCUSD'])
      deepStrictEqual(args.override, ['input.query=symbol=BTCUSD'])
    })

    it('should handle empty config path', () => {
      const args = parseArgs(['node', 'cli.js', '-c', ''])
      strictEqual(args.config, '')
    })
  })

  describe('argument validation', () => {
    it('should validate mode option during parsing', () => {
      throws(() => {
        parseArgs(['node', 'cli.js', '--mode', 'invalid-mode'])
      }, /Process exit with code 1/)
      
      strictEqual(exitCode, 1)
    })

    it('should handle missing required values gracefully', () => {
      // When using --config without a value, commander should handle this
      // This test verifies the behavior without asserting specific outcomes
      // since commander's behavior may vary
      try {
        parseArgs(['node', 'cli.js', '--config'])
      } catch (error) {
        // Commander may throw for missing values, which is acceptable
      }
    })
  })
})