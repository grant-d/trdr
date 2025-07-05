import { Command } from 'commander'

/**
 * Parsed command-line arguments
 */
export interface CliArgs {
  /** Path to the configuration file */
  config: string

  /** Execution mode: pipeline (run once), interactive (REPL), or server (HTTP API) */
  mode: 'pipeline' | 'interactive' | 'server'

  /** Configuration overrides in dot notation (e.g., input.path=/new/path) */
  override: string[]

  /** Verbosity level for logging */
  verbose: number

  /** Disable progress indicators */
  noProgress: boolean

  /** Show version information */
  version: boolean
}

/**
 * Default CLI arguments
 */
export const DEFAULT_ARGS: Partial<CliArgs> = {
  config: 'pipeline.json',
  mode: 'pipeline',
  override: [],
  verbose: 0,
  noProgress: false
}

/**
 * Parse command-line arguments using commander
 */
export function parseArgs(argv: string[]): CliArgs {
  const program = new Command()

  program
    .name('trdr')
    .description('Trading data processing pipeline')
    .version('1.0.0')
    .usage('[options]')

  program
    .option(
      '-c, --config <file>',
      'path to pipeline configuration file',
      DEFAULT_ARGS.config
    )
    .option(
      '-m, --mode <mode>',
      'execution mode: pipeline (run once), interactive (REPL), or server (HTTP API)',
      DEFAULT_ARGS.mode
    )
    .option(
      '-o, --override <override...>',
      'configuration overrides in dot notation (e.g., input.path=/new/path)',
      DEFAULT_ARGS.override
    )
    .option(
      '-v, --verbose',
      'increase verbosity (can be used multiple times: -v, -vv, -vvv)',
      (_, previous) => previous + 1,
      DEFAULT_ARGS.verbose
    )
    .option(
      '--no-progress',
      'disable progress indicators',
    )

  // Add examples to help
  program.addHelpText('after', `

Examples:
  $ trdr                                    # Run pipeline once with default pipeline.json
  $ trdr -c my-pipeline.json                # Use custom config file
  $ trdr -m interactive                     # Run in interactive REPL mode
  $ trdr -m server                          # Run HTTP API server
  $ trdr -o input.path=/data/new.csv        # Override input path
  $ trdr -o input.path=/data -o output.format=jsonl  # Multiple overrides
  $ trdr -vv                                # Run with verbose logging
  $ trdr --no-progress                      # Disable progress indicators
  
Configuration overrides use dot notation to target nested properties:
  input.path                - Override input file path
  input.format              - Override input format (csv, jsonl)
  output.path               - Override output path
  transformations.0.params  - Override first transformation parameters
  options.chunkSize         - Override processing chunk size
`)

  // Remove preAction hook - we'll validate after parsing

  // Parse arguments
  try {
    program.parse(argv)
  } catch (error) {
    // Re-throw with process exit simulation for testing
    const exitError = new Error(`Process exit with code 1`)
    throw exitError
  }
  const options = program.opts()

  // Validate mode option
  if (options.mode && !['pipeline', 'interactive', 'server'].includes(options.mode)) {
    console.error(`Invalid mode '${options.mode}'. Must be 'pipeline', 'interactive', or 'server'.`)
    process.exit(1)
  }

  return {
    config: options.config !== undefined ? options.config : DEFAULT_ARGS.config!,
    mode: (options.mode as 'pipeline' | 'interactive' | 'server') || DEFAULT_ARGS.mode!,
    override: options.override || DEFAULT_ARGS.override!,
    verbose: options.verbose || DEFAULT_ARGS.verbose!,
    noProgress: options.progress === false, // Commander converts --no-progress to progress: false
    version: options.version || false,
  }
}

/**
 * Display help information
 */
export function displayHelp(): void {
  const program = new Command()
  program
    .name('trdr')
    .description('Trading data processing pipeline')
    .version('1.0.0')

  program.help()
}

/**
 * Display version information
 */
export function displayVersion(): void {
  console.log('trdr version 1.0.0')
  console.log('Trading data processing pipeline')
}

/**
 * ArgsParser class for compatibility
 */
export class ArgsParser {
  /**
   * Parse command-line arguments
   */
  public static parse(argv: string[]): CliArgs {
    return parseArgs(argv)
  }

  /**
   * Display help
   */
  public static help(): void {
    displayHelp()
  }

  /**
   * Display version
   */
  public static version(): void {
    displayVersion()
  }
}
