import { stdin, stdout } from 'node:process'
import { createInterface } from 'node:readline'
import type { PipelineConfig } from '../interfaces'
import { ConfigValidator } from './config-validator'
import { PipelineFactory } from './pipeline-factory'
import { ProgressIndicator } from './progress-indicator'

/**
 * Interactive CLI mode for running pipelines
 */
export class InteractiveMode {
  private readonly config: PipelineConfig
  private readonly rl: ReturnType<typeof createInterface>
  private isRunning: boolean

  constructor(config: PipelineConfig) {
    this.config = config
    this.isRunning = false
    this.rl = createInterface({
      input: stdin,
      output: stdout,
      prompt: 'trdr> ',
    })
  }

  /**
   * Start interactive mode
   */
  public async start(): Promise<void> {
    console.log('Welcome to TRDR Interactive Mode')
    console.log('Type "help" for available commands')
    console.log('')

    this.setupHandlers()
    this.rl.prompt()
  }

  /**
   * Set up readline handlers
   */
  private setupHandlers(): void {
    this.rl.on('line', async (line) => {
      const command = line.trim().toLowerCase()

      try {
        await this.handleCommand(command)
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : error)
      }

      if (!this.isRunning) {
        this.rl.prompt()
      }
    })

    this.rl.on('close', () => {
      console.log('\nGoodbye!')
      // Don't call process.exit in tests - let the process exit naturally
      if (process.env.NODE_ENV !== 'test') {
        process.exit(0)
      }
    })
  }

  /**
   * Handle interactive commands
   */
  private async handleCommand(command: string): Promise<void> {
    const [cmd] = command.split(/\s+/)

    switch (cmd) {
      case 'help':
      case 'h':
        this.showHelp()
        break

      case 'run':
      case 'r':
        await this.runPipeline()
        break

      case 'config':
      case 'c':
        this.showConfig()
        break

      case 'validate':
      case 'v':
        this.validateConfig()
        break

      case 'status':
      case 's':
        this.showStatus()
        break

      case 'transforms':
      case 't':
        this.showTransforms()
        break

      case 'exit':
      case 'quit':
      case 'q':
        this.rl.close()
        break

      case '':
        // Empty command, just show prompt again
        break

      default:
        console.log(`Unknown command: ${cmd}`)
        console.log('Type "help" for available commands')
    }
  }

  /**
   * Show help message
   */
  private showHelp(): void {
    console.log(`
Available commands:
  help, h         Show this help message
  run, r          Run the pipeline
  config, c       Show current configuration
  validate, v     Validate current configuration
  status, s       Show pipeline status
  transforms, t   Show available transforms
  exit, quit, q   Exit interactive mode
    `.trim())
  }

  /**
   * Run the pipeline
   */
  private async runPipeline(): Promise<void> {
    if (this.isRunning) {
      console.log('Pipeline is already running')
      return
    }

    this.isRunning = true
    console.log('Starting pipeline...')

    try {
      const pipeline = await PipelineFactory.createPipeline(this.config as any)

      // Set up progress indicator
      const progress = new ProgressIndicator({
        width: 40,
        showPercentage: true,
        showCounts: true,
        showTime: true,
        showEta: true,
      })

      pipeline.onProgress((update) => {
        progress.update(update.current, update.total, update.message)
      })

      const startTime = Date.now()
      const result = await pipeline.execute()
      const duration = (Date.now() - startTime) / 1000

      progress.complete('Pipeline completed')

      console.log(`\nResults:`)
      console.log(`  Records processed: ${result.recordsProcessed}`)
      console.log(`  Records written: ${result.recordsWritten}`)
      console.log(`  Errors: ${result.errors}`)
      console.log(`  Duration: ${duration.toFixed(2)}s`)

    } catch (error) {
      console.error('\nPipeline failed:', error instanceof Error ? error.message : error)
    } finally {
      this.isRunning = false
    }
  }

  /**
   * Show current configuration
   */
  private showConfig(): void {
    console.log('\nCurrent Configuration:')
    console.log(JSON.stringify(this.config, null, 2))
  }

  /**
   * Validate configuration
   */
  private validateConfig(): void {
    const validation = ConfigValidator.validateWithDetails(this.config)

    if (validation.isValid) {
      console.log('✓ Configuration is valid')
    } else {
      console.log('✗ Configuration has errors:')
      validation.errors.forEach(error => {
        const prefix = error.severity === 'error' ? '  ERROR' : '  WARN'
        console.log(`${prefix}: ${error.field} - ${error.message}`)
      })
    }
  }

  /**
   * Show pipeline status
   */
  private showStatus(): void {
    console.log('\nPipeline Status:')
    console.log(`  Running: ${this.isRunning ? 'Yes' : 'No'}`)

    // Show input details based on type
    if (this.config.input.type === 'file') {
      console.log(`  Input: file (${this.config.input.path})`)
    } else if (this.config.input.type === 'provider') {
      console.log(`  Input: ${this.config.input.provider} provider (${this.config.input.symbols.join(', ')})`)
    }

    console.log(`  Output: ${this.config.output.format} (${this.config.output.path})`)
    console.log(`  Transforms: ${this.config.transformations.length}`)
  }

  /**
   * Show available transforms
   */
  private showTransforms(): void {
    console.log('\nConfigured Transforms:')

    if (this.config.transformations.length === 0) {
      console.log('  No transforms configured')
    } else {
      this.config.transformations.forEach((transform, index) => {
        console.log(`  ${index + 1}. ${transform.type}`)
        if (transform.params) {
          Object.entries(transform.params).forEach(([key, value]) => {
            console.log(`     ${key}: ${JSON.stringify(value)}`)
          })
        }
      })
    }

    console.log('\nAvailable Transform Types:')
    const available = PipelineFactory.getAvailableTransforms()
    available.forEach(type => {
      console.log(`  - ${type}`)
    })
  }

  /**
   * Clean up resources
   */
  public cleanup(): void {
    if (this.rl) {
      this.rl.close()
    }
  }
}
