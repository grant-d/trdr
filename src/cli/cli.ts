#!/usr/bin/env node

import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import { ArgsParser } from './args-parser'
import { ConfigLoader } from './config-loader'
import { ConfigOverrides } from './config-overrides'
import { ConfigValidator } from './config-validator'
import { InteractiveMode } from './interactive-mode'
import { PipelineFactory } from './pipeline-factory'
import { ProgressIndicator } from './progress-indicator'
import { ServerMode } from './server-mode'

/**
 * Get package version
 */
async function getPackageVersion(): Promise<string> {
  try {
    const packagePath = path.join(process.cwd(), 'package.json')
    const packageJson = JSON.parse(await fs.readFile(packagePath, 'utf-8'))
    return packageJson.version || '1.0.0'
  } catch {
    return '1.0.0'
  }
}

/**
 * Main CLI entry point
 */
async function main(): Promise<void> {
  try {
    // Parse command line arguments
    const args = ArgsParser.parse(process.argv)

    // Handle version flag
    if (args.version) {
      const version = await getPackageVersion()
      console.log(`trdr v${version}`)
      process.exit(0)
    }

    // Set up verbosity
    if (args.verbose > 0) {
      process.env.LOG_LEVEL = args.verbose >= 2 ? 'debug' : 'info'
    }

    // Check if config file exists
    try {
      await fs.access(args.config)
    } catch {
      console.error(`Configuration file not found: ${args.config}`)
      console.error('\nCreate a pipeline.json file or specify a different config with -c option')
      console.error('\nExample pipeline.json:')
      console.error(JSON.stringify({
        input: { type: 'csv', path: './data/input.csv' },
        output: { path: './data/output.jsonl', format: 'jsonl' },
        transformations: [],
        options: {},
      }, null, 2))
      process.exit(1)
    }

    // Load configuration
    console.log(`Loading configuration from: ${args.config}`)
    const rawConfig = await ConfigLoader.load(args.config)

    // Apply command line overrides
    const overriddenConfig = ConfigOverrides.apply(rawConfig, args.override)

    // Validate configuration
    const validation = ConfigValidator.validateWithDetails(overriddenConfig)

    if (!validation.isValid) {
      console.error('Configuration validation failed:')
      validation.errors.forEach((error: any) => {
        console.error(`  ${error.field}: ${error.message}`)
      })
      process.exit(1)
    }

    // Handle different modes
    switch (args.mode) {
      case 'pipeline':
        await runPipeline(overriddenConfig, args)
        break
        
      case 'interactive':
        await runInteractiveMode(overriddenConfig, args)
        break

      case 'server':
        await runServerMode(overriddenConfig, args)
        break

      default:
        console.error(`Unknown mode: ${args.mode}`)
        process.exit(1)
    }

  } catch (error) {
    console.error('Fatal error:', error instanceof Error ? error.message : error)
    process.exit(1)
  }
}

/**
 * Run the pipeline once
 */
async function runPipeline(config: any, args: any): Promise<void> {
  console.log('Creating pipeline...')
  const pipeline = await PipelineFactory.createPipeline(config)

  // Set up progress indicator if enabled
  if (!args.noProgress) {
    const progress = new ProgressIndicator()
    pipeline.onProgress((update) => {
      progress.update(update.current, update.total, update.message)
    })
  }

  console.log('Executing pipeline...')
  const startTime = Date.now()

  try {
    const result = await pipeline.execute()
    const duration = (Date.now() - startTime) / 1000

    console.log('\nPipeline completed successfully!')
    console.log(`  Records processed: ${result.recordsProcessed}`)
    console.log(`  Records written: ${result.recordsWritten}`)
    console.log(`  Errors: ${result.errors}`)
    console.log(`  Duration: ${duration.toFixed(2)}s`)

    if (result.coefficients && result.coefficients.length > 0) {
      console.log(`  Coefficients saved: ${result.coefficients.length}`)
    }
  } catch (error) {
    console.error('\nPipeline execution failed:', error instanceof Error ? error.message : error)
    process.exit(1)
  }
}

/**
 * Run in interactive mode
 */
async function runInteractiveMode(config: any, _args: any): Promise<void> {
  const interactive = new InteractiveMode(config)
  await interactive.start()
}

/**
 * Run in server mode
 */
async function runServerMode(config: any, _args: any): Promise<void> {
  const server = new ServerMode(config)
  await server.start()
}

// Export the main function for testing
export { main }

// Run the CLI if this is the main module
// Note: For ES modules, we always run main() when the file is executed
main().catch(error => {
  console.error('Unhandled error:', error)
  process.exit(1)
})
