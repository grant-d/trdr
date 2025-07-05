import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { URL } from 'node:url'
import type { PipelineConfig } from '../interfaces'
import { ConfigValidator } from './config-validator'
import { PipelineFactory } from './pipeline-factory'

/**
 * Server mode configuration
 */
export interface ServerConfig {
  /** Port to listen on */
  port?: number
  /** Host to bind to */
  host?: string
  /** Enable CORS */
  cors?: boolean
  /** Maximum request body size in bytes */
  maxBodySize?: number
}

/**
 * Pipeline execution request
 */
export interface ExecuteRequest {
  /** Optional configuration overrides */
  config?: Partial<PipelineConfig>
  /** Whether to return progress updates via SSE */
  stream?: boolean
}

/**
 * Server mode for running pipelines via HTTP API
 */
export class ServerMode {
  private readonly baseConfig: PipelineConfig
  private readonly serverConfig: Required<ServerConfig>
  private server?: Server
  private readonly activePipelines: Map<string, any>

  constructor(config: PipelineConfig, serverConfig: ServerConfig = {}) {
    this.baseConfig = config
    this.serverConfig = {
      port: serverConfig.port ?? 3000,
      host: serverConfig.host ?? 'localhost',
      cors: serverConfig.cors ?? true,
      maxBodySize: serverConfig.maxBodySize ?? 10 * 1024 * 1024, // 10MB
    }
    this.activePipelines = new Map()
  }

  /**
   * Start the HTTP server
   */
  public async start(): Promise<void> {
    this.server = createServer(this.handleRequest.bind(this))

    return new Promise((resolve, reject) => {
      this.server!.listen(this.serverConfig.port, this.serverConfig.host, () => {
        console.log(`TRDR Server listening on http://${this.serverConfig.host}:${this.serverConfig.port}`)
        console.log('Available endpoints:')
        console.log('  GET  /health        - Health check')
        console.log('  GET  /config        - Get current configuration')
        console.log('  POST /validate      - Validate configuration')
        console.log('  POST /execute       - Execute pipeline')
        console.log('  GET  /status/:id    - Get pipeline status')
        console.log('  POST /stop/:id      - Stop pipeline')
        resolve()
      })

      this.server!.on('error', reject)
    })
  }

  /**
   * Stop the server
   */
  public async stop(): Promise<void> {
    if (!this.server) return

    return new Promise((resolve) => {
      this.server!.close(() => {
        console.log('Server stopped')
        resolve()
      })
    })
  }

  /**
   * Handle HTTP requests
   */
  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    // Set CORS headers if enabled
    if (this.serverConfig.cors) {
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

      if (req.method === 'OPTIONS') {
        res.writeHead(204)
        res.end()
        return
      }
    }

    const url = new URL(req.url || '', `http://${req.headers.host}`)
    const path = url.pathname
    const method = req.method || 'GET'

    try {
      // Route requests
      if (method === 'GET' && path === '/health') {
        await this.handleHealth(req, res)
      } else if (method === 'GET' && path === '/config') {
        await this.handleGetConfig(req, res)
      } else if (method === 'POST' && path === '/validate') {
        await this.handleValidate(req, res)
      } else if (method === 'POST' && path === '/execute') {
        await this.handleExecute(req, res)
      } else if (method === 'GET' && path.startsWith('/status/')) {
        const id = path.substring('/status/'.length)
        await this.handleStatus(req, res, id)
      } else if (method === 'POST' && path.startsWith('/stop/')) {
        const id = path.substring('/stop/'.length)
        await this.handleStop(req, res, id)
      } else {
        this.sendError(res, 404, 'Not found')
      }
    } catch (error) {
      console.error('Request error:', error)
      this.sendError(res, 500, 'Internal server error')
    }
  }

  /**
   * Handle health check
   */
  private async handleHealth(_req: IncomingMessage, res: ServerResponse): Promise<void> {
    this.sendJson(res, 200, {
      status: 'ok',
      uptime: process.uptime(),
      activePipelines: this.activePipelines.size,
    })
  }

  /**
   * Handle get configuration
   */
  private async handleGetConfig(_req: IncomingMessage, res: ServerResponse): Promise<void> {
    this.sendJson(res, 200, this.baseConfig)
  }

  /**
   * Handle validate configuration
   */
  private async handleValidate(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.parseBody(req)
    const config = body.config || this.baseConfig

    const validation = ConfigValidator.validateWithDetails(config)

    this.sendJson(res, 200, {
      valid: validation.isValid,
      errors: validation.errors,
      errorMessages: validation.errorMessages,
      warningMessages: validation.warningMessages,
    })
  }

  /**
   * Handle execute pipeline
   */
  private async handleExecute(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.parseBody(req) as ExecuteRequest
    const pipelineId = this.generateId()

    // Merge configurations
    const config = body.config
      ? { ...this.baseConfig, ...body.config }
      : this.baseConfig

    // Validate configuration
    const validation = ConfigValidator.validateWithDetails(config)
    if (!validation.isValid) {
      this.sendJson(res, 400, {
        error: 'Invalid configuration',
        details: validation.errors,
      })
      return
    }

    // Execute pipeline
    try {
      const pipeline = await PipelineFactory.createPipeline(config as any)

      // Store pipeline reference
      this.activePipelines.set(pipelineId, {
        pipeline,
        startTime: Date.now(),
        status: 'running',
      })

      // If streaming is requested, set up SSE
      if (body.stream) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        })

        pipeline.onProgress((update) => {
          res.write(`data: ${JSON.stringify({
            type: 'progress',
            ...update,
          })}\n\n`)
        })
      }

      // Execute pipeline asynchronously
      pipeline.execute().then(result => {
        const pipelineInfo = this.activePipelines.get(pipelineId)
        if (pipelineInfo) {
          pipelineInfo.status = 'completed'
          pipelineInfo.result = result
        }

        if (body.stream) {
          res.write(`data: ${JSON.stringify({
            type: 'complete',
            result,
          })}\n\n`)
          res.end()
        }
      }).catch(error => {
        const pipelineInfo = this.activePipelines.get(pipelineId)
        if (pipelineInfo) {
          pipelineInfo.status = 'failed'
          pipelineInfo.error = error.message
        }

        if (body.stream) {
          res.write(`data: ${JSON.stringify({
            type: 'error',
            error: error.message,
          })}\n\n`)
          res.end()
        }
      })

      // Return pipeline ID immediately
      if (!body.stream) {
        this.sendJson(res, 202, {
          id: pipelineId,
          status: 'running',
          message: 'Pipeline execution started',
        })
      }
    } catch (error) {
      this.sendError(res, 500, 'Failed to start pipeline')
    }
  }

  /**
   * Handle get pipeline status
   */
  private async handleStatus(_req: IncomingMessage, res: ServerResponse, id: string): Promise<void> {
    const pipelineInfo = this.activePipelines.get(id)

    if (!pipelineInfo) {
      this.sendError(res, 404, 'Pipeline not found')
      return
    }

    const response: any = {
      id,
      status: pipelineInfo.status,
      startTime: pipelineInfo.startTime,
      duration: Date.now() - pipelineInfo.startTime,
    }

    if (pipelineInfo.result) {
      response.result = pipelineInfo.result
    }

    if (pipelineInfo.error) {
      response.error = pipelineInfo.error
    }

    this.sendJson(res, 200, response)
  }

  /**
   * Handle stop pipeline
   */
  private async handleStop(_req: IncomingMessage, res: ServerResponse, id: string): Promise<void> {
    const pipelineInfo = this.activePipelines.get(id)

    if (!pipelineInfo) {
      this.sendError(res, 404, 'Pipeline not found')
      return
    }

    // TODO: Implement pipeline cancellation
    this.sendJson(res, 501, {
      error: 'Pipeline cancellation not yet implemented',
    })
  }

  /**
   * Parse request body
   */
  private async parseBody(req: IncomingMessage): Promise<any> {
    return new Promise((resolve, reject) => {
      let body = ''
      let size = 0

      req.on('data', (chunk) => {
        size += chunk.length
        if (size > this.serverConfig.maxBodySize) {
          reject(new Error('Request body too large'))
          return
        }
        body += chunk.toString()
      })

      req.on('end', () => {
        try {
          resolve(body ? JSON.parse(body) : {})
        } catch (error) {
          reject(new Error('Invalid JSON'))
        }
      })

      req.on('error', reject)
    })
  }

  /**
   * Send JSON response
   */
  private sendJson(res: ServerResponse, status: number, data: any): void {
    res.writeHead(status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(data, null, 2))
  }

  /**
   * Send error response
   */
  private sendError(res: ServerResponse, status: number, message: string): void {
    this.sendJson(res, status, { error: message })
  }

  /**
   * Generate unique pipeline ID
   */
  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substring(2)
  }
}
