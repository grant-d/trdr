import { AgentSignal as BaseAgentSignal } from '@trdr/shared'

/**
 * Extended agent signal interface for database storage
 */
export interface AgentSignal extends BaseAgentSignal {
  /** Trading pair symbol */
  readonly symbol: string
  /** Market context at time of decision */
  readonly marketContext?: Record<string, any>
  /** Timestamp as Date object */
  readonly timestamp: Date
}