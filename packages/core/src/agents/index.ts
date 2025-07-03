export { AgentLifecycleManager, AgentPhase } from './agent-lifecycle'
export type { AgentLifecycleConfig, AgentReplacement, AgentState, HealthCheckResult } from './agent-lifecycle'
export { AgentOrchestrator } from './agent-orchestrator'
export { BaseAgent } from './base-agent'
export {
  bayesianConsensus, calculateConfidenceInterval, confidenceWeightedConsensus, exponentialWeightedConsensus, normalizeSignals,
  performanceWeightedConsensus, vetoConsensus
} from './consensus-algorithms'
export { PerformanceTracker } from './performance-tracker'
export type { MarketSnapshot, PerformanceHistory, PerformanceMetric, PerformanceSummary, PerformanceTrackerConfig, RecentPerformance, TradeOutcome, WeightAdjustment } from './performance-tracker'
export { PerformanceVisualizer } from './performance-visualizer'
export type { AgentDashboard, ComparativePerformance, PerformanceDataPoint, PerformanceSeries } from './performance-visualizer'
export * from './types'
