# Core Type Aliases for Record<string, unknown> and unknown

**Purpose:**
- Standardize and document all type aliases for generic object and unknown types in core
- Improve type safety, clarity, and maintainability
- Replace direct use of `Record<string, unknown>` and `unknown` with meaningful aliases

**Architecture:**
- Core modules use type aliases to represent flexible, extensible data structures
- Type aliases defined at the top of each relevant file or in shared types
- Aliases flow through event system, network, validation, and strategy modules
- All public interfaces and DTOs use named aliases for generic object/unknown types
- Prefer specific interfaces if structure is known; use aliases for open-ended data

**Dataflow:**
- Data enters system via events, network requests, or strategy configs
- Aliased types (`EventPayload`, `AgentReasoning`, etc.) used for payloads, configs, metadata
- Data flows through event bus, logger, validators, and strategy managers
- Aliases ensure consistent handling and documentation of flexible data
- Serialization/deserialization uses these aliases for type safety

**Conventions:**
- Always use a named alias for generic object or unknown types in public interfaces
- Document each alias in this file and in code JSDoc
- Prefer specific interfaces if the structure is known
- Use these aliases for clarity, maintainability, and future refactoring

**Summary Table**

| Alias Name              | Original Type              | Usage Location(s)                        | Description                                  |
|------------------------|---------------------------|-------------------------------------------|----------------------------------------------|
| `EventPayload`         | `Record<string, unknown>`  | `events/types.ts` (EventData)             | Arbitrary event payload data                 |
| `AgentReasoning`       | `Record<string, unknown>`  | `events/types.ts` (AgentSignalEvent)      | Reasoning for agent signals                  |
| `SystemConfig`         | `Record<string, unknown>`  | `events/types.ts` (SystemStartEvent)      | System configuration object                  |
| `SystemInfoDetails`    | `Record<string, unknown>`  | `events/types.ts` (SystemInfoEvent)       | Additional system info details               |
| `EventMetadata`        | `Record<string, unknown>`  | `events/event-logger.ts` (EventLogEntry)  | Metadata for event log entries               |
| `ValidationMetadata`   | `Record<string, unknown>`  | `market-data/data-validator.ts`           | Metadata for validation issues               |
| `IndicatorValueMap`    | `Record<string, number>`   | `indicators/interfaces.ts`                | Multi-value indicator result map             |
| `PositionSizingParams` | `Record<string, unknown>`  | `position-sizing/strategies/*.ts`         | Params for position sizing strategies        |
| `RequestBody`          | `unknown`                  | `interfaces/network-client.ts`            | HTTP request body                           |
| `ResponseData`         | `unknown`                  | `interfaces/network-client.ts`            | HTTP response data                          |
| `ErrorResponse`        | `unknown`                  | `interfaces/network-client.ts`            | HTTP error response                         |
| `EventSnapshot`        | `Record<string, unknown>`  | `events/market-data-events.ts`            | Serializable event snapshot                  |
| `UnknownEvent`         | `unknown`                  | `events/market-data-events.ts`            | Generic event for validation/size            |

**Alias Definitions and Usage**

---

### `EventPayload`
- **Type:** `Record<string, unknown>`
- **Where:** `events/types.ts` (EventData)
- **Use:** Store arbitrary event payloads. Use for extensible event data.

### `AgentReasoning`
- **Type:** `Record<string, unknown>`
- **Where:** `events/types.ts` (AgentSignalEvent)
- **Use:** Store agent reasoning details. Improves clarity over generic object.

### `SystemConfig`
- **Type:** `Record<string, unknown>`
- **Where:** `events/types.ts` (SystemStartEvent)
- **Use:** Store system configuration. Use for flexible config objects.

### `SystemInfoDetails`
- **Type:** `Record<string, unknown>`
- **Where:** `events/types.ts` (SystemInfoEvent)
- **Use:** Store additional system info. Use for extensible details.

### `EventMetadata`
- **Type:** `Record<string, unknown>`
- **Where:** `events/event-logger.ts` (EventLogEntry)
- **Use:** Store metadata for event logs. Use for arbitrary log context.

### `ValidationMetadata`
- **Type:** `Record<string, unknown>`
- **Where:** `market-data/data-validator.ts` (ValidationIssue)
- **Use:** Store metadata for validation issues. Use for extensible validation context.

### `IndicatorValueMap`
- **Type:** `Record<string, number>`
- **Where:** `indicators/interfaces.ts` (MultiValueIndicatorResult)
- **Use:** Store multiple indicator values. Use for indicators with multiple outputs.

### `PositionSizingParams`
- **Type:** `Record<string, unknown>`
- **Where:** `position-sizing/strategies/*.ts`
- **Use:** Store params for position sizing strategies. Use for flexible strategy configs.

### `RequestBody`
- **Type:** `unknown`
- **Where:** `interfaces/network-client.ts` (RequestOptions)
- **Use:** HTTP request body. Use for generic or unknown request payloads.

### `ResponseData`
- **Type:** `unknown`
- **Where:** `interfaces/network-client.ts` (NetworkResponse)
- **Use:** HTTP response data. Use for generic or unknown response payloads.

### `ErrorResponse`
- **Type:** `unknown`
- **Where:** `interfaces/network-client.ts` (NetworkError)
- **Use:** HTTP error response. Use for generic or unknown error payloads.

### `EventSnapshot`
- **Type:** `Record<string, unknown>`
- **Where:** `events/market-data-events.ts` (EventSerializer)
- **Use:** Serializable event snapshot. Use for event archiving and replay.

### `UnknownEvent`
- **Type:** `unknown`
- **Where:** `events/market-data-events.ts` (EventSerializer)
- **Use:** Generic event for validation and size calculation.

--- 