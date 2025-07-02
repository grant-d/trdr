import { createDatabase, type Database, type SQLiteConfig } from '@trdr/data'
import { eventBus } from '../events/event-bus'

/**
 * Factory function to create a database instance with the core eventBus
 * This ensures the data layer can emit events without directly depending on core
 */
export async function createDatabaseWithEventBus(config?: Partial<SQLiteConfig>): Promise<Database> {
  return createDatabase(config, eventBus)
}