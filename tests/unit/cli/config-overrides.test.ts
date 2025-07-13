import { strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { createDefaultPipelineConfig } from '../../../src/cli/config-loader'
import { applyOverrides } from '../../../src/cli/config-overrides'
import type { FileInputConfig } from '../../../src/interfaces'

describe('Config Overrides', () => {
  describe('applyOverrides', () => {
    it('should apply simple string override', () => {
      const config = createDefaultPipelineConfig()
      applyOverrides(config, ['input.path=/new/data.csv'])
      const fileInput = config.input as FileInputConfig
      strictEqual(fileInput.path, '/new/data.csv')
      strictEqual(fileInput.format, 'csv')
    })

    it('should apply numeric overrides', () => {
      const config = createDefaultPipelineConfig()
      applyOverrides(config, [
        'input.chunkSize=2000',
        'options.chunkSize=5000'
      ])
      const fileInput = config.input as FileInputConfig
      strictEqual(fileInput.chunkSize, 2000)
      strictEqual(config.options?.chunkSize, 5000)
    })
  })
})