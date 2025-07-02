import typescriptEslint from "@typescript-eslint/eslint-plugin"
import tsParser from "@typescript-eslint/parser"
import path from "node:path"
import { fileURLToPath } from "node:url"
import js from "@eslint/js"
import { FlatCompat } from "@eslint/eslintrc"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all
})

export default [{
  ignores: [
    "**/*.js",
    "**/node_modules",
    "**/build",
    "**/dist",
    "**/coverage",
    "**/jest.config.js",
    "**/__tests__/**/*",
    "**/__mocks__/**/*",
    "**/*.test.ts",
    "**/*.test.tsx"
  ],
}, {
  files: ["**/*.ts", "**/*.tsx"],
}, ...compat.extends(
  "eslint:recommended",
  "plugin:@typescript-eslint/recommended",
  "plugin:@typescript-eslint/recommended-requiring-type-checking",
  "plugin:@typescript-eslint/recommended-type-checked",
  "plugin:@typescript-eslint/stylistic-type-checked"
), {
  plugins: {
    "@typescript-eslint": typescriptEslint,
  },

  languageOptions: {
    parser: tsParser,
    ecmaVersion: 2022,
    sourceType: "module",

    parserOptions: {
      projectService: true,
      tsconfigRootDir: __dirname,
    },
  },

  rules: {
    // 'no-console': 'warn',
    // '@typescript-eslint/no-duplicate-type-constituents': 'warn',
    // '@typescript-eslint/no-redundant-type-constituents': 'warn',
    // "no-inner-declarations": "off",
    // "no-lone-blocks": "off",
    // "@typescript-eslint/array-type": "off",
    // "@typescript-eslint/ban-ts-comment": "off",
    // '@typescript-eslint/explicit-function-return-type': 'warn',
    // "@typescript-eslint/no-empty-interface": "off",
    // "@typescript-eslint/no-explicit-any": "warn",
    // // '@typescript-eslint/no-floating-promises': 'off',
    // // '@typescript-eslint/no-misused-promises': 'off',
    // "@typescript-eslint/no-non-null-assertion": "off",
    // // '@typescript-eslint/no-unsafe-assignment': 'off',
    // "@typescript-eslint/prefer-nullish-coalescing": "off", // ?? only works for null/undefined. We need || for '' too.
    // "@typescript-eslint/prefer-optional-chain": "warn",

    "@typescript-eslint/no-duplicate-type-constituents": "warn",
    "@typescript-eslint/no-redundant-type-constituents": "warn",
    "no-inner-declarations": "off",
    "no-lone-blocks": "off",
    "@typescript-eslint/array-type": "off",
    "@typescript-eslint/ban-ts-comment": "off",
    "@typescript-eslint/no-empty-interface": "off",
    "@typescript-eslint/no-explicit-any": "warn",
    // '@typescript-eslint/no-floating-promises': 'off',
    // '@typescript-eslint/no-misused-promises': 'off',
    "@typescript-eslint/no-non-null-assertion": "off",
    // '@typescript-eslint/no-unsafe-assignment': 'off',
    "@typescript-eslint/prefer-nullish-coalescing": "off", // ?? only works for null/undefined. We need || for '' too.
    "@typescript-eslint/prefer-optional-chain": "warn",

    "semi": ["error", "never"],
    // "@typescript-eslint/member-delimiter-style": [
    //   "error",
    //   {
    //     "multiline": {
    //       "delimiter": "none"
    //     }
    //   }
    // ],
    "@typescript-eslint/prefer-readonly": "error",
    "@typescript-eslint/explicit-function-return-type": [
      "warn",
      {
        "allowExpressions": true,
        "allowTypedFunctionExpressions": true
      }
    ],
    "@typescript-eslint/no-unused-vars": [
      "error",
      {
        "argsIgnorePattern": "^_",
        "varsIgnorePattern": "^_"
      }
    ],
    "@typescript-eslint/consistent-type-imports": [
      "error",
      {
        "prefer": "type-imports"
      }
    ],
    "no-console": ["warn", { "allow": ["warn", "error"] }]
  },
}]
