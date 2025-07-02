/**
 * Makes type optional (T or undefined).
 */
export type Optional<T> = T | undefined

/**
 * Makes all properties optional recursively.
 */
export type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>
} : T

/**
 * Removes readonly modifiers recursively.
 */
export type Mutable<T> = { -readonly [P in keyof T]: Mutable<T[P]> }
/**
 * Removes readonly modifiers at root level only.
 */
export type MutableRoot<T> = { -readonly [P in keyof T]: T[P] }

// Properties common to both A and B
// eg type Foo = Common<{ a: number, b: boolean}, { a?: number | null }> => { a: number }
// export type Common<A, B> = { [K in keyof A & keyof B]: A[K] }

// eg type Foo = Rename<Org, 'orgId', 'id'>
// export type Rename<K extends keyof T, N extends string, T> = Pick<T, Exclude<keyof T, K>> & {
//   [P in N]: T[K]
// }

/**
 * Unions type T with every property in object O.
 */
// export type Spray<O extends object, T> = {
//   [prop in keyof O]: O[prop] | T
// }

/**
 * Makes all object properties nullable.
 */
export type Nullable<O extends object> = {
  [prop in keyof O]: O[prop] | null
}

/**
 * Extracts keys from T where values are assignable to U.
 */
export type KeyOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never
}[keyof T]

/**
 * Picks properties from T where values are assignable to U.
 */
// export type PickKeyOfType<T, U> = Pick<T, KeyOfType<T, U>>

/**
 * Makes specific properties required.
 */
export type PickRequired<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>

/**
 * Makes specific properties optional.
 */
export type PickPartial<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

/**
 * Extracts value type from Record.
 */
// export type RecordValueType<T> = T extends Record<string, infer V> ? V : never

/**
 * Extracts array element type.
 */
export type ArrayMemberType<T> = T extends Array<infer V> ? V : never

/**
 * Converts object keys to lowercase.
 */
export type LowercaseKeys<T> = {
  [K in keyof T as Lowercase<string & K>]: T[K]
}

/**
 * Converts object keys to uppercase.
 */
export type UppercaseKeys<T> = {
  [K in keyof T as Uppercase<string & K>]: T[K]
}

/**
 * Predicate function type returning R (default boolean).
 */
export type Predicate<T, R = boolean> = (i: T) => R
