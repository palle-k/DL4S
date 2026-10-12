//
//  GPUCache.swift
//  DL4S
//
//  Created by Palle Klewitz on 08.10.26.
//  Copyright (c) 2026 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Synchronization

// Shapes that change in every step, such as the lengths of padded sequences, would let a cache without a limit grow without bound.
/// Values that take long to create, by key, with a limit on their number. A full cache removes the value that was used least
/// recently.
struct GPUCache<Key: Hashable, Value> {
    private struct Entry {
        var value: Value
        var lastUse: UInt64
    }

    /// Maximum number of values.
    let capacity: Int
    private var entries: [Key: Entry] = [:]
    private var uses: UInt64 = 0

    init(capacity: Int) {
        self.capacity = capacity
    }

    /// The value for the key, or nil when the cache has none.
    mutating func value(for key: Key) -> Value? {
        guard let index = entries.index(forKey: key) else {
            return nil
        }
        uses += 1
        entries.values[index].lastUse = uses
        return entries.values[index].value
    }

    /// Adds a value. When the cache is full, the value that was used least recently is removed and returned.
    mutating func insert(_ value: Value, for key: Key) -> Value? {
        var removed: Value?
        if entries.count >= capacity, entries[key] == nil, let oldest = entries.min(by: { $0.value.lastUse < $1.value.lastUse })?.key {
            removed = entries.removeValue(forKey: oldest)?.value
        }
        uses += 1
        entries[key] = Entry(value: value, lastUse: uses)
        return removed
    }

    /// Removes all values and returns them.
    mutating func removeAll() -> [Value] {
        let values = entries.values.map(\.value)
        entries.removeAll()
        return values
    }
}

extension GPUCache where Key: Sendable, Value: Sendable {
    // The creation runs without the lock, so that other threads can use the cache meanwhile.
    /// Returns the value for the key, and creates and adds it when the cache has none.
    ///
    /// When two threads create the value for the same key, both get their own value, and the cache keeps the second one.
    static func value(in cache: borrowing Mutex<Self>, for key: Key, create: () -> Value) -> Value {
        if let value = cache.withLock({ $0.value(for: key) }) {
            return value
        }
        let value = create()
        // A removed value is released after the lock, so that no deinitializer runs while the lock is held.
        let removed = cache.withLock { $0.insert(value, for: key) }
        _ = consume removed
        return value
    }
}
#endif
