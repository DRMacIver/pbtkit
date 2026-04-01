<div class="ai">

*The following summary was written by an LLM.*

The database module persists failing test cases to disk so they are replayed on the next run. When a test fails, its choice sequence is serialized and stored keyed by the test name. On the next run, the stored failure is loaded and replayed before any random generation begins — if it still fails, shrinking starts from that point rather than rediscovering the failure from scratch.

This is critical for CI workflows: a failure found locally is immediately reproducible on the next run, and the shrunk result is cached so re-running doesn't re-shrink.

The serialization format is a compact binary encoding: each choice value is tagged with its type (integer, boolean, bytes, float, string) and serialized with type-specific logic. This allows the database to round-trip choice sequences even when they contain mixed types.

</div>
