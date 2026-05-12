# Example reference output (optional)

When this skill produces complex structured output (e.g., a directory tree of
checkpoints + metrics + result JSONs), placing a small reference output here
helps users:

- Verify their run produced the right shape.
- Debug differences between expected and actual output.
- Understand the output schema without running the skill end-to-end.

When to add an `example/`:

- ✅ Skills that produce non-trivial directory structures (e.g., training output with multiple sub-dirs).
- ✅ Skills with results-format invariants users must preserve (e.g., specific JSON schema).
- ❌ Skills with single-file outputs whose schema is documented inline.
- ❌ Skills with very large outputs (don't bloat the repo).

Keep examples small (KB-scale). Strip any sensitive content. Use realistic but
synthetic input.
