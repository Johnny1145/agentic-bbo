# Constraints

- Suggestions must provide every `selfies_token_XX` categorical parameter declared by the task.
- Token values must come from the task search space vocabulary.
- `__EOS__` ends the SELFIES sequence; `__PAD__` is ignored.
- Empty, invalid, or undecodable sequences receive score `0.0` and loss `1.0`.
- The evaluator is deterministic and defaults to 40 evaluations unless overridden.
