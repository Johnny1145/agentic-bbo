# Submission Interface

A suggestion must contain:

```json
{
  "selfies_token_00": "[C]",
  "selfies_token_01": "[C]",
  "selfies_token_02": "__EOS__"
}
```

The task declares one categorical parameter per SELFIES token slot.
Optimizers must submit all `selfies_token_XX` parameters declared in the search space.
`__EOS__` ends the molecule sequence, and `__PAD__` is ignored.
