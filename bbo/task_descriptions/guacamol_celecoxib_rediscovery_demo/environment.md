# Environment Setup

One verified setup path is:

```bash
uv sync --extra dev --extra bo-tutorial
```

Minimal smoke command:

```bash
uv run --extra bo-tutorial python -m bbo.run --algorithm random_search --task guacamol_celecoxib_rediscovery_demo --max-evaluations 3 --no-plots
```

The task requires RDKit, `selfies`, and the bundled molecule data file.
