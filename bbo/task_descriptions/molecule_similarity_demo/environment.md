# Environment Setup

This task uses the repository-local scientific dataset layout, RDKit, and `selfies`.

One verified setup path is:

```bash
uv sync --extra dev --extra bo-tutorial
```

Minimal smoke command:

```bash
uv run python -m bbo.run --algorithm random_search --task molecule_similarity_demo --max-evaluations 3
```

If `uv` is unavailable, use an equivalent Python environment with RDKit and `selfies` installed, and keep the bundled `bbo/tasks/scientific/data/examples/Molecule/zinc.txt.gz` file available.
