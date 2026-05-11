# Environment Setup

One verified setup path is:

```bash
uv sync --extra dev --extra bo-tutorial
```

Minimal smoke command:

```bash
uv run --extra bo-tutorial python -m bbo.run --algorithm random_search --task guacamol_qed_selfies_demo --max-evaluations 3 --no-plots
```

The task requires RDKit, `selfies`, and the bundled `bbo/tasks/scientific/data/examples/Molecule/zinc.txt.gz` file.
