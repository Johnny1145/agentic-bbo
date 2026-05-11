# Goal

Minimize `guacamol_qed_loss`, where:

- `guacamol_qed_loss = 1.0 - guacamol_qed_score`
- `guacamol_qed_score` is RDKit QED for the decoded molecule

Lower loss is equivalent to generating molecules with higher QED.
