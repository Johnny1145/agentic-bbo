# Goal

Minimize `median1_loss`, where:

- `median1_loss = 1.0 - median1_score`
- `median1_score` is the geometric mean of ECFP4 Tanimoto similarity to camphor and menthol

Lower loss means the decoded molecule is more balanced between both target structures.
