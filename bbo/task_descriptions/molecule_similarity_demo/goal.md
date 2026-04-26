# Goal

Minimize the primary objective `similarity_loss`, where:

- `similarity_loss = 1.0 - tanimoto_similarity`
- `tanimoto_similarity` is computed between the decoded SELFIES candidate molecule and the Celecoxib target molecule
- the fingerprint is ECFP4, implemented as a Morgan fingerprint with radius 2

Lower loss is therefore equivalent to generating molecules that are more structurally similar to Celecoxib.
