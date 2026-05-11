# Goal

Minimize `aripiprazole_similarity_loss`, where:

- `aripiprazole_similarity_loss = 1.0 - aripiprazole_similarity_score`
- `aripiprazole_similarity_score` is FCFP4 Tanimoto similarity to Aripiprazole clipped at the GuacaMol threshold `0.75`

Any decoded molecule at or above the threshold receives the maximum score.
