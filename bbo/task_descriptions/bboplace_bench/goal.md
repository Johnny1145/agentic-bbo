# Goal

The goal is to minimize the scalar objective y returned by the evaluator.

- Decision variable x: a continuous vector of length `2 * n_macro`.
  - The first `n_macro` entries are proposed x-coordinates of macros.
  - The next `n_macro` entries are proposed y-coordinates of macros.
- Objective y: the HPWL value of the legal macro placement obtained after MGO decoding.
- Direction: lower y is better.
- One evaluation: submit one candidate vector x and receive one scalar y.
