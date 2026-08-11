# Constraints

The task dimension and feature order are determined by the validated `JOB_all.joblib` checkpoint.

The source paper describes a 197-knob large MySQL space. If the released checkpoint contains a different number of active features, the difference must be documented in run metadata rather than silently described as 196 or 197.

Each coordinate lies in `[0, 1]` and is decoded using the physical metadata for the corresponding knob.

The configuration space can contain mixed variable types. In particular:

- integer values are rounded after decoding;
- categorical values are selected by discrete bins;
- categorical indices do not imply semantic ordering;
- special values such as `0`, `-1`, or an automatic mode may not behave like ordinary numeric settings.

The optimizer may modify only active configuration coordinates.

One decoded configuration evaluated by the surrogate counts as one evaluation.
