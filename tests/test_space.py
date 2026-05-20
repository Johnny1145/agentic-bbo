from __future__ import annotations

import random

import numpy as np

from bbo.core import (
    CategoricalParam,
    FloatParam,
    IntParam,
    SearchSpace,
    StringParam,
    build_continuous_converter,
)


def test_search_space_sampling_and_numeric_roundtrip() -> None:
    space = SearchSpace(
        [
            FloatParam("lr", low=1e-3, high=1e-1, log=True, default=1e-2),
            IntParam("depth", low=2, high=8, default=4),
        ]
    )
    rng = random.Random(0)
    sample = space.sample(rng)
    vector = space.to_numeric_vector(sample)
    recovered = space.from_numeric_vector(vector)

    assert sample.keys() == recovered.keys()
    assert np.allclose(vector, space.to_numeric_vector(recovered))


def test_search_space_rejects_unknown_parameters() -> None:
    space = SearchSpace([CategoricalParam("mode", choices=("a", "b"), default="a")])
    try:
        space.validate_config({"mode": "a", "extra": 1})
    except ValueError as exc:
        assert "Unexpected parameters" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a validation error for an unknown parameter.")


def test_string_param_validates_and_samples_default() -> None:
    param = StringParam(
        "smiles",
        default="CCO",
        min_length=1,
        max_length=16,
        pattern=r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/]+",
    )

    assert param.coerce("C") == "C"
    assert param.effective_default() == "CCO"

    try:
        param.coerce("")
    except ValueError as exc:
        assert "length >=" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a validation error for a too-short string.")

    try:
        param.coerce("C C")
    except ValueError as exc:
        assert "pattern" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a validation error for a pattern mismatch.")


def test_string_param_requires_explicit_default() -> None:
    param = StringParam("smiles", min_length=0, max_length=16)

    try:
        param.effective_default()
    except ValueError as exc:
        assert "does not define a default" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected StringParam.effective_default to require an explicit default.")


def test_string_param_does_not_generic_sample() -> None:
    param = StringParam("smiles", default="CCO", min_length=0, max_length=16)

    try:
        param.sample(random.Random(0))
    except TypeError as exc:
        assert "generic random sampler" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected StringParam.sample to reject generic random sampling.")


def test_string_param_is_not_continuous_convertible() -> None:
    space = SearchSpace([StringParam("smiles", default="C", min_length=1, max_length=32)])

    try:
        space.numeric_bounds()
    except TypeError as exc:
        assert "non-numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected StringParam to be non-numeric.")

    try:
        build_continuous_converter(space)
    except TypeError as exc:
        assert "cannot be converted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected StringParam to reject continuous conversion.")
