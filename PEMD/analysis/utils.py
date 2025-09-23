"""Utility helpers shared across PEMD analysis modules."""

from __future__ import annotations

import numpy as np


def minimum_image_displacement(
    x0: np.ndarray,
    x1: np.ndarray,
    box_length: np.ndarray | float,
) -> np.ndarray:
    """Return the displacement vector under periodic boundary conditions.

    Parameters
    ----------
    x0, x1
        Arrays containing the reference coordinates. Broadcasting between the
        two operands is supported, matching the behaviour of ``numpy``
        arithmetic.
    box_length
        Simulation box lengths. Either a scalar (cubic box) or an array-like
        object with three components describing the orthogonal box lengths.

    Returns
    -------
    numpy.ndarray
        The displacement vectors taking the minimum image convention into
        account.
    """

    delta = np.asarray(x1) - np.asarray(x0)
    box = np.asarray(box_length)

    # ``ts.dimensions`` may provide 6 values (length + angles).  Only the
    # translational components are relevant for the minimum image convention.
    if box.ndim > 0 and box.shape[-1] == 6:
        box = box[..., :3]

    half_box = 0.5 * box
    delta = np.where(delta > half_box, delta - box, delta)
    delta = np.where(delta < -half_box, delta + box, delta)
    return delta


__all__ = ["minimum_image_displacement"]

