import numpy as np
import pytest

from ragplay.embeddings import cosine, top_k


def test_cosine_basic_geometry():
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [3.0, 0.0]])
    assert cosine(a, b)[0] == pytest.approx([1.0, 0.0, -1.0, 1.0])  # length doesn't matter


def test_cosine_shape():
    assert cosine(np.ones((2, 4)), np.ones((5, 4))).shape == (2, 5)


def test_top_k_orders_by_similarity():
    docs = np.array([[0.0, 1.0], [1.0, 0.1], [1.0, 0.0], [-1.0, 0.0]])
    hits = top_k(np.array([1.0, 0.0]), docs, k=2)
    assert [i for i, _ in hits] == [2, 1]
    assert hits[0][1] == pytest.approx(1.0)
