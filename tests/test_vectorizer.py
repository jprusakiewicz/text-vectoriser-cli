import unittest

from app.vectorizer import transform_and_pick_best_document, train_vectorizer
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from app.vectorizer import pick_best_document


class TestVectorizer(unittest.TestCase):
    @pytest.mark.unittest
    def test_vectorizer_with_tfidf(self):
        train_documents = [
            "The cat is sitting on the mat.",
            "Dogs are loyal animals and they bark.",
            "Birds can fly high in the sky.",
            "Fish swim in the ocean and lakes.",
            "The sun rises in the east every morning."
        ]
        test_documents = [
            "A dog is playing in the yard.",
            "Cats love to chase mice.",
            "Birds build nests in tall trees to lay eggs.",
            "Fish are found in both freshwater and saltwater environments.",
            "The sun sets in the west at the end of the day."
        ]
        query_text = "The sun rises and sets every day."
        vectorizer_type = 'tfidf'
        distance_metric = 'cosine'
        # when
        vectorizer = train_vectorizer(vectorizer_type, train_documents)

        best_idx = transform_and_pick_best_document(vectorizer=vectorizer, distance_metric=distance_metric,
                                                    query_text=query_text, test_documents=test_documents)

        # then
        assert isinstance(best_idx, int)
        assert best_idx == 4


class TestSimilarityPicker(unittest.TestCase):

    @pytest.mark.unittest
    def test_pick_best_document_with_unsupported_distance_metric(self):
        query_vec = np.array([1, 0, 1])
        test_vecs = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        distance_metric = 'unsupported'

        with pytest.raises(ValueError):
            pick_best_document(query_vec, test_vecs, distance_metric)

    @pytest.mark.unittest
    def test_pick_best_document_with_sparse_vectors(self):
        query_vec = csr_matrix(np.array([1, 0, 1]))
        test_vecs = csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]]))
        distance_metric = 'cosine'

        best_idx = pick_best_document(query_vec, test_vecs, distance_metric)

        assert isinstance(best_idx, int)
        assert best_idx == 0


@pytest.mark.unittest
@pytest.mark.parametrize(
    "distance_metric, expected",
    [
        ('cosine', 0),
        ('euclidean', 0),
        ('manhattan', 0),
    ]
)
def test_pick_best_document(distance_metric, expected):
    query_vec = np.array([1, 0, 1])
    test_vecs = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    best_idx = pick_best_document(query_vec, test_vecs, distance_metric)

    assert isinstance(best_idx, int)
    assert best_idx == expected
