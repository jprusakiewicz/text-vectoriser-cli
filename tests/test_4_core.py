import tempfile
import os

import pytest

from text_matcher.core import train_and_save_vectorizer, load_vectorizer_and_pick_best
from text_matcher.vectorizer_config import CountVectorizerConfig, HashingVectorizerConfig, TfidfVectorizerConfig


@pytest.mark.internet
@pytest.mark.parametrize(
    "url, metric, vectorizer_type",
    [
        ("https://pl.wikipedia.org/wiki/Technologia_haptyczna", 'cosine', CountVectorizerConfig()),
        ("https://pl.wikipedia.org/wiki/Sterowanie_predykcyjne", 'euclidean', HashingVectorizerConfig()),
        ("https://pl.wikipedia.org/wiki/Automat", 'manhattan', TfidfVectorizerConfig()),
    ]
)
def test_e2e(url, metric, vectorizer_type):
    """
    A dummy test that checks if the same article is matched with itself. May be useful i.e. for regression tests.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    try:
        train_and_save_vectorizer(temp_file.name, "data/train.csv", vectorizer_type)
        best_match = load_vectorizer_and_pick_best(metric,
                                                   url,
                                                   "data/test.csv", temp_file.name)

        assert best_match == url
    finally:
        os.unlink(temp_file.name)
