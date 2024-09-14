import pickle
from typing import Union, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances


def train_vectorizer(vectorizer_type: str, documents):
    match vectorizer_type:
        case 'count':
            vectorizer = CountVectorizer()
        case 'tfidf':
            vectorizer = TfidfVectorizer()
        case 'hashing':
            vectorizer = HashingVectorizer()
        case _:
            raise ValueError("Unsupported vectorizer type.")

    vectorizer.fit(documents)
    return vectorizer


def transform_and_pick_best_document(vectorizer: Union[CountVectorizer, TfidfVectorizer, HashingVectorizer],
                                     test_documents: List[str],
                                     query_text: str,
                                     distance_metric: str) -> int:
    # todo this is a good place for data preprocess like a stemming, lemmatization, stopwords removal, lowercase, etc.
    query_vec = vectorizer.transform([query_text]).toarray()[0]
    test_vecs = vectorizer.transform(test_documents)
    best_idx = pick_best_document(query_vec, test_vecs, distance_metric)
    return best_idx


def save_vectorizer(vectorizer, output_model_file):
    with open(output_model_file, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(vectorizer_model_file):
    with open(vectorizer_model_file, 'rb') as f:
        return pickle.load(f)


def pick_best_document(query_vec: Union[csr_matrix, np.ndarray], test_vecs: Union[csr_matrix, np.ndarray],
                       distance_metric: str) -> int:
    """
    Finds the document from the test set that is closest to the query vector based on the given distance metric.

    Args:
        query_vec (Union[np.ndarray, csr_matrix]): Vector representing the query document.
        test_vecs (Union[np.ndarray, csr_matrix]): Matrix of vectors representing the documents to search through.
        distance_metric (str): The distance metric to use ('cosine', 'euclidean', 'manhattan').

    Returns:
        int: Index of the closest document in `test_vecs`.
    """
    distance_func = get_distance_function(distance_metric)

    if hasattr(test_vecs, "toarray"):
        test_vecs = test_vecs.toarray()

    if hasattr(query_vec, "toarray"):
        query_vec = query_vec.toarray()

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    distances = distance_func(test_vecs, query_vec)

    if distance_metric == 'cosine':
        best_index = np.argmax(distances).item()
    else:
        best_index = np.argmin(distances).item()

    return best_index


def get_distance_function(distance_metric: str):
    match distance_metric:
        case 'cosine':
            distance_func = cosine_similarity
        case 'euclidean':
            distance_func = euclidean_distances
        case 'manhattan':
            distance_func = manhattan_distances
        case _:
            raise ValueError(
                f"Unsupported distance metric: {distance_metric}. Choose from 'cosine', 'euclidean', 'manhattan'.")
    return distance_func
