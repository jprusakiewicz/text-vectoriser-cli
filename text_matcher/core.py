from typing import List

from text_matcher.vectorizer import train_vectorizer, save_vectorizer, load_vectorizer, transform_and_pick_best_document
from text_matcher.vectorizer_config import VectorizerConfig
from text_matcher.wikipedia_connector import get_wikipedia_core_text_content, get_wikipedia_core_texts_contents


def train_and_save_vectorizer(output_model_path: str, train_file: str, vectorizer_config: VectorizerConfig):
    urls = load_data(train_file)
    documents = get_wikipedia_core_texts_contents(urls)
    # this is a good place for data preprocess like a stemming, lemmatization, stopwords removal, lowercase, etc.
    vectorizer = train_vectorizer(vectorizer_config, list(documents.values()))
    save_vectorizer(vectorizer, output_model_path)


def load_vectorizer_and_pick_best(distance_metric: str, query_url: str, test_file: str, vectorizer_path: str) -> str:
    vectorizer = load_vectorizer(vectorizer_path)

    test_urls = load_data(test_file)

    test_documents_unprocessed = get_wikipedia_core_texts_contents(
        test_urls)
    query_text = get_wikipedia_core_text_content(query_url)

    best_idx = transform_and_pick_best_document(vectorizer, list(test_documents_unprocessed.values()), query_text,
                                                distance_metric)
    best_match_url = list(test_documents_unprocessed.keys())[best_idx]
    return best_match_url


def load_vectorizer_and_pick_best_for_all(distance_metric: str, query_urls: List[str], test_file: str,
                                          vectorizer_path: str) -> dict:
    vectorizer = load_vectorizer(vectorizer_path)

    test_urls = load_data(test_file)

    test_documents_unprocessed = get_wikipedia_core_texts_contents(
        test_urls)
    query_texts = get_wikipedia_core_texts_contents(query_urls)

    best_matches = {}
    for url, text in query_texts.items():
        best_idx = transform_and_pick_best_document(vectorizer, list(test_documents_unprocessed.values()), text,
                                                    distance_metric)
        best_match_url = list(test_documents_unprocessed.keys())[best_idx]
        best_matches[url] = best_match_url
    return best_matches


def reverse_lookup(d, value):
    return next((k for k, v in d.items() if v == value), None)


def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]
