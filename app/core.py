from typing import List

# from app.wikipedia_scraper import scrape_wikipedia_core_texts_contents
from app.vectorizer import train_vectorizer, save_vectorizer, load_vectorizer, transform_and_pick_best_document
from app.wikipedia_connector import get_wikipedia_core_content, get_wikipedia_core_texts_contents


def train_and_save_vectorizer(output_model_path: str, train_file: str, vectorizer_type: str):
    urls = load_data(train_file)
    # documents = scrape_wikipedia_core_texts_contents(urls)
    documents = get_wikipedia_core_texts_contents(urls)
    vectorizer = train_vectorizer(vectorizer_type, list(documents.values()))
    save_vectorizer(vectorizer, output_model_path)


def load_vectorizer_and_pick_best(distance_metric: str, query_url: str, test_file: str, vectorizer_path: str):
    vectorizer = load_vectorizer(vectorizer_path)

    test_urls = load_data(test_file)
    # test_documents = scrape_wikipedia_core_texts_contents(test_urls)
    test_documents = get_wikipedia_core_texts_contents(test_urls)
    query_text = get_wikipedia_core_content(query_url)

    best_idx = transform_and_pick_best_document(vectorizer, list(test_documents.values()), query_text, distance_metric)
    best_match_text = list(test_documents.values())[best_idx]
    best_match_url = reverse_lookup(test_documents, best_match_text)
    return best_match_url


def reverse_lookup(d, value):
    return next((k for k, v in d.items() if v == value), None)


def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]
