import re
from typing import List, Dict

import requests

FILTER = ["== Zobacz też ==", "== Przypisy ==", "== Linki zewnętrzne ==", "== Bibliografia =="]


class ArticleNotFound(Exception):
    """
    Exception raised for errors related to article retrieval.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def get_wikipedia_core_texts_contents(urls: List[str], raise_on_error=False) -> Dict[str, str]:
    documents = {}
    for url in urls:
        try:
            text = get_wikipedia_core_text_content(url)
            cleaned_text = remove_sections_and_clean_text(text, FILTER)
            documents.update({url: cleaned_text})
        except ArticleNotFound:
            if raise_on_error:
                raise ArticleNotFound(f"Article not found: {url}")
    return documents


def get_wikipedia_core_text_content(url: str) -> str:
    title = _get_title_from_url(url)
    text = _fetch_wikipedia_article(title)
    cleaned_text = remove_sections_and_clean_text(text, FILTER)
    return cleaned_text


def remove_sections_and_clean_text(text: str, headers: list) -> str:
    """
    Remove sections from the text based on header names and clean up the text.

    Args:
        text (str): The input text from which sections will be removed.
        headers (list): List of headers to be removed, including their section format.

    Returns:
        str: The cleaned text with specified sections and headers removed.
    """
    # Remove each specified section including the header
    for header in headers:
        pattern = re.compile(rf"{re.escape(header)}.*?(?=^==\s|\Z)", re.DOTALL | re.MULTILINE)
        text = re.sub(pattern, '', text)

    # Remove any remaining headers and extra spaces/newlines
    text = re.sub(r'^==.*?==\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def validate(pages: dict):
    if "-1" in pages:
        raise ArticleNotFound("Article not found")


def _fetch_wikipedia_article(title: str) -> str:
    """
    Fetches the plain text content of a Wikipedia article using the Wikipedia API.

    Args:
        title (str): The title of the Wikipedia article (e.g., "AIML").

    Returns:
        str: The plain text content of the article.
    """
    url = "https://pl.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "format": "json",
        "explaintext": True,
        "titles": title
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data["query"]["pages"]
        validate(pages)
        page_content = next(iter(pages.values()))["extract"]
        return page_content
    else:
        raise ArticleNotFound(f"Failed to fetch article: {response.status_code}")


def _get_title_from_url(url: str) -> str:
    """
    Extracts the title of a Wikipedia article from its URL.

    Args:
        url (str): The URL of the Wikipedia article.

    Returns:
        str: The title of the article.
    """
    title = url.split("/")[-1]
    return title
