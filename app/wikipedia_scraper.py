from typing import List, Dict

import requests
from bs4 import BeautifulSoup


def scrape_wikipedia_core_texts_contents(urls: List[str]) -> Dict[str, str]:
    documents = {}
    for url in urls:
        text = scrape_wikipedia_core_text_content(url)
        if text:
            documents.update({url: text})
    return documents


def scrape_wikipedia_core_text_content(url: str) -> str:
    content: bytes = _get_html_content(url)
    text = _parse_html_content(content)
    return text


def _get_html_content(url: str) -> bytes:
    return requests.get(url).content


def _parse_html_content(content: bytes) -> str:
    soup = BeautifulSoup(content)

    # Try to locate the main content of a Wikipedia article, if present
    main_content = soup.find('div', {'id': 'mw-content-text'})

    # Fallback to the entire body if main content is not found
    if not main_content:
        main_content = soup.find('body')

    if not main_content:
        return ""  # Return empty if no content is found

    # Remove unwanted elements (e.g., navigation, edit links, etc.)
    for element in main_content.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside']):
        element.decompose()

    # Remove edit and tool links from Wikipedia-like content
    for unwanted_class in ['mw-editsection', 'toc', 'mw-normal-catlinks', 'navbox', 'metadata']:
        for element in main_content.find_all('div', {'class': unwanted_class}):
            element.decompose()

    # Extract and clean the text
    text = main_content.get_text(separator=' ')
    cleaned_text = ' '.join(text.split())

    return cleaned_text
