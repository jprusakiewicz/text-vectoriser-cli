import unittest
import pytest
from unittest.mock import patch, Mock
from text_matcher.wikipedia_connector import get_wikipedia_core_texts_contents, get_wikipedia_core_content, ArticleNotFound


class TestWikipediaConnector(unittest.TestCase):
    @pytest.mark.unittest
    @patch('requests.get')
    def test_fetches_article_content_successfully(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "extract": "Test content"
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        result = get_wikipedia_core_content("https://pl.wikipedia.org/wiki/Test")

        self.assertEqual(result, "Test content")

    @pytest.mark.unittest
    @patch('requests.get')
    def test_raises_exception_when_article_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "-1": {}
                }
            }
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ArticleNotFound):
            get_wikipedia_core_content("https://pl.wikipedia.org/wiki/Test")

    @pytest.mark.unittest
    @patch('requests.get')
    def test_raises_exception_when_request_fails(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with self.assertRaises(ArticleNotFound):
            get_wikipedia_core_content("https://pl.wikipedia.org/wiki/Test")

    @pytest.mark.unittest
    @patch('text_matcher.wikipedia_connector.get_wikipedia_core_content')
    def test_fetches_multiple_articles_successfully(self, mock_get_content):
        mock_get_content.side_effect = ["Test content 1", "Test content 2"]

        result = get_wikipedia_core_texts_contents(
            ["https://pl.wikipedia.org/wiki/Test1", "https://pl.wikipedia.org/wiki/Test2"])

        self.assertEqual(result, {
            "https://pl.wikipedia.org/wiki/Test1": "Test content 1",
            "https://pl.wikipedia.org/wiki/Test2": "Test content 2"
        })

    @pytest.mark.unittest
    @patch('text_matcher.wikipedia_connector.get_wikipedia_core_content')
    def test_skips_articles_when_not_found_and_not_raising(self, mock_get_content):
        mock_get_content.side_effect = [ArticleNotFound("Article not found"), "Test content 2"]

        result = get_wikipedia_core_texts_contents(
            ["https://pl.wikipedia.org/wiki/Test1", "https://pl.wikipedia.org/wiki/Test2"])

        self.assertEqual(result, {
            "https://pl.wikipedia.org/wiki/Test2": "Test content 2"
        })

    @pytest.mark.unittest
    @patch('text_matcher.wikipedia_connector.get_wikipedia_core_content')
    def test_raises_exception_when_article_not_found_and_raising(self, mock_get_content):
        mock_get_content.side_effect = [ArticleNotFound("Article not found"), "Test content 2"]

        with self.assertRaises(ArticleNotFound):
            get_wikipedia_core_texts_contents(
                ["https://pl.wikipedia.org/wiki/Test1", "https://pl.wikipedia.org/wiki/Test2"], raise_on_error=True)


if __name__ == '__main__':
    unittest.main()
