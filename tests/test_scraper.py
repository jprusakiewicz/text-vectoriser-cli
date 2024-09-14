import unittest
from unittest.mock import patch, Mock

import pytest

from app.wikipedia_scraper import scrape_wikipedia_core_text_content


class TestScraper(unittest.TestCase):
    @patch('requests.get')
    @pytest.mark.unittest
    def test_fetch_and_parse_html(self, mock_get):
        """Tests fetching HTML content and parsing it."""
        # given
        mock_response = Mock()
        mock_response.content = b'<html><body>Test</body></html>'
        mock_get.return_value = mock_response

        expected_output = 'Test'
        # when
        cleaned_text = scrape_wikipedia_core_text_content("https://fake_wiki_address.com")
        # then
        self.assertTrue(cleaned_text)
        self.assertEqual(cleaned_text, expected_output)

    @pytest.mark.internet
    def test_clean_and_extract_text(self):
        """Tests cleaning HTML and extracting text."""
        # given
        url = "https://pl.wikipedia.org/wiki/AIML"
        # when
        text = scrape_wikipedia_core_text_content(url)
        # then
        self.assertIn('( Artificial Intelligence Markup Language )', text)
        self.assertNotIn("Wersja do druku", text)
        self.assertNotIn("kod źródłowy", text)
        self.assertNotIn("Zobacz też", text)


if __name__ == '__main__':
    unittest.main()
