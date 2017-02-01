import bs4
import mock
import requests
import unittest
import warnings

from reddit_analyze.api import scraper


class APIPostiveTestCases(unittest.TestCase):

    def setUp(self):
        super(APIPostiveTestCases, self).setUp()
        warnings.filterwarnings("ignore")

        # Setup the BS4 Objects we will expect.
        soup_a = bs4.BeautifulSoup("<p></p>", 'html.parser')
        tag_a = soup_a.p
        tag_a.append(" ok ")

        soup_b = bs4.BeautifulSoup("<p></p>", 'html.parser')
        tag_b = soup_b.p
        tag_b.append(" awesome ")

        self.content = [tag_a, tag_b]

        sentence_a = '<p> ok </p>'
        sentence_b = '<p> awesome </p>'

        # Valid Response to check against, shares fields requests.Response()
        valid_response = mock.MagicMock(name='mock_response')
        valid_response.content = "<html> <body> {sentence_a} {sentence_b} </body> </html>".format(sentence_a=sentence_a, sentence_b=sentence_b)
        valid_response.status_code = 200

        requests.get = mock.MagicMock(return_value=valid_response)

    def tearDown(self):
        super(APIPostiveTestCases, self).tearDown()

    def test_parse_user(self):
        sc = scraper.Scraper()
        tree = sc.parse_user("bob")

        self.assertEqual(self.content, tree)

    def test_parse_listing(self):
        sc = scraper.Scraper()
        tree = sc.parse_listing("funny", "AWTRY17")

        self.assertEqual(self.content, tree)
