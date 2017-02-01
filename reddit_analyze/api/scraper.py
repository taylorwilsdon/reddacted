import requests

from bs4 import BeautifulSoup

from api import API


class Scraper(API):
    """Web Scrapper which obtains data to perform sentiment analysis on.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects. The data is obtained via Web Scrapping.
    """

    def parse_listing(self, subreddit, article):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit

       :return: a list of comments from an article.
       """
        url = "https://www.reddit.com/r/{subreddit}/comments/{article}"
        url = url.format(subreddit=subreddit, article=article)
        # TODO: pass user-agent in header
        # http://stackoverflow.com/questions/30992791/http-429-too-many-requests-when-accessing-a-reddit-json-page-only-once-using-ja
        page = requests.get(url)

        soup = BeautifulSoup(page.content, 'html.parser')
        tree = soup.body.find_all('p', class_='', text=True)

        tree = filter(None, list(tree))
        return tree

    def parse_user(self, username):
        """Parses a listing and extracts the comments from it.

       :param username: a user

       :return: a list of comments from a user.
       """
        url = "https://www.reddit.com/user/{username}"
        url = url.format(username=username)
        # TODO: pass user-agent in header
        # http://stackoverflow.com/questions/30992791/http-429-too-many-requests-when-accessing-a-reddit-json-page-only-once-using-ja
        page = requests.get(url)

        soup = BeautifulSoup(page.content, 'html.parser')
        tree = soup.body.find_all('p', class_='', text=True)

        tree = filter(None, list(tree))

        return tree
