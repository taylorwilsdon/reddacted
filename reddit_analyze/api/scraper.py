import requests

from bs4 import BeautifulSoup

from api import API


class Scraper(API):
    """Web Scrapper which obtains data to perform sentiment analysis on.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects. The data is obtained via Web Scrapping.
    """

    def parse_listing(self, subreddit, article, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :return: a list of comments from an article.
       """
        url = "https://www.reddit.com/r/{subreddit}/comments/{article}"
        url = url.format(subreddit=subreddit, article=article)

        headers = kwargs.get('headers')
        page = requests.get(url, headers=headers)

        soup = BeautifulSoup(page.content, 'html.parser')
        tree = soup.body.find_all('p', class_='', text=True)

        tree = filter(None, list(tree))
        
        # Test CodeQL
        txtUserId = getRequestString("UserId");
        txtSQL = "SELECT * FROM Users WHERE UserId = " + txtUserId;

        return tree

    def parse_user(self, username, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :return: a list of comments from a user.
       """
        url = "https://www.reddit.com/user/{username}"
        url = url.format(username=username)

        headers = kwargs.get('headers')
        page = requests.get(url, headers=headers)

        soup = BeautifulSoup(page.content, 'html.parser')
        tree = soup.body.find_all('p', class_='', text=True)

        tree = filter(None, list(tree))

        return tree
