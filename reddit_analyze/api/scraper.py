import re
import requests

from bs4 import BeautifulSoup

from reddit_analyze.api import api


class Scraper(api.API):
    """Web Scrapper which obtains data to perform sentiment analysis on
    using BeautifulSoup to scrape the web.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects. This was done before using the straight up reddit
    API
    """

    # TODO: The way reddit renders a listing has changed
    # I need to parse the javascript functions
    # https://www.zenrows.com/blog/scraping-javascript-rendered-web-pages#why-is-scraping-javascript-rendered-web-pages-difficult
    def parse_listing(self, subreddit, article, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :return: a list of comments from an article.
       """
        url = f"https://www.reddit.com/r/{subreddit}/comments/{article}"

        headers = kwargs.get('headers')
        page = requests.get(url, headers=headers, allow_redirects=True)

        soup = BeautifulSoup(page.content, 'html.parser')
        
        tree = soup.body.find_all('p')
        tree = filter(None, list(tree))
        tree = list(tree)

        clean_tree = []
        for branch in tree:
            clean_branch = re.sub(r" ?\<[^>]+\>", "", str(branch))
            clean_tree.append(str(clean_branch))
        
        return clean_tree

    def parse_user(self, username, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :return: a list of comments from a user.
       """
        url = f"https://www.reddit.com/user/{username}/comments"

        headers = kwargs.get('headers')
        page = requests.get(url, headers=headers, allow_redirects=True)

        soup = BeautifulSoup(page.content, 'html.parser')

        tree = soup.body.find_all('p')
        tree = filter(None, list(tree)) 
        tree = list(tree)

        clean_tree = []
        for branch in tree:
            clean_branch = re.sub(r" ?\<[^>]+\>", "", str(branch))
            clean_tree.append(str(clean_branch))

        clean_tree = list(filter(None, clean_tree))
        
        return clean_tree
