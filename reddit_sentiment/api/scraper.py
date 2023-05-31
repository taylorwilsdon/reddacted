import logging
from types import BuiltinMethodType
import requests

from reddit_analyze.api import api


class Scraper(api.API):
    """The Reddit Class obtains data to perform sentiment analysis by
    scraping the Reddit json endpoint.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def parse_listing(self, subreddit, article, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :return: a list of comments from an article.
       """
        url = f"https://www.reddit.com/r/{subreddit}/{article}.json"
        headers = kwargs.get('headers')
        
        try:
            response = requests.get(url, headers = headers)
        except Exception as e:
            logging.error("Error obtaining article information: %s" % e)
            return []
        
        comments = []
        json_resp = response.json()

        for top in range(0, len(json_resp)):
            if json_resp[top]["data"]["children"]:
                children = json_resp[top]["data"]["children"]
                for child in range(0, len(children)):
                    data = children[child]["data"]
                    if "body" in data:
                         # remove empty spaces and weird reddit strings
                        comment = data["body"].rstrip()
                        comment = " ".join(comment.split())
                        comment = comment.replace("&amp;#x200B;", "")
                        
                        if comment != "":
                            comments.append(comment)

        return comments

    def parse_user(self, username, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :return: a list of comments from a user.
       """
        url = f"https://www.reddit.com/user/{username}.json"
        headers = kwargs.get('headers')
        try:
            response = requests.get(url, headers = headers)
        except Exception as e:
            logging.error("Error obtaining user information: %s" % e)
            return []

        comments = []
        json_resp = response.json()

        if json_resp["data"]["children"]:
            children = json_resp["data"]["children"]
            for child in range(0, len(children)):
                data = children[child]["data"]
                if "body" in data:
                    # remove empty spaces and weird reddit strings
                    comment = data["body"].rstrip()
                    comment = " ".join(comment.split())
                    comment = comment.replace("&amp;#x200B;", "")

                    if comment != "":
                        comments.append(comment)
        
        return comments
