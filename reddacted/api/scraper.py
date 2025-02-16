import logging
import sys
from types import BuiltinMethodType
import requests
from functools import wraps

from reddacted.api import api
from reddacted.utils.exceptions import handle_exception

def log_context(func):
    """Decorator to add context information to log messages"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get the calling context once
        context = f"[{__name__}:{func.__name__}:{sys._getframe().f_lineno}]"
        
        # Create a wrapper logger that prepends context
        original_debug = self.logger.debug
        original_error = self.logger.error
        
        def debug_with_context(msg, *args, **kwargs):
            original_debug(f"{context} {msg}", *args, **kwargs)
            
        def error_with_context(msg, *args, **kwargs):
            original_error(f"{context} {msg}", *args, **kwargs)
        
        # Temporarily replace logger methods
        self.logger.debug = debug_with_context
        self.logger.error = error_with_context
        
        try:
            return func(self, *args, **kwargs)
        finally:
            # Restore original logger methods
            self.logger.debug = original_debug
            self.logger.error = original_error
    
    return wrapper

class Scraper(api.API):
    """The Reddit Class obtains data to perform sentiment analysis by
    scraping the Reddit json endpoint.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def __init__(self):
        """Initialize Scraper with logging"""
        self.logger = logging.getLogger(__name__)

    @log_context
    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :return: a list of comments from an article.
       """
        self.logger.debug(f"Parsing listing for subreddit={subreddit}, article={article}, limit={limit}")
        url = f"https://www.reddit.com/r/{subreddit}/{article}.json?limit={limit}"
        headers = kwargs.get('headers')
        self.logger.debug(f"Request URL: {url}")
        self.logger.debug(f"Request headers: {headers}")
        try:
            response = requests.get(url, headers=headers)
            self.logger.debug(f"Response status code: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error obtaining article information: {e}")
            return []

        comments = []
        json_resp = response.json()
        self.logger.debug(f"Retrieved {len(json_resp)} top-level JSON objects")

        for top in range(0, len(json_resp)):
            self.logger.debug(f"Processing top-level object {top+1}/{len(json_resp)}")
            if json_resp[top]["data"]["children"]:
                children = json_resp[top]["data"]["children"]
                for child in range(0, len(children)):
                    data = children[child]["data"]
                    if "body" in data:
                         # remove empty spaces and weird reddit strings
                        comment_text = data["body"].rstrip()
                        comment_text = " ".join(comment_text.split())
                        comment_text = comment_text.replace("&amp;#x200B;", "")
                        if comment_text != "":
                            comment_data = {
                                'text': comment_text,
                                'upvotes': data["ups"],
                                'downvotes': data["downs"],
                                'permalink': data["permalink"]
                            }
                            self.logger.debug(f"Added comment: ups={data['ups']}, downs={data['downs']}, text_preview='{comment_text[:50]}...'")
                            comments.append(comment_data)

        self.logger.debug(f"Returning {len(comments)} processed comments")
        return comments

    @log_context
    def parse_user(self, username, limit=100, sort='new', time_filter='all', **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :param limit: maximum number of comments to return
       :param sort: Sort method ('hot', 'new', 'controversial', 'top')
       :param time_filter: Time filter for 'top' ('all', 'day', 'hour', 'month', 'week', 'year')
       :return: a list of comments from a user.
       """
        url = f"https://www.reddit.com/user/{username}.json?limit={limit}&sort={sort}"
        if sort in ['top', 'controversial']:
            url += f"&t={time_filter}"
        self.logger.debug(f"Completed scraping for user {username}")
        headers = kwargs.get('headers')
        try:
            response = requests.get(url, headers = headers)
        except Exception as e:
            self.logger.error(f"Error obtaining user information: {e}")
            return []

        comments = []
        json_resp = response.json()

        if json_resp["data"]["children"]:
            children = json_resp["data"]["children"]
            for child in range(0, len(children)):
                data = children[child]["data"]
                if "body" in data:
                    # remove empty spaces and weird reddit strings
                    comment_text = data["body"].rstrip()
                    comment_text = " ".join(comment_text.split())
                    comment_text = comment_text.replace("&amp;#x200B;", "")
                    if comment_text != "":
                        comments.append({
                            'text': comment_text,
                            'upvotes': data["ups"],
                            'downvotes': data["downs"],
                            'permalink': data["permalink"]
                        })
        self.logger.debug(f"Reddact is scraping {url}...")
        return comments
