import logging
from types import BuiltinMethodType

import os
import praw

from reddit_analyze.api import api


class Reddit(api.API):
    """The Reddit Class obtains data to perform sentiment analysis on
    using the Reddit API.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def __init__(self):
        username = os.environ["REDDIT_USERNAME"]
        if not username:
            logging.error("Request will be un-authenticated, missing the 'REDDIT_USERNAME' environment variable")
            return
        password = os.environ["REDDIT_PASSWORD"]
        if not password:
            logging.error("Request will be un-authenticated, missing the 'REDDIT_PASSWORD' environment variable")
            return
        client_id = os.environ["REDDIT_CLIENT_ID"]
        if not client_id:
            logging.error("Request will be un-authenticated, missing the 'REDDIT_CLIENT_ID' environment variable")
            return
        client_secret = os.environ["REDDIT_CLIENT_SECRET"]
        if not client_secret:
            logging.error("Request will be un-authenticated, missing the 'REDDIT_CLIENT_SECRET' environment variable")
            return
  
        # Setup the reddit client
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            password=password,
            user_agent="Reddit Sentiment Analyzer u/{username}",
            username=username,
        )
        reddit.read_only = True
        self.reddit = reddit

    def parse_listing(self, subreddit, article, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :return: a list of comments from an article.
       """
        url = f"https://www.reddit.com/r/{subreddit}/comments/{article}"
        submission = self.reddit.submission(url=url)
        comments = submission.comments.new(limit=None)
        
        return comments

    def parse_user(self, username, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :return: a list of comments from a user.
       """
        redditor = self.reddit.redditor({username})
        comments = redditor.comments.new(limit=None)
        
        return comments