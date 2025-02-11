import logging
from types import BuiltinMethodType
import time
import os
import praw
from reddacted.api import api


class AuthenticationRequiredError(Exception):
    """Raised when authentication is required but not configured"""
    pass


class Reddit(api.API):
    """The Reddit Class obtains data to perform sentiment analysis on
    using the Reddit API.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def __init__(self):
        self.authenticated = False

        # Check for all required credentials first
        required_vars = {
            "REDDIT_USERNAME": os.environ.get("REDDIT_USERNAME"),
            "REDDIT_PASSWORD": os.environ.get("REDDIT_PASSWORD"),
            "REDDIT_CLIENT_ID": os.environ.get("REDDIT_CLIENT_ID"),
            "REDDIT_CLIENT_SECRET": os.environ.get("REDDIT_CLIENT_SECRET")
        }

        if None in required_vars.values():
            missing = [k for k, v in required_vars.items() if v is None]
            logging.error(f"Missing authentication variables: {', '.join(missing)}")
            return

        try:
            # Initialize authenticated client
            self.reddit = praw.Reddit(
                client_id=required_vars["REDDIT_CLIENT_ID"],
                client_secret=required_vars["REDDIT_CLIENT_SECRET"],
                password=required_vars["REDDIT_PASSWORD"],
                user_agent=f"reddacted u/{required_vars['REDDIT_USERNAME']}",
                username=required_vars["REDDIT_USERNAME"],
            )
            self.authenticated = True
        except Exception as e:
            logging.error(f"Authentication failed: {str(e)}")

    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param subreddit: a subreddit
       :param article: an article associated with the subreddit
       :param limit: maximum number of comments to return (None for unlimited)
       :return: a list of comments from an article.
       """
        url = f"https://www.reddit.com/r/{subreddit}/comments/{article}"
        submission = self.reddit.submission(url=url)
        comments = submission.comments.new(limit=limit)

        return comments

    def delete_comments(self, comment_ids: list[str], batch_size: int = 10) -> dict[str, any]:
        """
        Delete comments in batches with rate limiting
        :param comment_ids: List of comment IDs to delete
        :param batch_size: Number of comments to process per batch
        :return: Dict with results and statistics
        """
        if not self.authenticated:
            raise AuthenticationRequiredError("Full authentication required for comment deletion")

        results = {
            'processed': 0,
            'success': 0,
            'failures': 0,
            'errors': []
        }

        for i in range(0, len(comment_ids), batch_size):
            batch = comment_ids[i:i+batch_size]
            try:
                for comment_id in batch:
                    try:
                        comment = self.reddit.comment(id=comment_id)
                        comment.delete()
                        results['success'] += 1
                    except Exception as e:
                        results['failures'] += 1
                        results['errors'].append({
                            'comment_id': comment_id,
                            'error': str(e)
                        })
                    # Respect Reddit's API rate limit (1 req/sec)
                    time.sleep(1.1)

                results['processed'] += len(batch)
            except praw.exceptions.APIException as e:
                logging.error(f"API rate limit exceeded: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
                continue

        return results

    def parse_user(self, username, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

       :param username: a user
       :param limit: maximum number of comments to return (None for unlimited)
       :return: a list of comments from a user.
       """
        redditor = self.reddit.redditor({username})
        comments = redditor.comments.new(limit=limit)

        return comments
