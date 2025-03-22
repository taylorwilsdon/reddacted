from types import BuiltinMethodType
import requests
from reddacted.api import api
from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.exceptions import handle_exception

logger = get_logger(__name__)


class Scraper(api.API):
    """The Reddit Class obtains data to perform sentiment analysis by
    scraping the Reddit json endpoint.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def __init__(self):
        """Initialize Scraper"""
        pass

    @with_logging(logger)
    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :return: a list of comments from an article.
        """
        logger.debug_with_context(
            f"Parsing listing for subreddit={subreddit}, article={article}, limit={limit}"
        )
        url = f"https://www.reddit.com/r/{subreddit}/{article}.json?limit={limit}"
        headers = kwargs.get("headers")
        logger.debug_with_context(f"Request URL: {url}")
        logger.debug_with_context(f"Request headers: {headers}")
        try:
            response = requests.get(url, headers=headers)
            logger.debug_with_context(f"Response status code: {response.status_code}")
        except Exception as e:
            handle_exception(e, "Error obtaining article information", debug=True)
            return []

        comments = []
        json_resp = response.json()
        logger.debug_with_context(f"Retrieved {len(json_resp)} top-level JSON objects")

        for top in range(0, len(json_resp)):
            logger.debug_with_context(f"Processing top-level object {top+1}/{len(json_resp)}")
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
                                "text": comment_text,
                                "upvotes": data["ups"],
                                "downvotes": data["downs"],
                                "permalink": data["permalink"],
                                "id": data["id"],
                            }
                            logger.debug_with_context(
                                f"Added comment: ups={data['ups']}, downs={data['downs']}, text_preview='{comment_text[:50]}...'"
                            )
                            comments.append(comment_data)

        logger.debug_with_context(f"Returning {len(comments)} processed comments")
        return comments

    @with_logging(logger)
    def parse_user(self, username, limit=100, sort="new", time_filter="all", **kwargs):
        """Parses a listing and extracts the comments from it.

        :param username: a user
        :param limit: maximum number of comments to return
        :param sort: Sort method ('hot', 'new', 'controversial', 'top')
        :param time_filter: Time filter for 'top' ('all', 'day', 'hour', 'month', 'week', 'year')
        :return: a list of comments from a user.
        """
        url = f"https://www.reddit.com/user/{username}.json?limit={limit}&sort={sort}"
        if sort in ["top", "controversial"]:
            url += f"&t={time_filter}"
        logger.debug_with_context(f"Completed scraping for user {username}")
        headers = kwargs.get("headers")
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            handle_exception(e, "Error obtaining user information", debug=True)
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
                        comments.append(
                            {
                                "text": comment_text,
                                "upvotes": data["ups"],
                                "downvotes": data["downs"],
                                "permalink": data["permalink"],
                                "id": data["id"],
                            }
                        )
        logger.debug_with_context(f"Reddact is scraping {url}...")
        return comments
