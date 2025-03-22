from types import BuiltinMethodType
import time
import os
from typing import List, Dict, Any
import praw
from reddacted.api import api
from reddacted.utils.logging import get_logger, with_logging
from reddacted.utils.exceptions import handle_exception

logger = get_logger(__name__)


class AuthenticationRequiredError(Exception):
    """Raised when authentication is required but not configured"""

    pass


@with_logging(logger)
class Reddit(api.API):
    """The Reddit Class obtains data to perform sentiment analysis on
    using the Reddit API.

    It allows an unauthenticated user to obtain data to analyze various
    reddit objects.
    """

    def __init__(self):
        """Initialize Reddit API client. Will attempt authenticated access first,
        falling back to read-only mode if credentials are not provided."""
        self.authenticated = False
        self.reddit = None  # Initialize to None by default

        # Check for all required credentials first
        required_vars = {
            "REDDIT_USERNAME": os.environ.get("REDDIT_USERNAME"),
            "REDDIT_PASSWORD": os.environ.get("REDDIT_PASSWORD"),
            "REDDIT_CLIENT_ID": os.environ.get("REDDIT_CLIENT_ID"),
            "REDDIT_CLIENT_SECRET": os.environ.get("REDDIT_CLIENT_SECRET"),
        }

        if None in required_vars.values():
            missing = [k for k, v in required_vars.items() if v is None]
            logger.warning(
                f"Reddit API authentication requires environment variables: {', '.join(missing)}. "
                "Falling back to read-only mode. Some features like comment deletion will be unavailable."
            )
            try:
                # Initialize read-only client
                self.reddit = praw.Reddit(user_agent="reddacted:read_only_client")
                logger.info("Successfully initialized read-only Reddit client")
            except Exception as e:
                logger.error(f"Failed to initialize read-only client: {str(e)}")
            return

        logger.debug_with_context("Attempting to initialize authenticated Reddit client")
        try:
            # Initialize authenticated client
            self.reddit = praw.Reddit(
                client_id=required_vars["REDDIT_CLIENT_ID"],
                client_secret=required_vars["REDDIT_CLIENT_SECRET"],
                password=required_vars["REDDIT_PASSWORD"],
                user_agent=f"reddacted u/{required_vars['REDDIT_USERNAME']}",
                username=required_vars["REDDIT_USERNAME"],
                check_for_async=False,
            )
            self.authenticated = True
            logger.debug_with_context("Successfully authenticated with Reddit API")
        except Exception as e:
            handle_exception(e, "Authentication Failed")

    @with_logging(logger)
    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param limit: maximum number of comments to return (None for unlimited)
        :return: a list of comments from an article
        """
        if self.reddit is None:
            logger.error("Reddit client initialization failed - cannot fetch comments")
            return []

        mode = "authenticated" if self.authenticated else "read-only"
        logger.info(f"Fetching comments for article '{article}' in {mode} mode")
        logger.debug_with_context(
            f"Parsing listing for subreddit={subreddit}, article={article}, limit={limit}"
        )
        submission = self.reddit.submission(id=article)
        logger.debug_with_context(f"Retrieved submission: title='{submission.title}'")
        logger.debug_with_context("Expanding 'more comments' links")
        submission.comments.replace_more(limit=None)
        comments = []

        for comment in submission.comments.list():
            comment_data = {
                "text": comment.body.rstrip(),
                "upvotes": comment.ups,
                "downvotes": comment.downs,
                "permalink": comment.permalink,
                "id": comment.id,
            }
            logger.debug_with_context(
                f"Processing comment: ups={comment.ups}, downs={comment.downs}, text_preview='{comment.body[:50]}...'"
            )
            comments.append(comment_data)

        return comments[:limit] if limit else comments

    def _process_comments(
        self, comment_ids: list[str], action: str, batch_size: int = 10
    ) -> dict[str, any]:
        """
        Process comments in batches with rate limiting
        :param comment_ids: List of comment IDs to process
        :param action: Action to perform ('delete' or 'update')
        :param batch_size: Number of comments to process per batch
        :return: Dict with results and statistics
        """
        logger.debug("Starting _process_comments")
        if not self.authenticated:
            raise AuthenticationRequiredError(f"Full authentication required for comment {action}")

        results = {
            "processed": 0,
            "success": 0,
            "failures": 0,
            "successful_ids": [],
            "failed_ids": [],
            "errors": [],
        }

        for i in range(0, len(comment_ids), batch_size):
            batch = comment_ids[i : i + batch_size]
            try:
                for comment_id in batch:
                    try:
                        comment = self.reddit.comment(id=comment_id)
                        if action == "delete":
                            logger.debug(f"Deleting comment ID {comment}")
                            comment.delete()
                            results["successful_ids"].append(comment_id)
                            results["success"] += 1
                        elif action == "update":
                            logger.debug(f"Updating comment ID {comment}")
                            comment.edit(
                                "This comment has been reddacted to preserve online privacy - see r/reddacted for more info"
                            )
                            results["successful_ids"].append(comment_id)
                            results["success"] += 1
                    except Exception as e:
                        results["failures"] += 1
                        results["failed_ids"].append(comment_id)
                        results["errors"].append({"comment_id": comment_id, "error": str(e)})
                    # Respect Reddit's API rate limit (1 req/sec)
                    time.sleep(1.1)

                results["processed"] += len(batch)
            except praw.exceptions.APIException as e:
                handle_exception(e, "Reddit API Rate Limit Exceeded")
                time.sleep(60)  # Wait 1 minute before retrying
                continue

        return results

    def delete_comments(self, comment_ids: list[str], batch_size: int = 10) -> dict[str, any]:
        """
        Delete comments in batches with rate limiting
        :param comment_ids: List of comment IDs to delete
        :param batch_size: Number of comments to process per batch
        :return: Dict with results and statistics
        """
        return self._process_comments(comment_ids, "delete", batch_size)

    def update_comments(self, comment_ids: list[str], batch_size: int = 10) -> dict[str, any]:
        """
        Update comments in batches with rate limiting to replace content with 'r/reddacted'
        :param comment_ids: List of comment IDs to update
        :param batch_size: Number of comments to process per batch
        :return: Dict with results and statistics
        """
        return self._process_comments(comment_ids, "update", batch_size)

    @with_logging(logger)
    def search_comments(
        self, query: str, subreddit: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for comments containing specific text.

        Args:
            query: Text to search for
            subreddit: Optional subreddit to limit search to
            limit: Maximum number of results to return

        Returns:
            List of comment dictionaries

        Raises:
            AuthenticationRequiredError: If not authenticated
        """
        if not self.authenticated:
            raise AuthenticationRequiredError("Authentication required for comment search")

        logger.debug_with_context(f"Searching for '{query}' in {subreddit or 'all'}")

        try:
            comments = []
            search_params = {"q": query, "limit": limit, "type": "comment"}
            if subreddit:
                results = self.reddit.subreddit(subreddit).search(**search_params)
            else:
                results = self.reddit.subreddit("all").search(**search_params)

            for result in results:
                if isinstance(result, praw.models.Comment):
                    comments.append(
                        {
                            "text": result.body.rstrip(),
                            "upvotes": result.ups,
                            "downvotes": result.downs,
                            "permalink": result.permalink,
                            "id": result.id,
                        }
                    )
                if len(comments) >= limit:
                    break

            return comments
        except Exception as e:
            handle_exception(e, f"Failed to search for '{query}'", debug=True)
            return []

    @with_logging(logger)
    def parse_user(self, username, limit=100, sort="new", time_filter="all", **kwargs):
        """Parses a listing and extracts the comments from it.

        :param username: a user
        :param limit: maximum number of comments to return (None for unlimited)
        :param sort: Sort method ('hot', 'new', 'controversial', 'top')
        :param time_filter: Time filter for 'top' ('all', 'day', 'hour', 'month', 'week', 'year')
        :return: a list of comments from a user
        :raises: prawcore.exceptions.NotFound if user doesn't exist
        :raises: prawcore.exceptions.Forbidden if user is private/banned
        """
        if self.reddit is None:
            logger.error("Reddit client initialization failed - cannot fetch comments")
            return []

        mode = "authenticated" if self.authenticated else "read-only"
        logger.info(f"Fetching comments for user '{username}' in {mode} mode")
        logger.debug(f"Using sort method: {sort}")
        try:
            redditor = self.reddit.redditor(username)
            comments = []

            # Get the appropriate comment listing based on sort
            if sort == "hot":
                comment_listing = redditor.comments.hot(limit=limit)
            elif sort == "new":
                comment_listing = redditor.comments.new(limit=limit)
            elif sort == "controversial":
                comment_listing = redditor.comments.controversial(
                    limit=limit, time_filter=time_filter
                )
            elif sort == "top":
                comment_listing = redditor.comments.top(limit=limit, time_filter=time_filter)
            else:
                comment_listing = redditor.comments.new(limit=limit)  # default to new

            for comment in comment_listing:
                comment_data = {
                    "text": comment.body.rstrip(),
                    "upvotes": comment.ups,
                    "downvotes": comment.downs,
                    "permalink": comment.permalink,
                    "id": comment.id,
                }

                # If text matching is enabled, only include matching comments
                if "text_match" in kwargs:
                    logger.debug_with_context(
                        f"Text match enabled: searching for '{kwargs['text_match']}' in comment {comment_data['id']}"
                    )
                    if kwargs["text_match"].lower() in comment_data["text"].lower():
                        logger.debug_with_context(f"Match found in comment {comment_data['id']}")
                        comments.append(comment_data)
                    else:
                        logger.debug_with_context(f"No match found in comment {comment_data['id']}")
                else:
                    logger.debug_with_context(
                        f"No text match filter, including comment {comment_data['id']}"
                    )
                    comments.append(comment_data)

                if len(comments) >= limit:
                    break

            return comments
        except Exception as e:
            handle_exception(e, f"Failed to fetch comments for user '{username}'", debug=True)
            return []
