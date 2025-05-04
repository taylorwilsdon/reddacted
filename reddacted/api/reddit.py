from types import BuiltinMethodType
import time
import os
from typing import List, Dict, Any, Optional # Added Optional
from typing import List, Dict, Any
import uuid  # Added for random string generation
import praw
from reddacted.api import api
from reddacted.utils.log_handler import get_logger, with_logging
from reddacted.utils.log_handler import handle_exception

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

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_random_string=False):
        """Initialize Reddit API client. Prioritizes credentials from config,
        then environment variables, falling back to read-only mode.

        Args:
            config: Optional dictionary containing configuration values (including credentials).
            use_random_string: Whether to use random UUIDs instead of standard message when updating comments.
        """
        self.authenticated = False
        self.reddit = None
        self.use_random_string = use_random_string
        config = config or {} # Ensure config is a dict

        logger.debug_with_context(f"Initializing Reddit client. Config provided: {bool(config)}, Use random string: {use_random_string}")

        # --- Try credentials from config first ---
        username = config.get("reddit_username")
        password = config.get("reddit_password")
        client_id = config.get("reddit_client_id")
        client_secret = config.get("reddit_client_secret")

        # Check if enable_auth is explicitly True in config, otherwise don't use config creds
        auth_enabled_in_config = config.get("enable_auth", False)

        if auth_enabled_in_config and all([username, password, client_id, client_secret]):
            logger.info_with_context("Attempting authentication using credentials from configuration (auth enabled).")
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    password=password,
                    user_agent=f"reddacted u/{username}",
                    username=username,
                    check_for_async=False,
                )
                logger.info_with_context("Successfully authenticated with Reddit API using configuration.")
                logger.debug_with_context(f"Granted scopes (config auth): {self.reddit.auth.scopes()}") # Log scopes
                self.authenticated = True
                return # Exit if successful
            except Exception as e:
                logger.warning_with_context(f"Authentication with config credentials failed: {e}. Falling back...")
                # Continue to try environment variables
        elif not auth_enabled_in_config and any([username, password, client_id, client_secret]):
             logger.info_with_context("Credentials found in config, but 'enable_auth' is false. Skipping config auth attempt.")


        # --- Fallback to environment variables ---
        logger.debug_with_context("Checking environment variables for Reddit credentials.")
        env_username = os.environ.get("REDDIT_USERNAME")
        env_password = os.environ.get("REDDIT_PASSWORD")
        env_client_id = os.environ.get("REDDIT_CLIENT_ID")
        env_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

        if all([env_username, env_password, env_client_id, env_client_secret]):
            # Only use env vars if config auth wasn't explicitly enabled and successful
            if not (auth_enabled_in_config and self.authenticated):
                logger.info_with_context("Attempting authentication using credentials from environment variables.")
                try:
                    self.reddit = praw.Reddit(
                        client_id=env_client_id,
                        client_secret=env_client_secret,
                        password=env_password,
                        user_agent=f"reddacted u/{env_username}",
                        username=env_username,
                        check_for_async=False,
                    )
                    logger.info_with_context("Successfully authenticated with Reddit API using environment variables.")
                    logger.debug_with_context(f"Granted scopes (env auth): {self.reddit.auth.scopes()}") # Log scopes
                    self.authenticated = True
                    return # Exit if successful
                except Exception as e:
                    logger.warning_with_context(f"Authentication with environment variable credentials failed: {e}. Falling back...")
                    # Continue to try read-only
            else:
                 logger.debug_with_context("Skipping environment variable auth attempt as config auth was enabled and successful.")

        # --- Fallback to read-only mode ---
        if not self.authenticated: # Only attempt read-only if not already authenticated
            missing_sources = []
            if not auth_enabled_in_config or not all([username, password, client_id, client_secret]):
                missing_sources.append("configuration")
            if not all([env_username, env_password, env_client_id, env_client_secret]):
                missing_sources.append("environment variables")

            logger.warning_with_context(
                f"Reddit API authentication credentials not found or incomplete in { ' or '.join(missing_sources) }. "
                "Falling back to read-only mode. Some features like comment deletion/update will be unavailable."
            )
            try:
                # Use client_id/secret from config OR env vars if available for read-only
                read_only_client_id = config.get("reddit_client_id") or env_client_id
                read_only_client_secret = config.get("reddit_client_secret") or env_client_secret

                if read_only_client_id and read_only_client_secret:
                     logger.debug_with_context("Attempting read-only initialization with client_id/secret.")
                     self.reddit = praw.Reddit(
                         client_id=read_only_client_id,
                         client_secret=read_only_client_secret,
                         user_agent="reddacted:read_only_client_v3" # Updated user agent slightly
                     )
                     logger.info_with_context("Successfully initialized read-only Reddit client (with client ID/secret).")
                elif read_only_client_id:
                     logger.debug_with_context("Attempting read-only initialization with client_id only.")
                     self.reddit = praw.Reddit(
                         client_id=read_only_client_id,
                         user_agent="reddacted:read_only_client_v3"
                     )
                     logger.info_with_context("Successfully initialized read-only Reddit client (with client ID only).")
                else:
                     # PRAW requires at least client_id for read-only access usually.
                     # If neither config nor env vars provide it, initialization will likely fail here.
                     logger.error_with_context("Cannot initialize read-only Reddit client: Missing 'client_id' in both config and environment variables.")
                     # Optionally, raise an error or let the PRAW error propagate
                     # raise ValueError("Missing required client_id for Reddit API access.")
                     # For now, let PRAW handle the potential error if it occurs without client_id
                     self.reddit = praw.Reddit(user_agent="reddacted:read_only_client_v3") # This line might fail
                     logger.info_with_context("Attempted read-only Reddit client initialization (without client ID/secret - may fail).")


            except Exception as e:
                # Log the specific PRAW error if initialization fails
                logger.error_with_context(f"Failed to initialize read-only client: {str(e)}")
                # self.reddit remains None

    @with_logging(logger)
    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a listing and extracts the comments from it.

        :param subreddit: a subreddit
        :param article: an article associated with the subreddit
        :param limit: maximum number of comments to return (None for unlimited)
        :return: a list of comments from an article
        """
        if self.reddit is None:
            logger.error_with_context("Reddit client initialization failed - cannot fetch comments")
            return []

        mode = "authenticated" if self.authenticated else "read-only"
        logger.info_with_context(f"Fetching comments for article '{article}' in {mode} mode")
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
        self,
        comment_ids: list[str],
        action: str,
        batch_size: int = 10,
        update_content: str = None,  # Added parameter for update text
    ) -> dict[str, any]:
        """
        Process comments in batches with rate limiting.

        :param comment_ids: List of comment IDs to process.
        :param action: Action to perform ('delete' or 'update').
        :param batch_size: Number of comments to process per batch.
        :param update_content: The text to use when updating comments (only used if action='update').
        :return: Dict with results and statistics.
        """
        logger.debug_with_context("Starting _process_comments")
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
                            logger.debug_with_context(f"Deleting comment ID {comment.id}") # Use comment_id for clarity
                            comment.delete()
                            results["successful_ids"].append(comment_id)
                            results["success"] += 1
                        elif action == "update":
                            logger.debug_with_context(f"Updating comment ID {comment.id} with content: '{update_content[:50]}...'") # Use comment_id
                            if update_content is None:
                                # Should not happen if called via update_comments, but provides a fallback.
                                logger.warning_with_context(f"No update_content provided for comment {comment_id}, skipping edit.")
                            else:
                                comment.edit(update_content)
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

    def update_comments(
        self,
        comment_ids: list[str],
        batch_size: int = 10,
        use_random_string: bool = None,  # Can be explicitly provided or use instance default
    ) -> dict[str, any]:
        """
        Update comments in batches with rate limiting.

        Replaces content either with a standard redaction message or a random UUID.

        :param comment_ids: List of comment IDs to update.
        :param batch_size: Number of comments to process per batch.
        :param use_random_string: If True, replace content with a random UUID; otherwise, use the standard message.
                                 If None, uses the value set during Reddit instance initialization.
        :return: Dict with results and statistics.
        """
        # Use instance default if not explicitly provided
        if use_random_string is None:
            use_random_string = self.use_random_string
            
        if use_random_string:
            content_to_write = str(uuid.uuid4())
            logger.info_with_context(f"Updating comments with random UUIDs. Example: {content_to_write}")
        else:
            content_to_write = "This comment has been reddacted to preserve online privacy - see r/reddacted for more info"
            logger.info_with_context("Updating comments with standard redaction message.")

        return self._process_comments(
            comment_ids, "update", batch_size, update_content=content_to_write
        )

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
            logger.error_with_context("Reddit client initialization failed - cannot fetch comments")
            return []

        mode = "authenticated" if self.authenticated else "read-only"
        logger.info_with_context(f"Fetching comments for user '{username}' in {mode} mode")
        logger.debug_with_context(f"Using sort method: {sort}")
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
