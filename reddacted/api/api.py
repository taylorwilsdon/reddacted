import abc
import six


@six.add_metaclass(abc.ABCMeta)
class API(object):
    """Base API Interface

    The API is responsible for gathering data to perform a sentiment
    analysis on.
    """

    @abc.abstractmethod
    def parse_listing(self, subreddit, article, limit=100, **kwargs):
        """Parses a Listing Reddit Object.

        Args:
            subreddit: Subreddit to parse
            article: Article ID to parse
            limit: Maximum number of comments to return (None for unlimited)
        """
        pass

    @abc.abstractmethod
    def parse_user(self, username, limit=100, **kwargs):
        """Parses a User Reddit Object.

        Args:
            username: Username to parse
            limit: Maximum number of comments to return (None for unlimited)
        """
        pass
