import abc
import six


@six.add_metaclass(abc.ABCMeta)
class API(object):
    """Base API Interface

    The API is responsible for gathering data to perform a sentiment
    analysis on.
    """

    @abc.abstractmethod
    def parse_listing(self, subreddit, article, **kwargs):
        """Parses a Listing Reddit Object."""
        pass

    @abc.abstractmethod
    def parse_user(self, username, **kwargs):
        """Parses a User Reddit Object."""
        pass
