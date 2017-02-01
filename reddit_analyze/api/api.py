import abc
import six


@six.add_metaclass(abc.ABCMeta)
class API(object):
    """Base API Interface

    The API is responsible for gathering data to perform a sentiment
    analysis on. Note: Currenty scraper(bs4) is the only class we use,
    but may consider using the reddit apis in the future.
    """

    @abc.abstractmethod
    def parse_listing(self, subreddit, article):
        """Parses a Listing Reddit Object."""
        pass

    @abc.abstractmethod
    def parse_user(self, username):
        """Parses a User Reddit Object."""
        pass
