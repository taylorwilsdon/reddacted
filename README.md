# Reddit Sentiment Analyzer

The Reddit Sentiment Analyzer uses [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
to build a score of various Reddit Objects. These objects include:

* Listings: `/r/<subreddit>/comments/<article>`
* Users: `/u/<username>`

The score is generated based on the sentiment of the comments. Note: It only analyzes the top comments.

## Installation ##

It is recommended that you first create a python virtual environment to not
overwrite pip dependancies in your system. See [virtualenvs](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

1. `pip install -r requirements.txt` # install requirements
2. `python -m nltk.downloader vader_lexicon` # installs required nltk packages
3. `sudo python setup.py install` # installs the Reddit Sentiment Analyzer

## Using the CLI ##

* `reddit-analyze listing <subreddit> <article>` Outputs the sentiment score of a particular Listing.

* `reddit-analyze user <name>` Outputs the sentiment score of a particular User.

Note: Passing `--output-file <your file>` after any of the above will generate a
file with the sentiment analysis.

## Testing ##

I have included a number of unit tests to validate the application. In order to
run the tests, simply perform the following:

1. `pip install -r test-requirements.txt` # installs needed test requirements
2. `nosetests <test directory>` # runs the tests

## Common Exceptions ##

### too many requests ###

This may happen often if you/others are constantly performing an analysis of reddit.
Simply try again in a few seconds.

### the page you requested does not exist ###

Simply a 404, which means that the provided username, or subreddit and article
combination do not point to a valid page.
