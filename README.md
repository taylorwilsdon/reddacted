# Reddit Sentiment Analyzer 😁😐😕

The Reddit Sentiment Analyzer is a [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) CLI tool for [Reddit](https://www.reddit.com/). Reddit Sentiment gathers comments and builds a sentiment score for the following reddit abstractions:

- **Listings**: Get the sentiment based on top comments in an article
- **Users**:  Get the sentiment based on the most recent comments submitted

## Installation ##

```bash
$ pip install reddit-sentiment
```

## Using the CLI ##

The CLI allows the following commands:

- `reddit-sentiment listing <subreddit> <article>` Outputs the sentiment score of a particular Listing.
- `reddit-sentiment user <name>` Outputs the sentiment score of a particular User.

### Additional Options ###

#### Reporting ####

Passing `--output-file file_name.txt` after any of the above will generate a file with detailed sentiment analysis results.

#### Authentication ####

Passing `--enable-auth True` will allow the reddit request to be authenticated, which allows you to grab more/cleaner data. It will check if you have valid environment variables configured and use them to authenticate.

## Authentication ##

Before running an commands, in order to ensure that we are able to use the reddit API consecutively, we should authenticate with reddit. In order to do this the following is required:

- **Reddit Account**: You can sign up at [https://www.reddit.com/account/register/](https://www.reddit.com/account/register/)
- **Reddit App**: Click on the **are you a developer? create an app...** button at the bottom of [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- **Reddit API Access**: You can request access at [https://www.reddit.com/wiki/api/](https://www.reddit.com/wiki/api/)

Once the above is complete, we should set the following environment variables:

```bash
$ export REDDIT_USERNAME=your-username
$ export REDDIT_PASSWORD=your-password
$ export REDDIT_CLIENT_ID=your-client-id
$ export REDDIT_CLIENT_SECRET=your-client-secret

# verify env variables have been set
$ printenv
```

Now when running the CLI, all requests will be authenticated.

## Development ##

It is recommended that you first create a python virtual environment to not overwrite pip dependencies in your system. See [virtualenvs](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

1. Clone this repository

2. Change directory to application path

3. Install application requirements

```bash
$ pip install -r requirements.txt
```

4. Install required **nltk** packages

```bash
$ python -m nltk.downloader vader_lexicon
```

5. Make changes to the code

6. Install the application from source code

```bash
$ sudo python setup.py install
```

Now you can go ahead and test the new features you have implemented! Contributions welcome, feel free to contribute by:

- Opening an Issue
- Creating a PR with additions/fixes

## Testing ##

I have included a number of unit tests to validate the application. In order to run the tests, simply perform the following:

1. Install pytest

```bash
$ pip install pytest
```

2. Clone this repository

3. Change directory to application path

4. Install application requirements

```bash
$ pip install -r requirements.txt
```

5. Install required **nltk** packages

```bash
$ python -m nltk.downloader vader_lexicon
```

6. Install application test requirements

```bash
$ pip install -r test-requirements.txt
```

7. Run Unit tests

```bash
$ pytest tests
```

## Common Exceptions ##

### too many requests ###

This may happen often as reddit performs rate-limiting to it's API. This can be solved by either authenticating or simply try again in a few seconds.

### the page you requested does not exist ###

Simply a 404, which means that the provided username, or subreddit and article combination do not point to a valid page.