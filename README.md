# reddact: AI-Powered Reddit Privacy Suite

[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25_Client--Side_Processing-success)](https://example.com)
[![AI Analysis](https://img.shields.io/badge/AI-PII_Detection-blueviolet)](https://example.com)

**Next-generation anonymous content management with neural privacy protection**

## What is reddact?
- Clean up your online footprint without blowing away everyything, analyze the content of comments to identify anything that might be likely to reveal PII that you may not want correlated with your anonymous username and perform sentiment analysis on the content of those posts. 
- Easy, lazy, self hosted - the way an aging former engineer with a career doing things right at the enterprise cale would clean up your dirty laundry. 

✅ **Zero-Trust Architecture**  
- Client-side execution only
- No tracking or external calls
- Session-based authentication
- Keep your nonsense comments without unintentionally doxing yourself someday off in the future when you run for mayor. 


- **Users**:  Get the sentiment based on the most recent comments submitted

## Installation ##

```bash
$ pip install reddact
```

## Using the CLI ##

The CLI allows the following commands:

- `reddact user <name>` Outputs the sentiment score of a particular User.

### Additional Options ###

#### Reporting ####

Passing `--output-file file_name.txt` after the user command will generate a file with detailed sentiment analysis results.

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

If you're unauthenticated, reddit has relatively low rate limits for it's API. Either authenticate against your account, or just wait a sec and  try again.

### the page you requested does not exist ###

Simply a 404, which means that the provided username does not point to a valid page.
