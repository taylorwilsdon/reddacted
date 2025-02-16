# reddacted: AI-Powered Reddit Privacy Suite

[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25_Client--Side_Processing-success)](https://github.com/taylorwilsdon)
[![AI Analysis](https://img.shields.io/badge/AI-PII_Detection-blueviolet)](https://example.com)

## Table of Contents
- [What is reddacted?](#what-is-reddacted)
- [Installation](#installation)
- [Using the CLI](#using-the-cli)
- [FAQ](#faq)
- [Support & Community](#support--community)
- [Troubleshooting](#troubleshooting)
- [Authentication](#authentication)
- [Development](#development)
- [Testing](#testing)
- [Common Exceptions](#common-exceptions)

**Next-generation digital footprint management with llm, sentiment analysis & pii detection insights into your true privacy** · *For aging engineers who care about their future political careers* 🏛️

## What is reddacted?
- Clean up your online footprint without blowing away everything, analyze the content of comments to identify anything that might be likely to reveal PII that you may not want correlated with your anonymous username and perform sentiment analysis on the content of those posts.
- Easy, lazy, self hosted - the way an aging former engineer with a career doing things right at the enterprise cale would clean up your dirty laundry.

🛡️ **PII Detection** - Find potential personal info leaks in comments using AI/Regex

🤫 **Sentiment Analysis** - Understand the emotional tone of your Reddit history

🔒 **Zero-Trust Architecture** - Client-side execution only, no data leaves your machine unless you choose to use a hosted API. Fully compatible with all OpenAI compatible endpoints.

⚡ **Self-Host Ready** - Local: You can use any model via Ollama, llama.cpp, vLLM or other platform capable of exposing an OpenAI-compatible endpoint. LiteLLM works just dandy. • Cloud: OpenAI-compatible endpoints

📊 **Smart Cleanup** - Preserve valuable contributions while removing risky content
### 🔐 Can I trust this with my data?
```bash
# you don't have to - read the code for yourself, only reddit is called
reddacted user yourusername \
  --local-llm "localhost:11434"
```
- Client-side execution only, no tracking or external calls
- Session-based authentication if you choose - it is optional unless you want to delete
- Keep your nonsense comments with lots of upvotes and good vibes without unintentionally doxing yourself someday off in the future when you run for mayor.

```
reddacted user taylorwilsdon --limit 3
```
https://github.com/user-attachments/assets/db088d58-2f53-4513-95cc-d4b70397ff82


## Installation

```bash
# Install globally
pip install .

# Or install in development mode (for contributors)
pip install -e .
```

That's it! No PATH configuration needed.

## Using the CLI

Install once with:
```bash
pip install .
```

Then run directly:
```bash
reddacted user <username> [--output-file analysis.txt] [--enable-auth]
reddacted listing <subreddit> <article> [--output-file results.csv]
```

https://github.com/user-attachments/assets/10934119-95f5-4f62-b8e0-ebb883fbd57b


## 💡 FAQ



## Support & Community
Join our subreddit: [r/reddacted](https://reddit.com/r/reddacted)

### ❓ How accurate is the PII detection, really?
Surprisingly good. Good enough that I run it against my own stuff in delete mode. It's basically a defense-in-depth approach combining these, and I'll probably add upvotes/downvotes into the logic at some point:
- **AI Detection**: Doesn't need a crazy smart model, don't waste your money on r1 or o1. Cheap & light models like gpt-4o-mini, gpt-3.5-turbo, qwen2.5:3b or 7b and Mistral are all plenty. Don't use something too dumb or it will be inconsistent, a 0.5b model will produce unreliable results. It works well with cheap models like qwen2.5:3b (potato can run this) and gpt-4o-mini, which is like 15 cents per million tokens
- **Pattern Matching**: 50+ regex rules for common PII formats does a first past sweep for the obvious stuff
- **Context Analysis**: Are you coming off as a dick? Perhaps that factors into your decision to clean up. Who could say, mine are all smiley faces.

**Q:** How does the AI handle false positives?
**A:** Adjust confidence threshold (default 0.7) per risk tolerance. You're building a repo from source off some random dude's github - don't run this and just delete a bunch of shit blindly, you're a smart person. Review your results, and if it is doing something crazy, please tell me.

**Q:** What LLMs are supported?
**A:** Local: any model via Ollama, vLLM or other platform capable of exposing an openai-compatible endpoint. • Cloud: OpenAI-compatible endpoints
**Q:** Is my data sent externally?
**A:** If you choose to use a hosted provider, yes - in cloud mode - local analysis stays fully private.

## Troubleshooting

If you get "command not found" after installation:
1. Check Python scripts directory is in your PATH:
```bash
# Typical Linux/Mac location
export PATH="$HOME/.local/bin:$PATH"

# Typical Windows location
set PATH=%APPDATA%\Python\Python311\Scripts;%PATH%
```
2. Verify installation location:
```bash
pip show reddacted
```

## Authentication

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

## Development

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

## Testing

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

## Common Exceptions

### too many requests

If you're unauthenticated, reddit has relatively low rate limits for it's API. Either authenticate against your account, or just wait a sec and  try again.

### the page you requested does not exist

Simply a 404, which means that the provided username does not point to a valid page.

> **Pro Tip**: Always review changes before executing deletions!
