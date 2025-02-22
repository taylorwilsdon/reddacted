# reddacted: AI-Powered Reddit Privacy Suite

[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25_Client--Side_Processing-success)](https://github.com/taylorwilsdon)
[![AI Analysis](https://img.shields.io/badge/AI-PII_Detection-blueviolet)](https://github.com/taylorwilsdon/reddacted)
![GitHub License](https://img.shields.io/github/license/taylorwilsdon/reddacted)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/taylorwilsdon/reddacted)
![PyPI - Version](https://img.shields.io/pypi/v/reddacted)

<img width="450" alt="image" src="https://github.com/user-attachments/assets/338fa47e-88da-4844-9728-ace17983cd83" />

## What is reddacted?
**Local LLM powererd, highly performant privacy analysis leveraging AI, sentiment analysis & PII detection to provide insights into your true privacy with bulk remediation** 

· *For aging engineers who want to protect their future political careers* 🏛️

🛡️ **PII Detection** - Analyze the content of comments to identify anything that might be likely to reveal PII that you may not want correlated with your anonymous username and perform sentiment analysis on the content of those posts

🤫 **Sentiment Analysis** - Understand the emotional tone of your Reddit history, combined with upvote/downvote counts & privacy risks you can choose which posts to reddact based on a complete picture of their public perception

🔒 **Zero-Trust Architecture** - Client-side execution only, no data leaves your machine unless you choose to use a hosted API. Fully compatible with all OpenAI compatible endpoints

⚡ **Self-Host Ready** - Easy, lazy, completely local: You can use any model via Ollama, llama.cpp, vLLM or other platform capable of exposing an OpenAI-compatible endpoint. LiteLLM works just dandy. • Cloud: OpenAI-compatible endpoints

📊 **Smart Cleanup** - Preserve valuable contributions while removing risky content - clean up your online footprint without blowing away everything

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
# Install from public PyPi
pip install reddacted

# Install globally
pip install .

# Or install from source with all dependencies
pip install -r requirements.txt
pip install .
```

## Quick Start

```bash
# Analyze a user's comments
reddacted user spez --limit 5

# Analyze a specific post
reddacted listing r/privacy abc123 --limit 10
```

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

Before running any commands that require authentication, you'll need to set up your Reddit API credentials. Here's how:

1. **Create a Reddit Account**: If you don't have one, sign up at [https://www.reddit.com/account/register/](https://www.reddit.com/account/register/)

2. **Create a Reddit App**:
   - Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
   - Click "are you a developer? create an app..." at the bottom
   - Choose "script" as the application type
   - Set "reddacted" as both the name and description
   - Use "http://localhost:8080" as the redirect URI
   - Click "create app"

3. **Get Your Credentials**:
   - After creating the app, note down:
     - Client ID: The string under "personal use script"
     - Client Secret: The string labeled "secret"

4. **Set Environment Variables**:
```bash
$ export REDDIT_USERNAME=your-reddit-username
$ export REDDIT_PASSWORD=your-reddit-password
$ export REDDIT_CLIENT_ID=your-client-id
$ export REDDIT_CLIENT_SECRET=your-client-secret
```

Now when running the CLI with `--enable-auth`, all requests will be properly authenticated. These credentials are also automatically used if all environment variables are present, even without the `--enable-auth` flag.

## Advanced Usage

### Text Filtering

You can filter comments using these arguments:

- `--text-match "search phrase"` - Only analyze comments containing specific text (requires authentication)
- `--skip-text "skip phrase"` - Skip comments containing specific text pattern

For example:
```bash
# Only analyze comments containing "python"
reddacted user spez --text-match "python"

# Skip comments containing "deleted"
reddacted user spez --skip-text "deleted"

# Combine both filters
reddacted user spez --text-match "python" --skip-text "deleted"
```

## Development

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install in development mode with test dependencies:
```bash
pip install -e ".[dev]"
```

That's it! The package handles all other dependencies automatically, including NLTK data.

## Testing

Run the test suite:
```bash
pytest tests
```

Want to contribute? Great! Feel free to:
- Open an Issue
- Submit a Pull Request

## Common Exceptions

### too many requests

If you're unauthenticated, reddit has relatively low rate limits for it's API. Either authenticate against your account, or just wait a sec and  try again.

### the page you requested does not exist

Simply a 404, which means that the provided username does not point to a valid page.

> **Pro Tip**: Always review changes before executing deletions!
