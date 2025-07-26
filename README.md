# 🛡️ reddacted

<div align="center">

### AI-Powered Reddit Privacy Suite

[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25_Client--Side_Processing-success)](https://github.com/taylorwilsdon)
[![AI Analysis](https://img.shields.io/badge/AI-PII_Detection-blueviolet)](https://github.com/taylorwilsdon/reddacted)
![GitHub License](https://img.shields.io/github/license/taylorwilsdon/reddacted)
![PyPI - Version](https://img.shields.io/pypi/v/reddacted)
[![PyPI Downloads](https://static.pepy.tech/badge/reddacted)](https://pepy.tech/projects/reddacted)

<p><i>Local LLM powered, highly performant privacy analysis leveraging AI, sentiment analysis & PII detection<br>to provide insights into your true privacy with bulk remediation</i></p>

<p><i>For aging engineers who want to protect their future political careers</i> 🏛️</p>

</div>

<div align="center">
  <img width="800" alt="reddacted demo" src="https://github.com/user-attachments/assets/934113f1-4a38-4985-935c-b247688ccac8">
</div>

<div align="center">
  <video width="800" src="https://github.com/user-attachments/assets/ef96ac1a-3b3b-4fb6-a912-1328b6a0d83a"></video>

</div>

## ✨ Key Features

<table>
  <tr>
    <td align="center">🛡️<br/><b>PII Detection</b></td>
    <td>Analyze the content of comments to identify anything that might reveal PII that you may not want correlated with your anonymous username</td>
  </tr>
  <tr>
    <td align="center">🤫<br/><b>Sentiment Analysis</b></td>
    <td>Understand the emotional tone of your Reddit history, combined with upvote/downvote counts & privacy risks to choose which posts to reddact</td>
  </tr>
  <tr>
    <td align="center">🔒<br/><b>Zero-Trust Architecture</b></td>
    <td>Client-side execution only, no data leaves your machine unless you choose to use a hosted API. Fully compatible with all OpenAI compatible endpoints</td>
  </tr>
  <tr>
    <td align="center">⚡<br/><b>Self-Host Ready</b></td>
    <td>Use any model via Ollama, llama.cpp, vLLM or other platform capable of exposing an OpenAI-compatible endpoint. LiteLLM works just dandy.</td>
  </tr>
  <tr>
    <td align="center">📊<br/><b>Smart Cleanup</b></td>
    <td>Preserve valuable contributions while removing risky content - clean up your online footprint without blowing away everything</td>
  </tr>
</table>

## 🔐 Can I trust this with my data?

<div align="center">
<p><i>You don't have to - read the code for yourself, only reddit is called</i></p>
</div>

```bash
# Run with local LLM - you'll be guided through configuration
reddacted user yourusername
```

- ✅ Client-side execution only, no tracking or external calls
- ✅ Session-based authentication if you choose - it is optional unless you want to delete
- ✅ Keep your nonsense comments with lots of upvotes and good vibes without unintentionally doxing yourself
- ✅ All configuration stored locally in `config.json`

```bash
# Quick analysis with custom limit
reddacted user taylorwilsdon --limit 3
```

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Can I trust this with my data?](#-can-i-trust-this-with-my-data)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Available Commands](#available-commands)
  - [Common Arguments](#common-arguments)
  - [LLM Configuration](#llm-configuration)
- [How accurate is the PII detection?](#-how-accurate-is-the-pii-detection-really)
- [FAQ](#-faq)
- [Troubleshooting](#-troubleshooting)
- [Authentication](#-authentication)
- [Advanced Usage](#-advanced-usage)
- [Development](#-development)
- [Testing](#-testing)
- [Common Exceptions](#-common-exceptions)
- [Support & Community](#-support--community)

## 📥 Installation

```bash
# Install from brew (recommended)
brew install taylorwilsdon/tap/reddacted

# Install from PyPI (recommended)
pip install reddacted

# Or install from source
git clone https://github.com/taylorwilsdon/reddacted.git
cd reddacted
pip install -e ".[dev]"  # Installs with development dependencies
```

## 🚀 Usage

reddacted now features a guided configuration flow that makes setup easy. Simply run any command and you'll be prompted to configure your settings through an interactive interface:

```bash
# Most basic possible quick start - launches the guided configuration flow
reddacted user spez

# The guided flow will prompt you to:
# - Choose between OpenAI or local LLM
# - Enter your API key or local LLM URL
# - Select your model from available options
# - Configure authentication settings
# - Set analysis preferences (limit, sort, time filter, etc.)
# - Save your configuration for future use
```

### Configuration Options

The interactive configuration flow includes:

- **LLM Settings**: Choose between OpenAI API or local LLM endpoint (like Ollama)
- **Authentication**: Enable Reddit API authentication if needed
- **Analysis Options**: Set comment limits, sort order, time filters
- **Output Options**: Configure file output, PII filtering preferences
- **Advanced Settings**: Text matching patterns, batch sizes for bulk operations

Your configuration is automatically saved to `config.json` for reuse.

### Example Commands

Once configured, you can run commands like:

```bash
# Analyze a user's recent comments (uses saved config)
reddacted user spez

# Analyze a specific subreddit post
reddacted listing r/privacy abc123

# Bulk comment management
reddacted delete abc123,def456  # Delete comments
reddacted update abc123,def456  # Replace with standard redaction message
```

### Override Configuration

You can still override saved settings with command-line arguments:

```bash
# Override the saved limit
reddacted user spez --limit 50

# Use a different model temporarily
reddacted user spez --model "gpt-4-turbo"

# Enable authentication for this run only
reddacted user spez --enable-auth
```

### Available Commands

| Command | Description |
|---------|-------------|
| `user` | Analyze a user's comment history |
| `listing` | Analyze a specific post and its comments |
| `delete` | Delete comments by their IDs |
| `update` | Replace comment content with r/reddacted |

### Common Arguments

| Argument | Description |
|----------|-------------|
| `--limit N` | Maximum comments to analyze (default: 100, 0 for unlimited) |
| `--sort` | Sort method: hot, new, controversial, top (default: new) |
| `--time` | Time filter: all, day, hour, month, week, year (default: all) |
| `--output-file` | Save detailed analysis to a file |
| `--enable-auth` | Enable Reddit API authentication |
| `--disable-pii` | Skip PII detection |
| `--pii-only` | Show only comments containing PII |
| `--text-match` | Search for comments containing specific text |
| `--skip-text` | Skip comments containing specific text pattern |
| `--batch-size` | Comments per batch for delete/update (default: 10) |
| `--use-random-string` | Use random UUID instead of standard message when updating comments |

### LLM Configuration

The guided configuration flow will help you set up your LLM preferences. You can choose between:

1. **Local LLM** (Ollama, vLLM, etc.):
   - Default endpoint: `http://localhost:11434`
   - Automatically fetches available models
   - No API key required

2. **OpenAI API**:
   - Enter your OpenAI API key
   - Select from available OpenAI models
   - Supports custom API base URLs

Configuration values are saved to `config.json` and can be overridden with command-line flags:

| Flag | Description |
|------|-------------|
| `--local-llm URL` | Override local LLM endpoint |
| `--openai-key KEY` | Override OpenAI API key |
| `--model NAME` | Override model selection |

<div class="note">
<b>Note:</b> Environment variables are also supported:

```bash
export OPENAI_API_KEY="your-api-key"
export REDDIT_USERNAME="your-username"
export REDDIT_PASSWORD="your-password"
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
```

These will be automatically loaded if present.
</div>

## ❓ How accurate is the PII detection, really?

Surprisingly good. Good enough that I run it against my own stuff in delete mode. It's basically a defense-in-depth approach combining these methods:

<div class="detection-methods">
  <div class="method">
    <h3>📊 AI Detection</h3>
    <p>Doesn't need a crazy smart model, don't waste your money on r1 or o1.</p>
    <ul>
      <li>Cheap & light models like qwen3:8b, gpt-4.1-nano, qwen2.5:7b, Mistral SSmall or gemma3:14b are all plenty</li>
      <li>Don't use something too dumb or it will be inconsistent, a 0.5b model will produce unreliable results</li>
      <li>Works fine with cheap models like qwen2.5:3b (potato can run this) and gpt-4o-mini (~15¢ per million tokens), but gets better with 7b and up</li>
    </ul>
  </div>
  
  <div class="method">
    <h3>🔍 Pattern Matching</h3>
    <p>50+ regex rules for common PII formats does a first past sweep for the obvious stuff</p>
  </div>
  
  <div class="method">
    <h3>🧠 Context Analysis</h3>
    <p>Are you coming off as a dick? Perhaps that factors into your decision to clean up. Who could say, mine are all smiley faces.</p>
  </div>
</div>

## 💡 FAQ

<details>
<summary><b>Q: How does the AI handle false positives?</b></summary>
<p>Adjust confidence threshold (default 0.7) per risk tolerance. You're building a repo from source off some random dude's github - don't run this and just delete a bunch of stuff blindly, you're a smart person. Review your results, and if it is doing something crazy, please tell me.</p>
</details>

<details>
<summary><b>Q: What LLMs are supported?</b></summary>
<p><b>Local:</b> any model via Ollama, vLLM or other platform capable of exposing an openai-compatible endpoint.<br>
<b>Cloud:</b> OpenAI-compatible endpoints</p>
</details>

<details>
<summary><b>Q: Is my data sent externally?</b></summary>
<p>If you choose to use a hosted provider, yes - in cloud mode - local analysis stays fully private.</p>
</details>

## 🔧 Troubleshooting

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

## 🔑 Authentication

Before running any commands that require authentication, you'll need to set up your Reddit API credentials:

<div class="auth-steps">
  <div class="step">
    <h3>Step 1: Create a Reddit Account</h3>
    <p>If you don't have one, sign up at <a href="https://www.reddit.com/account/register/">https://www.reddit.com/account/register/</a></p>
  </div>
  
  <div class="step">
    <h3>Step 2: Create a Reddit App</h3>
    <ul>
      <li>Go to <a href="https://www.reddit.com/prefs/apps">https://www.reddit.com/prefs/apps</a></li>
      <li>Click "are you a developer? create an app..." at the bottom</li>
      <li>Choose "script" as the application type</li>
      <li>Set "reddacted" as both the name and description</li>
      <li>Use "http://localhost:8080" as the redirect URI</li>
      <li>Click "create app"</li>
    </ul>
  </div>
  
  <div class="step">
    <h3>Step 3: Get Your Credentials</h3>
    <p>After creating the app, note down:</p>
    <ul>
      <li>Client ID: The string under "personal use script"</li>
      <li>Client Secret: The string labeled "secret"</li>
    </ul>
  </div>
  
  <div class="step">
    <h3>Step 4: Set Environment Variables</h3>
    
```bash
export REDDIT_USERNAME=your-reddit-username
export REDDIT_PASSWORD=your-reddit-password
export REDDIT_CLIENT_ID=your-client-id
export REDDIT_CLIENT_SECRET=your-client-secret
```
  </div>
</div>

These credentials are also automatically used if all environment variables are present, even without the `--enable-auth` flag.

## 🧙‍♂️ Advanced Usage

### Text Filtering

You can filter comments using these arguments:

| Argument | Description |
|----------|-------------|
| `--text-match "search phrase"` | Only analyze comments containing specific text (requires authentication) |
| `--skip-text "skip phrase"` | Skip comments containing specific text pattern |

For example:

```bash
# Only analyze comments containing "python"
reddacted user spez --text-match "python"

# Skip comments containing "deleted"
reddacted user spez --skip-text "deleted"

# Combine both filters
reddacted user spez --text-match "python" --skip-text "deleted"
```

## 👨‍💻 Development

This project uses [UV](https://github.com/astral-sh/uv) for building and publishing. Here's how to set up your development environment:

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install UV:

```bash
pip install uv
```

3. Install in development mode with test dependencies:

```bash
pip install -e ".[dev]"
```

4. Build the package:

```bash
uv build --sdist --wheel
```

5. Create a new release:

```bash
./release.sh
```

The release script will:
- Build the package with UV
- Create and push a git tag
- Create a GitHub release
- Update the Homebrew formula
- Publish to PyPI (optional)

That's it! The package handles all other dependencies automatically, including NLTK data.

## 🧪 Testing

Run the test suite:

```bash
pytest tests
```

Want to contribute? Great! Feel free to:
- Open an Issue
- Submit a Pull Request

## ⚠️ Common Exceptions

<div class="exceptions">
  <div class="exception">
    <h3>too many requests</h3>
    <p>If you're unauthenticated, reddit has relatively low rate limits for it's API. Either authenticate against your account, or just wait a sec and try again.</p>
  </div>
  
  <div class="exception">
    <h3>the page you requested does not exist</h3>
    <p>Simply a 404, which means that the provided username does not point to a valid page.</p>
  </div>
</div>

> **Pro Tip**: Always review changes before executing deletions!

## 🌐 Support & Community

<div align="center">
  <p>Join our subreddit: <a href="https://reddit.com/r/reddacted">r/reddacted</a></p>
</div>
