# 🛡️ reddacted

<div align="center">

### AI-Powered Reddit Privacy Suite

[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25_Client--Side_Processing-success)](https://github.com/taylorwilsdon)
[![AI Analysis](https://img.shields.io/badge/AI-PII_Detection-blueviolet)](https://github.com/taylorwilsdon/reddacted)
![GitHub License](https://img.shields.io/github/license/taylorwilsdon/reddacted)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/taylorwilsdon/reddacted)
![PyPI - Version](https://img.shields.io/pypi/v/reddacted)
![PyPI Downloads](https://img.shields.io/pypi/dm/reddacted?style=flat&logo=pypi&logoColor=white&label=Downloads&labelColor=005da7&color=blue)

<p><i>Local LLM powered, highly performant privacy analysis leveraging AI, sentiment analysis & PII detection<br>to provide insights into your true privacy with bulk remediation</i></p>

<p><i>For aging engineers who want to protect their future political careers</i> 🏛️</p>

</div>

<div align="center">
  <img width="800" alt="reddacted demo" src="https://github.com/user-attachments/assets/934113f1-4a38-4985-935c-b247688ccac8">
</div>

<div align="center">
  <video width="800" src="https://github.com/user-attachments/assets/ef96ac1a-3b3b-4fb6-a912-1328b6a0d83a"></video>

</div>

<div align="center">
  <img width="800" alt="reddacted results" src="https://github.com/user-attachments/assets/338fa47e-88da-4844-9728-ace17983cd83">
</div>

## ✨ Key Features

<div align="center">
<table>
  <tr>
    <td align="center"><h3>🛡️</h3></td>
    <td><b>PII Detection</b><br>Analyze the content of comments to identify anything that might reveal PII that you may not want correlated with your anonymous username</td>
  </tr>
  <tr>
    <td align="center"><h3>🤫</h3></td>
    <td><b>Sentiment Analysis</b><br>Understand the emotional tone of your Reddit history, combined with upvote/downvote counts & privacy risks to choose which posts to reddact</td>
  </tr>
  <tr>
    <td align="center"><h3>🔒</h3></td>
    <td><b>Zero-Trust Architecture</b><br>Client-side execution only, no data leaves your machine unless you choose to use a hosted API. Fully compatible with all OpenAI compatible endpoints</td>
  </tr>
  <tr>
    <td align="center"><h3>⚡</h3></td>
    <td><b>Self-Host Ready</b><br>Use any model via Ollama, llama.cpp, vLLM or other platform capable of exposing an OpenAI-compatible endpoint. LiteLLM works just dandy.</td>
  </tr>
  <tr>
    <td align="center"><h3>📊</h3></td>
    <td><b>Smart Cleanup</b><br>Preserve valuable contributions while removing risky content - clean up your online footprint without blowing away everything</td>
  </tr>
</table>
</div>

## 🔐 Can I trust this with my data?

<div align="center">
<p><i>You don't have to - read the code for yourself, only reddit is called</i></p>
</div>

```bash
reddacted user yourusername --local-llm "http://localhost:11434"
```

- ✅ Client-side execution only, no tracking or external calls
- ✅ Session-based authentication if you choose - it is optional unless you want to delete
- ✅ Keep your nonsense comments with lots of upvotes and good vibes without unintentionally doxing yourself

```bash
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

```bash
# Most basic possible quick start - this will walk you through selecting your LLM in the command line
reddacted user spez

# Analyze a user's recent comments with local LLM specified
reddacted user spez \
  --limit 5 \
  --local-llm "http://localhost:11434" \
  --model "qwen2.5:3b" \
  --sort new

# Analyze controversial comments with OpenAI
export OPENAI_API_KEY="your-api-key"
reddacted user spez \
  --sort controversial \
  --time month \
  --model "gpt-4" \
  --limit 10 \
  --pii-only

# Analyze a specific subreddit post with PII filter disabled
reddacted listing r/privacy abc123 \
  --local-llm "http://localhost:11434" \
  --model "qwen2.5:3b" \
  --disable-pii \
  --sort new

# Search for specific content (requires auth)
reddacted user spez \
  --enable-auth \
  --text-match "python" \
  --skip-text "deleted" \
  --sort top \
  --time all

# Bulk comment management
reddacted delete abc123,def456 --batch-size 5  # Delete comments
reddacted update abc123,def456                 # Replace with r/reddacted
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

### LLM Configuration

| Argument | Description |
|----------|-------------|
| `--local-llm URL` | Local LLM endpoint (OpenAI compatible) |
| `--openai-key KEY` | OpenAI API key |
| `--openai-base URL` | Custom OpenAI API base URL |
| `--model NAME` | Model to use (default: gpt-4 for OpenAI) |

<div class="note">
<b>Note:</b> For cloud-based analysis using OpenAI, you can either use the <code>--openai-key</code> flag or set the environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```
</div>

## ❓ How accurate is the PII detection, really?

Surprisingly good. Good enough that I run it against my own stuff in delete mode. It's basically a defense-in-depth approach combining these methods:

<div class="detection-methods">
  <div class="method">
    <h3>📊 AI Detection</h3>
    <p>Doesn't need a crazy smart model, don't waste your money on r1 or o1.</p>
    <ul>
      <li>Cheap & light models like gpt-4o-mini, gpt-3.5-turbo, qwen2.5:3b or 7b and Mistral are all plenty</li>
      <li>Don't use something too dumb or it will be inconsistent, a 0.5b model will produce unreliable results</li>
      <li>Works well with cheap models like qwen2.5:3b (potato can run this) and gpt-4o-mini (~15¢ per million tokens)</li>
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

<style>
.note {
  background-color: #f8f9fa;
  border-left: 4px solid #1976d2;
  padding: 15px;
  margin: 20px 0;
  border-radius: 4px;
}

.detection-methods, .auth-steps, .exceptions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.method, .step, .exception {
  background-color: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.method h3, .step h3, .exception h3 {
  margin-top: 0;
  border-bottom: 1px solid #e0e0e0;
  padding-bottom: 10px;
}
</style>
