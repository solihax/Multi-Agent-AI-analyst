"""
config.py
Loads API keys and sets up the two LLMs used across the project.

Colab note: input().strip() is used instead of getpass or .env because
.env files don't work reliably with the VS Code -> Colab remote kernel
setup, and getpass hangs in VS Code's Jupyter frontend.
"""

import os
from langchain_openai import ChatOpenAI

PROXY_BASE_URL = "https://saidazam-litellm-proxy.hf.space/v1"


def load_gemini_key():
    """Prompts for the Gemini/proxy key and stores it in the environment."""
    if not os.environ.get("GEMINI_API_KEY"):
        key = input("Enter your Gemini/proxy API key: ").strip()
        os.environ["GEMINI_API_KEY"] = key
    print("Gemini key loaded:", bool(os.environ.get("GEMINI_API_KEY")))
    return os.environ["GEMINI_API_KEY"]


def load_tavily_key():
    """Prompts for the Tavily key and stores it in the environment. Optional —
    web_agent degrades gracefully if this is never set."""
    if not os.environ.get("TAVILY_API_KEY"):
        key = input("Enter your Tavily API key (blank to skip web search): ").strip()
        if key:
            os.environ["TAVILY_API_KEY"] = key
    print("Tavily key loaded:", bool(os.environ.get("TAVILY_API_KEY")))
    return os.environ.get("TAVILY_API_KEY")


def get_llms():
    """Returns (llm_flash, llm_lite) — both point at gemini-flash-lite, the
    only chat model currently on the proxy (see mentor's model_list). Kept
    as two names since graph.py/agents.py use them for different roles
    (llm_flash = supervisor/critic, llm_lite = generate/data/code) — if a
    second, larger model gets added to the proxy later, only this function
    needs to change."""
    api_key = load_gemini_key()
    llm_flash = ChatOpenAI(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-flash-lite")
    llm_lite = ChatOpenAI(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-flash-lite")
    return llm_flash, llm_lite


def get_embedder():
    """Embeddings via the proxy (gemini-embedding), replacing the local
    SentenceTransformer. Returns a langchain_openai.OpenAIEmbeddings."""
    from langchain_openai import OpenAIEmbeddings
    api_key = load_gemini_key()
    return OpenAIEmbeddings(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-embedding")
