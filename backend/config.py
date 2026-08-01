"""
config.py (backend version)
Non-interactive — reads GEMINI_API_KEY / TAVILY_API_KEY from environment
variables only (set these in the Render dashboard). No input() prompts,
since Render runs with no terminal attached.
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROXY_BASE_URL = "https://saidazam-litellm-proxy.hf.space/v1"


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it in Render > Environment.")
    return key


def get_llms():
    api_key = get_gemini_key()
    llm_flash = ChatOpenAI(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-flash-lite")
    llm_lite = ChatOpenAI(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-flash-lite")
    return llm_flash, llm_lite


def get_embedder():
    api_key = get_gemini_key()
    return OpenAIEmbeddings(base_url=PROXY_BASE_URL, api_key=api_key, model="gemini-embedding")
