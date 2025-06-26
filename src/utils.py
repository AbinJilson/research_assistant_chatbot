import asyncio
import logging
import os
import re
import stat
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Defer heavy/problematic imports
cv2 = None
np = None

def _import_image_dependencies():
    """Import heavy image libraries locally to avoid Streamlit watcher issues."""
    global cv2, np
    if cv2 is None:
        try:
            import cv2 as cv2_
            cv2 = cv2_
            import numpy as np_
            np = np_
        except ImportError as e:
            logger.warning(f"Image processing libraries not available: {e}")
            # Create dummy objects to avoid errors if cv2/np are not installed
            cv2 = type('DummyCV2', (), {'__getattr__': lambda s, n: (lambda *a, **kw: None)})()
            np = type('DummyNumPy', (), {'__getattr__': lambda s, n: (lambda *a, **kw: None)})()

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree. If the error is a PermissionError, it changes the
    file's permissions and retries the deletion. This is a common pattern for Windows.
    """
    if issubclass(exc_info[0], PermissionError):
        try:
            logger.warning(f"PermissionError deleting {path}. Attempting to change permissions.")
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception as e:
            logger.error(f"Failed to remove {path} even after changing permissions: {e}", exc_info=True)
            raise exc_info[1]
    else:
        raise exc_info[1]

def run_async_in_thread(coro):
    """Helper function to run async tasks in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    event = threading.Event()
    result = None
    exception = None

    def run_in_new_loop():
        nonlocal result, exception
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            loop.close()
            event.set()

    thread = threading.Thread(target=run_in_new_loop)
    thread.start()
    event.wait()  # Wait for the thread to finish
    if exception:
        raise exception
    return result

def custom_clean_text(text: str) -> str:
    """Cleans text by applying a series of cleaning functions from RAG implementation."""
    # Basic cleaning steps (expand as needed)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.strip()
    return text


def split_into_sentences(text: str, tokenizer, min_tokens: int = 30, max_tokens: int = 384, nlp=None) -> List[str]:
    """Splits text into sentences respecting token limits using a tokenizer and optional spaCy nlp."""
    if nlp is None:
        # Fallback: simple split by period
        raw_sentences = text.split('.')
    else:
        doc = nlp(text)
        raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def get_token_length(segment: str) -> int:
        return len(tokenizer.encode(segment)) - 2 if hasattr(tokenizer, 'encode') else len(segment.split())

    sentences = []
    current_chunk = []
    current_length = 0
    for sentence in raw_sentences:
        if get_token_length(sentence) < min_tokens:
            continue
        sentence_tokens = get_token_length(sentence)
        if sentence_tokens > max_tokens:
            # Split long sentence
            words = sentence.split()
            for i in range(0, len(words), max_tokens):
                chunk = ' '.join(words[i:i+max_tokens])
                if get_token_length(chunk) >= min_tokens:
                    sentences.append(chunk)
            continue
        if current_length + sentence_tokens > max_tokens:
            if current_chunk:
                sentences.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    if current_chunk and get_token_length(' '.join(current_chunk)) >= min_tokens:
        sentences.append(' '.join(current_chunk))
    return sentences
