import asyncio
import logging
import os
import stat
import threading

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
