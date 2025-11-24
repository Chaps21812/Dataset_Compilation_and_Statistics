import functools
import time
import requests


def retry_on_443(max_retries=10, delay=2, backoff=1.5):
    """
    Retries the wrapped function when:
        - It returns a response with status_code == 443
        - It raises a ConnectionError/SSLError (typical for HTTPS handshake failures)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while True:
                try:
                    resp = func(*args, **kwargs)

                    # If it *did* return an HTTP response with 443
                    if hasattr(resp, "status_code") and resp.status_code == 443:
                        raise RuntimeError("HTTP 443 returned by server")

                    return resp  # success

                except (requests.exceptions.SSLError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout,
                        RuntimeError) as e:

                    retries += 1
                    if retries > max_retries:
                        raise

                    print(f"[retry_on_443] Retry {retries}/{max_retries}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator