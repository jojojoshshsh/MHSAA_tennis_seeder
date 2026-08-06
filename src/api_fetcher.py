# api_fetcher.py — shared auth/header/retry logic for all HTTP calls.

import asyncio
import logging
import os
import time

import aiohttp

import config as _config   # imported as module so callers can override YEAR at runtime

_TIMEOUT = aiohttp.ClientTimeout(total=20)

# ---------------------------------------------------------------------------
# Auth helper — call once, store token in env for the process lifetime.
# ---------------------------------------------------------------------------

def get_token_from_env() -> str:
    return os.environ.get("TENNIS_TOKEN", "undefined")


def set_token_in_env(token: str) -> None:
    """Store a freshly fetched token so all subsequent requests use it."""
    os.environ["TENNIS_TOKEN"] = token


async def login(session: aiohttp.ClientSession, email: str, password: str) -> str | None:
    """
    POST /auth/login and return the bearer token string, or None on failure.
    Automatically stores the token via set_token_in_env().
    """
    url = "https://api.tennisreporting.com/auth/login"
    payload = {"email": email, "password": password}
    try:
        async with session.post(url, json=payload, timeout=_TIMEOUT) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                token = data.get("token")
                if token:
                    set_token_in_env(token)
                    logging.info("login: token acquired (first 12 chars: %s…)", token[:12])
                    return token
            logging.error("login: HTTP %s", resp.status)
    except Exception as exc:
        logging.error("login: %s", exc)
    return None


def _get_headers():
    token = get_token_from_env()
    return {
        "Accept":          "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection":      "keep-alive",
        "Content-Type":    "application/json",
        "Origin":          "https://tennisreporting.com",
        "Referer":         "https://tennisreporting.com/",
        "User-Agent":      (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
        "token":           token,
        "Cache-Control":   "no-cache",
        "Pragma":          "no-cache",
    }


# ---------------------------------------------------------------------------
# Re-authentication on 401
# ---------------------------------------------------------------------------
#
# Tokens expire mid-crawl on long runs (many schools × many bracket slices).
# Once that happens every subsequent call 401s and burns its full retry
# budget for nothing. _reauthenticate() re-runs login() using the same
# TENNIS_EMAIL/TENNIS_PASSWORD env vars main_fetch.py already requires, and
# is guarded so concurrent 401s (this crawler runs many requests in
# parallel) trigger at most one real login call, with a hard cap so a
# genuinely bad credential doesn't spin forever.

_REAUTH_LOCK          = asyncio.Lock()
_REAUTH_MIN_INTERVAL  = 5.0   # seconds — don't re-login more often than this
_MAX_REAUTH_ATTEMPTS  = 5     # give up re-authenticating after this many failures
_reauth_state = {"attempts": 0, "last_ts": 0.0}


async def _reauthenticate(session: aiohttp.ClientSession) -> bool:
    """
    Re-run login() using TENNIS_EMAIL/TENNIS_PASSWORD from the environment
    and store the fresh token. Returns True if the caller should retry the
    request (a fresh token is in place, or one was *just* fetched by
    another coroutine), False if re-auth is not possible/exhausted and the
    caller should stop retrying.
    """
    async with _REAUTH_LOCK:
        now = time.monotonic()
        if now - _reauth_state["last_ts"] < _REAUTH_MIN_INTERVAL:
            # Another coroutine refreshed the token moments ago (this
            # crawler fires many concurrent requests, so a bunch of them
            # can all 401 around the same time). Assume it's fresh and
            # let the caller retry with it instead of logging in again.
            return True

        if _reauth_state["attempts"] >= _MAX_REAUTH_ATTEMPTS:
            logging.error(
                "Re-auth: giving up after %d failed re-login attempts this run.",
                _reauth_state["attempts"],
            )
            return False

        email = os.environ.get("TENNIS_EMAIL")
        password = os.environ.get("TENNIS_PASSWORD")
        if not email or not password:
            logging.error("Re-auth: TENNIS_EMAIL/TENNIS_PASSWORD not set in env — cannot re-login.")
            return False

        logging.warning("Re-auth: got HTTP 401 — token appears expired, logging in again…")
        token = await login(session, email, password)
        _reauth_state["last_ts"] = time.monotonic()

        if token:
            _reauth_state["attempts"] = 0
            logging.info("Re-auth: success — fresh token in place.")
            return True

        _reauth_state["attempts"] += 1
        logging.error(
            "Re-auth: login attempt failed (%d/%d).",
            _reauth_state["attempts"], _MAX_REAUTH_ATTEMPTS,
        )
        return False


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

async def fetch_school_report(session, school_id, gender_id=1,
                               year: int | None = None,
                               retries=3, backoff=2.0):
    """
    year defaults to config.YEAR if not provided, allowing per-call overrides
    so a multi-year runner can reuse this function without patching config.
    """
    import random
    bust = random.randint(100000, 999999)
    effective_year = year if year is not None else _config.YEAR
    url = (
        f"https://api.tennisreporting.com/report/school/{school_id}"
        f"?year={effective_year}&genderId={gender_id}"
        f"&isNotVarsity={_config.IS_NOT_VARSITY}&_={bust}"
    )
    logging.info("fetch_school_report url=%s", url)
    return await _get(session, url, f"school {school_id}", retries, backoff)


async def fetch_event(session, event_id, retries=3, backoff=2.0):
    url = f"https://api.tennisreporting.com/event/{event_id}"
    return await _get(session, url, f"event {event_id}", retries, backoff)


async def fetch_seed_list(
    session,
    event_id,
    division_id,
    host_id,
    match_type,
    flight,
    is_consolation=False,
    retries=3,
    backoff=2.0,
):
    url = f"https://api.tennisreporting.com/event/{event_id}/seed_list_by_params"
    payload = {
        "division":      division_id,
        "host":          host_id,
        "matchType":     match_type,
        "flight":        flight,
        "isConsolation": is_consolation,
    }
    label = f"seed_list e={event_id} h={host_id} {match_type}[{flight}]"
    return await _post(session, url, payload, label, retries, backoff)


async def fetch_bracket(
    session,
    event_id,
    host_id,
    division_id,
    match_type,
    flight,
    is_consolation=False,
    retries=3,
    backoff=2.0,
):
    url = f"https://api.tennisreporting.com/event/{event_id}/host/{host_id}/bracket/get"
    payload = {
        "division":      division_id,
        "host":          host_id,
        "matchType":     match_type,
        "flight":        flight,
        "isConsolation": is_consolation,
    }
    label = f"bracket e={event_id} h={host_id} {match_type}[{flight}]"
    return await _post(session, url, payload, label, retries, backoff)


# ---------------------------------------------------------------------------
# Internal GET / POST with retry
# ---------------------------------------------------------------------------

async def _get(session, url, label, retries, backoff):
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, headers=_get_headers(), timeout=_TIMEOUT) as resp:
                logging.info("%s: HTTP %s", label, resp.status)
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    if isinstance(data, dict):
                        meets = data.get("meets", [])
                        logging.info("%s: got %d meets", label, len(meets))
                    return data
                if resp.status == 304:
                    logging.warning("%s: 304 Not Modified — returning None", label)
                    return None
                if resp.status == 401:
                    if not await _reauthenticate(session):
                        logging.error("%s: re-auth failed — aborting retries.", label)
                        return None
                    # Fresh token is in place; retry immediately without
                    # the exponential backoff (that sleep was meant for
                    # transient errors, not an expired token).
                    continue
                logging.warning(
                    "%s: HTTP %s (attempt %d/%d)", label, resp.status, attempt, retries
                )
        except asyncio.TimeoutError:
            logging.warning("%s: timeout (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:
            logging.error("%s: %s (attempt %d/%d)", label, exc, attempt, retries)
        if attempt < retries:
            await asyncio.sleep(backoff * attempt)
    logging.error("%s: giving up after %d attempts.", label, retries)
    return None


async def _post(session, url, payload, label, retries, backoff):
    for attempt in range(1, retries + 1):
        try:
            async with session.post(
                url, headers=_get_headers(), json=payload, timeout=_TIMEOUT
            ) as resp:
                logging.info("%s: HTTP %s", label, resp.status)
                if resp.status == 200:
                    return await resp.json(content_type=None)
                if resp.status == 304:
                    logging.warning("%s: 304 Not Modified — returning None", label)
                    return None
                if resp.status == 401:
                    if not await _reauthenticate(session):
                        logging.error("%s: re-auth failed — aborting retries.", label)
                        return None
                    # Fresh token is in place; retry immediately without
                    # the exponential backoff (that sleep was meant for
                    # transient errors, not an expired token).
                    continue
                logging.warning(
                    "%s: HTTP %s (attempt %d/%d)", label, resp.status, attempt, retries
                )
        except asyncio.TimeoutError:
            logging.warning("%s: timeout (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:
            logging.error("%s: %s (attempt %d/%d)", label, exc, attempt, retries)
        if attempt < retries:
            await asyncio.sleep(backoff * attempt)
    logging.error("%s: giving up after %d attempts.", label, retries)
    return None
