# api_fetcher.py — shared auth/header/retry logic for all HTTP calls.

import asyncio
import logging
import os

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
