"""The weather for a named place, as a tool the model can call.

One ``FunctionTool``, :func:`weather_forecast`, exposed through
:func:`build_weather_tools` whenever ``WEATHER_ENABLED`` is not false. It answers the
question a person actually asks -- *"me fala a temperatura para osasco amanhã"* -- with
every daily metric the provider publishes for that place and that day: max and min
temperature, apparent temperature, humidity, rain chance, precipitation, wind, gusts,
UV, sunshine, sunrise/sunset, radiation and evapotranspiration.

**Why a dedicated API rather than the web tier.** Rule 13 already sends the model to
``web_search``/``web_fetch``, and for a forecast that path is worse in every dimension
that matters: the answer is behind a page built for humans, the numbers are the same on
ten sites, and a search snippet is the single most poisonable channel the agent has
(a page author's own ``<meta name="description">``). A purpose-built JSON endpoint gives
one authoritative number per metric, one request, and a fact rather than a paragraph to
read. It is also the only way to answer *"and humidity?"* -- a follow-up on the same
place and day is one parameter away here and a second search on the web tier.

**Open-Meteo by default, and configurable.** Free, keyless, and the only no-key
provider that publishes all of the above; ``WEATHER_GEOCODING_URL`` and
``WEATHER_FORECAST_URL`` point it somewhere else (a mirror, or a self-hosted instance)
without touching the code. Both are *operator-configured*, exactly like ``SEARXNG_URL``,
which is why they are vetted and dialled with ``allow_private=True`` -- a self-hosted
endpoint is on a Docker bridge. Link-local stays refused either way. See
``web_search.check_url``.

**The one thing this cannot do is guess.** Two failure modes are designed against
rather than handled:

* **A place name is not a place.** "Springfield" is three US cities of near-identical
  size; "Valencia" is two countries, and the *larger* one (Venezuela, 1.62M) is not
  the one Open-Meteo ranks first. :func:`resolve_place` refuses to choose and returns
  the candidates, because a confidently wrong city's forecast is worse than a question.
* **A named city is not always the city meant.** "Nova York" resolves to *Nova Iorque,
  Maranhão, Brasil* -- population 4,320 -- because that is a real place with that name,
  and "New York" resolves to *York, Nebraska* -- 7,864 -- which is a plausible-looking
  city and a worse answer. Both are measured; neither is fixed by ``WEATHER_LANGUAGE``
  or by any code. Only ``place="New York, NY"`` (or ``"Nova Iorque"``, the Portuguese
  spelling of the real one) lands on 40.71, -74.01. So the resolved place (name, state,
  country, population) is **in the payload** and rule 17 makes the model check it against
  what the user asked before reporting numbers -- which is the only thing standing
  between a wrong county and a confident answer.

**Errors are data.** Every path returns a readable ``{"error": ..., "hint": ...}`` and
none of them raises, for ``web_search._error``'s reason verbatim: the model has to be
able to read a failure and fall through to another tool rather than have the turn die.
That includes a date the provider cannot serve -- Open-Meteo answers a request past its
horizon with HTTP 400 and a ``reason`` naming the range, which is passed through rather
than flattened, because "no forecast that far out" is something the user can act on.

**The returned strings are sanitised rather than framed.** Unlike a fetched page, this
payload is numbers the code selected, and the only free text is a gazetteer name that
the model has to reproduce in its answer. Wrapping it in ``<untrusted_content>`` would
put the wrapper in the user's answer; instead :func:`_field` collapses whitespace, drops
markup characters and bounds the length, which is what actually makes it inert.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin

from google.adk.tools.function_tool import FunctionTool

# The SSRF-pinned HTTP path is shared rather than reimplemented. web_search owns the
# vetting, the address pinning and the body ceiling, and a second GET helper here would
# be a second place for a future fix to be forgotten -- which is exactly what gotcha 19
# records. The private names are imported deliberately and listed here so the coupling
# is visible: `_http_get` (the pinned GET), `_fold` (accent folding, so this file and
# the age-gate vocabulary cannot disagree on what "amanha" is), `_BodyTooLarge`, and
# `record_returned_urls` (the citation allow-list a `[web]` line is checked against).
from .web_search import (
    _BodyTooLarge,
    _fold,
    _http_get,
    check_url,
    record_returned_urls,
)

#: ``false`` removes the tool without unsetting anything. Mirrors
#: ``WEB_SEARCH_ENABLED`` and ``CACHE_ENABLED``: opt-out rather than opt-in, because a
#: deployment that never heard of this tier has no way to configure it and would
#: silently keep answering the weather from a training cutoff.
WEATHER_ENABLED_ENV = "WEATHER_ENABLED"

#: Both endpoints are operator-configurable, for a mirror or a self-hosted instance.
GEOCODING_URL_ENV = "WEATHER_GEOCODING_URL"
FORECAST_URL_ENV = "WEATHER_FORECAST_URL"

#: Rendered in slot two of every ``[web]`` source line this tool produces. "Where this
#: came from" is the question that slot answers, so it names the retrieval tier and
#: never the vault -- see ``agent._web_provider_label``.
PROVIDER_ENV = "WEATHER_PROVIDER"

#: Language of the gazetteer's own names. Measured, not assumed: with ``pt``, "Londres"
#: resolves to London (8.96M) instead of a 2,627-person village in Catamarca, and with
#: the default ``en`` it resolves to the Argentine one. The converse is real too --
#: neither setting fixes "Nova York", which is why the resolved place is reported and
#: checked rather than assumed right.
LANGUAGE_ENV = "WEATHER_LANGUAGE"

DEFAULT_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_PROVIDER = "open-meteo"
DEFAULT_LANGUAGE = "pt"

GEOCODING_TIMEOUT_S = 10.0
FORECAST_TIMEOUT_S = 12.0

#: Wall-clock ceiling for one call -- two requests plus redirects. Per-operation
#: timeouts multiply the way ``FETCH_TOTAL_TIMEOUT_S`` does in web_search, so without
#: this an endpoint that dribbles one byte per read timeout holds the turn open.
TOTAL_TIMEOUT_S = 30.0

#: Ceiling on a JSON body, passed to the streaming guard so it aborts rather than
#: buffering. A forecast for 14 days is a few kilobytes; two orders of magnitude is
#: already generous.
MAX_RESPONSE_BYTES = 1 * 1024 * 1024

#: How many candidates to ask the geocoder for, and how many to name back to the model.
#: Measured: "Springfield" returns ten, of which the first three are the candidates
#: that matter and the rest are places with different names that happen to rank.
GEOCODING_COUNT = 10
MAX_REPORTED_CANDIDATES = 8

#: Days in one answer. Open-Meteo's own horizon is 16 ahead and 92 behind; this is
#: clamped well inside it so an over-long request is bounded by the code rather than by
#: a 400, and anything the provider still refuses is reported with its own reason.
MAX_DAYS = 14

#: How far ahead the top candidate must lead on population before the choice is made
#: without asking. Measured against real queries:
#:
#:     "Osasco"    728,615 vs        645  -> decisive (the Italian hamlet)
#:     "Paris"   2,138,551 vs     24,782  -> decisive
#:     "Lisboa"    517,802 vs          0  -> decisive
#:     "Barcelona" 1,686,208 vs   815,141  -> 2.1x, asked
#:     "Valencia"   824,340 vs 1,619,470  -> the *runner-up* is bigger, asked
#:     "Springfield" 170,188 vs   154,341  -> 1.1x, asked
#:
#: Four is deliberately conservative. Choosing wrong across a border reports another
#: hemisphere's weather as fact, while the cost of asking is one turn -- and rule 15
#: already grants the question when the options really fork.
DOMINANCE_RATIO = 4

#: GeoNames ``feature_code`` as an administrative rank, used as the *other* decisive
#: signal: a national capital beats a state capital beats a populated place, whatever
#: the populations are. "Londres" (PPLC, 8.96M) over the Argentine hamlet is decided by
#: this alone. Unranked codes sort last, so an unknown code can never win on it.
_FEATURE_RANK = {
    "PPLC": 0,
    "PPLA": 1,
    "PPLA2": 2,
    "PPLA3": 3,
    "PPLA4": 4,
    "PPL": 5,
    "PPLX": 6,
}
_UNRANKED_FEATURE = 7

#: Open-Meteo daily variable -> the key it is reported under. Every output name carries
#: its own unit, because the tool answers in metric only and a bare ``humidity_mean``
#: invites the model to say "84 degrees". Verified 200 against the live API: one
#: variable wanted here (``vapour_pressure_deficit``) is not served on the free tier,
#: and a single unknown name makes the whole request 400, which is why the list is
#: verified rather than assembled from the documentation.
_DAILY_FIELDS: dict[str, str] = {
    "weather_code": "condition_code",
    "temperature_2m_max": "temperature_max_c",
    "temperature_2m_min": "temperature_min_c",
    "temperature_2m_mean": "temperature_mean_c",
    "apparent_temperature_max": "apparent_temperature_max_c",
    "apparent_temperature_min": "apparent_temperature_min_c",
    "relative_humidity_2m_min": "humidity_min_pct",
    "relative_humidity_2m_mean": "humidity_mean_pct",
    "relative_humidity_2m_max": "humidity_max_pct",
    "precipitation_probability_max": "rain_chance_pct",
    "precipitation_sum": "precipitation_mm",
    "precipitation_hours": "precipitation_hours",
    "rain_sum": "rain_mm",
    "showers_sum": "showers_mm",
    "snowfall_sum": "snowfall_cm",
    "wind_speed_10m_max": "wind_max_kmh",
    "wind_gusts_10m_max": "wind_gusts_max_kmh",
    "wind_direction_10m_dominant": "wind_direction_deg",
    "uv_index_max": "uv_index_max",
    "sunshine_duration": "sunshine_hours",
    "daylight_duration": "daylight_hours",
    "sunrise": "sunrise",
    "sunset": "sunset",
    "shortwave_radiation_sum": "radiation_mj_m2",
    "et0_fao_evapotranspiration": "evapotranspiration_mm",
}

#: Open-Meteo hourly variable -> the key it is reported under. Same convention as
#: :data:`_DAILY_FIELDS` (one place names the provider's spelling and the output's), and
#: verified 200 one at a time against the live API for the same reason: a single
#: unknown name 400s the whole request, so this list cannot be assembled from the
#: documentation.
#:
#: Six variables, not more, and the choice is a payload-size one rather than a
#: completeness one. Measured: these six for one day arrive as **1,341 bytes** in the
#: compact form below, against **15,385** raw for fourteen. Every extra variable costs
#: another 24 numbers on every weather turn, and the free tier that serves most turns
#: here is the one least able to afford the context -- so the set is what a curve and
#: an "is it raining this afternoon" question actually need, and not everything the
#: provider publishes. Deliberately absent: ``weather_code``, because 24 condition
#: phrases is prose the model must translate rather than numbers it can plot, and the
#: daily row already carries the day's condition.
_HOURLY_FIELDS: dict[str, str] = {
    "temperature_2m": "temperature_c",
    "apparent_temperature": "apparent_c",
    "precipitation_probability": "rain_chance_pct",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_kmh",
}

#: How many hours of the series the payload carries.
#:
#: A **cap that does not move with ``days``**, which is the property worth having: at
#: fourteen days the provider's own block is 336 hours and ~18 KB of the model's
#: context, on a turn whose whole point is usually one day. Twenty-four is one day --
#: the day asked about, on every request whose ``days`` is 1, which is the default and
#: the common case -- and the curve drawn from it is the width a markdown line can
#: hold. Withheld hours are reported as ``hourly.truncated``, after
#: ``digest_tools.read_day_digest``: the true total stays visible, so a capped list can
#: never be read as a quiet day.
MAX_HOURLY_HOURS = 24

#: Session-state key holding the hourly series this turn's ``weather_forecast`` call
#: returned, for the same reason and under the same ``invocation_id`` keying as
#: ``web_search.WEB_URLS_STATE_KEY``: the chart is rendered in
#: ``sources.render_sources`` from the answer's own callback, which has no view of the
#: tool result -- and ``session.events`` is not populated under ``adk web``.
HOURLY_STATE_KEY = "_weather_hourly_by_invocation"

#: Durations the provider publishes in seconds and a reader thinks in hours.
_SECONDS_DIVISOR = {"sunshine_hours": 3600.0, "daylight_hours": 3600.0}

#: Local clock times, trimmed from the provider's ``2026-10-05T05:43``.
_CLOCK_FIELDS = ("sunrise", "sunset")

#: WMO weather codes, in the vocabulary Open-Meteo publishes them under. English on
#: purpose: the answer's language follows the user, and a code the model has to look up
#: is worse than a short English phrase it can translate.
WEATHER_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

#: Units block, restated rather than left implicit. Open-Meteo's own ``daily_units`` is
#: echoed beside this one and they must agree; a reader that has to infer what ``8.7``
#: means is a reader that reports it wrong.
UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "rain_chance": "%",
    "precipitation": "mm",
    "rain": "mm",
    "showers": "mm",
    "snowfall": "cm",
    "precipitation_hours": "h",
    "wind": "km/h",
    "wind_direction": "degrees",
    "uv_index": "index (0-11+)",
    "sunshine": "h",
    "daylight": "h",
    "radiation": "MJ/m²",
    "evapotranspiration": "mm",
    "elevation": "m",
    "times": "local to the place, 24h",
}

#: Relative day names the tool resolves *itself*, against the server's own clock.
#:
#: In code rather than left to the model, because "amanhã" is arithmetic and a model
#: doing arithmetic from a date it read out of a tool is one hallucinated day away.
#: Both languages, because the user's phrasing is not the tool's to assume -- and the
#: accent is folded before the lookup, so "amanha" is the same word.
_RELATIVE_DAYS = {
    "today": 0,
    "hoje": 0,
    "tomorrow": 1,
    "amanha": 1,
    "yesterday": -1,
    "ontem": -1,
}

_MARKUP = str.maketrans("", "", "<>`\"'\\|{}[]()")


def _field(value: Any, limit: int = 80) -> str:
    """One string from the provider, made inert and short.

    Collapses whitespace, drops the characters that could carry markup or a pipe into
    the rendered answer, and bounds the length. This is the treatment a gazetteer name
    needs -- see the module docstring for why it is sanitisation rather than
    ``<untrusted_content>`` framing. Applied to the free-text fields only; numbers are
    never routed through here.
    """
    text = " ".join(str(value or "").split())
    return text.translate(_MARKUP)[:limit]


def weather_enabled() -> bool:
    """Whether the weather tool is offered at all."""
    raw = os.environ.get(WEATHER_ENABLED_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def provider_name() -> str:
    """The retrieval-tier name a ``[web]`` source line carries for this tool."""
    return (os.environ.get(PROVIDER_ENV) or "").strip() or DEFAULT_PROVIDER


def _geocoding_url() -> str:
    return (os.environ.get(GEOCODING_URL_ENV) or "").strip() or DEFAULT_GEOCODING_URL


def _forecast_url() -> str:
    return (os.environ.get(FORECAST_URL_ENV) or "").strip() or DEFAULT_FORECAST_URL


def _language() -> str:
    return (os.environ.get(LANGUAGE_ENV) or "").strip() or DEFAULT_LANGUAGE


def build_weather_tools() -> list:
    """The weather tool, or ``[]`` when ``WEATHER_ENABLED`` is false.

    Same shape as ``build_obsidian_tools``, ``build_gmail_tools`` and
    ``build_web_search_tools``: the gate lives entirely in here and the caller spreads
    the result unconditionally. Note what is *not* a gate -- there is no key and no
    configured URL to be missing, because both defaults are public endpoints.
    """
    if not weather_enabled():
        return []
    return [FunctionTool(weather_forecast)]


@dataclass(frozen=True)
class Place:
    """One geocoded place. Every string field is already sanitised."""

    name: str
    latitude: float
    longitude: float
    admin1: str = ""
    country: str = ""
    country_code: str = ""
    timezone: str = ""
    elevation_m: float | None = None
    population: int | None = None
    feature_code: str = ""

    @property
    def rank(self) -> int:
        """Administrative rank; lower is more significant. See :data:`_FEATURE_RANK`."""
        return _FEATURE_RANK.get(self.feature_code, _UNRANKED_FEATURE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "admin1": self.admin1,
            "country": self.country,
            "country_code": self.country_code,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "timezone": self.timezone,
            "population": self.population,
        }

    def label(self) -> str:
        """``Osasco, São Paulo, Brasil`` -- the shortest honest description of it."""
        parts = [self.name, self.admin1, self.country]
        return ", ".join(p for p in parts if p) or self.name


class AmbiguousPlace(ValueError):
    """The place name matched several places and none of them clearly wins.

    A ``ValueError`` so every caller that already handles one handles this too, with
    ``candidates`` carried alongside the message -- and the tool turns it into the one
    payload shape the model is told what to do about.
    """

    def __init__(self, message: str, candidates: list[Place]) -> None:
        super().__init__(message)
        self.candidates = candidates


def _number(value: Any) -> float | None:
    """A float from JSON, or ``None`` for null/absent/non-numeric.

    ``bool`` is rejected explicitly: ``True`` is an ``int`` in Python, so a provider
    that ever sent a boolean where a number belongs would otherwise become ``1.0``.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _place_from(raw: Any) -> Place | None:
    """One candidate, or ``None`` when it cannot be used.

    Every field is read with ``.get`` and every string sanitised: a candidate missing
    ``country`` really does occur in the live response (measured), and the alternative
    was a ``KeyError`` out of a tool whose contract is that it never raises.
    """
    if not isinstance(raw, dict):
        return None
    name = _field(raw.get("name"))
    latitude = _number(raw.get("latitude"))
    longitude = _number(raw.get("longitude"))
    if not name or latitude is None or longitude is None:
        return None
    population = raw.get("population")
    return Place(
        name=name,
        latitude=latitude,
        longitude=longitude,
        admin1=_field(raw.get("admin1")),
        country=_field(raw.get("country")),
        country_code=_field(raw.get("country_code"), 8).upper(),
        timezone=_field(raw.get("timezone"), 64),
        elevation_m=_number(raw.get("elevation")),
        # `isinstance(True, int)` is true in Python, so a boolean is excluded here for
        # the same reason `_number` excludes it: it would otherwise become a population
        # of 1.
        population=population if isinstance(population, int) and not isinstance(population, bool) else None,
        feature_code=_field(raw.get("feature_code"), 16),
    )


def _named_like(candidates: list[Place], name: str) -> list[Place]:
    """The candidates whose *own name* contains what was asked for.

    The geocoder ranks by relevance, not by name match, so the tail of a common name is
    other places: ten results for "Springfield" include a *Palmyra* and a *Jackson*
    (both in Missouri). Offering those to a user who asked about Springfield is noise
    that makes the question harder to answer, and it also lets an unrelated settlement
    dilute the dominance test in :func:`resolve_place`.

    The filter is abandoned when it would empty the list rather than turn a match into
    "no place found", because the API also *translates*: "Nova York" matches the real
    place named "Nova Iorque" in any language, and a name-equality filter would report
    that as nothing at all. Every candidate is a better answer than an error there.
    """
    wanted = _fold(name).strip()
    if not wanted:
        return candidates
    named = [c for c in candidates if wanted in _fold(c.name)]
    return named or candidates


def _api_json(
    url: str, *, timeout: float, with_error_body: bool = False
) -> tuple[int, dict[str, Any]]:
    """GET a JSON document from a configured endpoint.

    Returns ``(status, document)``: the parsed object on 2xx, ``({}, )``-shaped empty
    mapping when there is nothing to parse, and on an error status either ``{}`` or --
    with ``with_error_body`` -- the provider's own error document.

    That flag exists for one caller and one reason. Open-Meteo answers a request past
    its horizon with HTTP 400 and ``{"reason": "Parameter 'start_date' is out of
    allowed range from 2026-07-03 to 2026-10-19"}``, and that sentence is the whole
    difference between a model that can tell the user why and one that reports "HTTP
    400". ``web_search._http_get`` discards error bodies, because ``web_fetch`` has no
    use for them; so it takes the flag rather than gaining a second GET helper here.
    """
    current = url
    for _ in range(3):
        vetted: list = []
        try:
            problem = check_url(current, vetted=vetted, allow_private=True)
        except Exception as exc:  # never raise out of a "never raise" function
            raise ValueError(f"could not resolve the weather service: {exc}") from exc
        if problem:
            raise ValueError(f"cannot reach the weather service: {problem}")
        if not vetted:
            raise ValueError("cannot reach the weather service: no address to connect to")

        try:
            status, location, payload = _http_get(
                current,
                timeout=timeout,
                max_bytes=MAX_RESPONSE_BYTES,
                address=vetted[0],
                include_error_body=with_error_body,
            )
        except _BodyTooLarge as exc:
            raise ValueError(f"the weather service sent too much data: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"the weather service could not be reached: {exc}") from exc

        if status in (301, 302, 303, 307, 308):
            if not location:
                raise ValueError("the weather service redirected with no Location header")
            # urljoin, for the same reason fetch_page_text uses it: a Location may be
            # absolute, root-relative or bare-relative, and the hop is re-vetted above.
            current = urljoin(current, location)
            continue

        if payload is None:
            # An error status, or a redirect with no body. Either way there is nothing
            # to parse, and ``web_search._http_get`` only returns a body on an error
            # when the caller asked for one.
            return status, {}
        _content_type, body = payload
        try:
            document = json.loads(body.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"the weather service returned non-JSON: {exc}") from exc
        # A scalar or a list parses fine and then has no .get, so without this the
        # AttributeError would escape the tool -- the exact bug the "errors are data"
        # contract exists to prevent.
        if not isinstance(document, dict):
            raise ValueError(
                f"the weather service returned a JSON {type(document).__name__}, "
                "not the expected object"
            )
        return status, document

    raise ValueError("too many redirects from the weather service")


def _reason_for(status: int, document: dict[str, Any]) -> str:
    """The provider's own explanation of a refusal, when it gave one."""
    for key in ("reason", "error", "message"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return _field(value, 300)
    return ""


def geocode(place: str, *, country: str = "", base_url: str | None = None) -> list[Place]:
    """Places named ``place``, best first. Raises ``ValueError``; never returns empty.

    Two corrections are applied to what the geocoder returns, in this order: the tail of
    its ranking is dropped by :func:`_named_like`, and then ``country`` is applied
    **here rather than in the query string**.

    That second one is a measured correction, not a stylistic one. Open-Meteo's
    geocoding endpoint accepts a ``country`` parameter and ignores it: with
    ``country=BR`` it still returns the Italian Osasco, and with ``country=US`` it still
    returns three Springfields. Filtering on ``country_code`` from the response is
    therefore the only version of the filter that does anything, and sending the
    parameter anyway would advertise a guarantee the API does not offer.
    """
    name = _field(place, 120)
    if not name:
        raise ValueError("no place name was given")

    base = (base_url if base_url is not None else _geocoding_url()).rstrip("/")
    query = {
        "name": name,
        "count": GEOCODING_COUNT,
        "format": "json",
        "language": _language(),
    }
    url = f"{base}?{urlencode(query)}"

    status, document = _api_json(url, timeout=GEOCODING_TIMEOUT_S)
    if status >= 400:
        detail = _reason_for(status, document)
        raise ValueError(
            f"the geocoder returned HTTP {status}" + (f": {detail}" if detail else "")
        )

    found: list[Place] = []
    for raw in document.get("results") or []:
        candidate = _place_from(raw)
        if candidate is not None:
            found.append(candidate)

    wanted = _field(country, 8).upper()
    if wanted:
        found = [c for c in found if c.country_code == wanted]
    if not found:
        if wanted:
            raise ValueError(
                f"no place called {name!r} in country {wanted!r} was found. That is a "
                "two-letter country code, not a state or region."
            )
        raise ValueError(f"no place called {name!r} was found")
    return _named_like(found, name)


def resolve_place(place: str, *, country: str = "", base_url: str | None = None) -> Place:
    """The one place ``place`` names, or :class:`AmbiguousPlace` when it names several.

    Two independent signals decide, and either is enough -- see
    :data:`DOMINANCE_RATIO` and :data:`_FEATURE_RANK` for the measured queries behind
    them.

    The way out of a genuine tie is *not* the ``country`` argument alone: a US state is
    not a country, and ``Springfield, Illinois`` or ``Springfield, IL`` as the ``place``
    resolves the ambiguity in one call (measured). ``country`` is there for the
    cross-border case -- ``Valencia`` with ``country=ES`` -- where the two candidates
    are in different countries.
    """
    found = geocode(place, country=country, base_url=base_url)
    if len(found) == 1:
        return found[0]

    best = found[0]
    rivals = found[1:]
    best_population = best.population or 0
    rival_population = max((r.population or 0) for r in rivals)
    by_population = best_population >= DOMINANCE_RATIO * max(rival_population, 1)
    by_rank = best.rank < min(rival.rank for rival in rivals)
    if by_population or by_rank:
        return best

    names = ", ".join(f"{r.label()} ({r.population:,} hab.)" if r.population else r.label() for r in found)
    raise AmbiguousPlace(
        f"{place!r} matches {len(found)} places and none of them clearly wins: {names}",
        found,
    )


def start_date(raw: str, *, today: datetime.date | None = None) -> datetime.date:
    """The first day a ``date`` argument names.

    Accepts an ISO date, ``today``/``tomorrow``/``yesterday`` in English or Portuguese
    (accents folded), and ``""`` for today. Anything else raises, because a relative
    expression it cannot resolve must not silently become *today* -- "last Tuesday"
    answered with today's forecast is a wrong answer wearing the shape of a right one.
    """
    base = today or datetime.date.today()
    text = _fold(raw or "").strip()
    if not text:
        return base
    if text in _RELATIVE_DAYS:
        return base + datetime.timedelta(days=_RELATIVE_DAYS[text])
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"{raw!r} is not a date this tool understands. It accepts YYYY-MM-DD, "
            "'today', 'tomorrow' or 'yesterday' (or 'hoje', 'amanha', 'ontem'). For "
            "anything else call current_datetime first and pass the day as YYYY-MM-DD."
        ) from None


def _forecast_query(place: Place, first: datetime.date, days: int) -> str:
    base = _forecast_url().rstrip("/")
    query = {
        "latitude": round(place.latitude, 4),
        "longitude": round(place.longitude, 4),
        "daily": ",".join(_DAILY_FIELDS),
        # The hourly block rides along on the same request rather than as a second one.
        # Open-Meteo answers both from one document in one call, so this costs no
        # extra round trip and no extra failure mode -- there is no second request that
        # can succeed while the first fails.
        "hourly": ",".join(_HOURLY_FIELDS),
        "timezone": "auto",
        # start_date/end_date rather than past_days/forecast_days: the requested range
        # is an absolute one, and asking for it directly is what lets the provider
        # answer in the place's own timezone and refuse a range it cannot serve.
        "start_date": first.isoformat(),
        "end_date": (first + datetime.timedelta(days=days - 1)).isoformat(),
    }
    return f"{base}?{urlencode(query)}"


def _hourly_window(
    block: Any, offset_seconds: int, now: datetime.datetime | None
) -> tuple[dict | None, dict | None]:
    """The provider's hourly block, renamed, capped, and paired with ``now``.

    Returns ``(series, current)``, and **either may be absent**: a provider that sends
    no hourly block yields no keys at all rather than empty ones, which is the same
    discipline ``weather_forecast`` already applies to an unresolved place (no
    ``forecast`` key, so there is nothing to quote from a refusal).

    Parallel arrays rather than a list of row objects, because the shape decides the
    size: 24 rows of five named fields serialise to **2,772** bytes where the same data
    as parallel arrays is **1,341** -- and every weather turn pays it. The ``time`` axis
    is stated once and every other array lines up with it by index.

    ``current`` is the hour *containing* now, chosen by flooring rather than rounding:
    at 16:19 the answer is the 16:00 reading, and rounding up would report a forecast
    for a moment that has not happened. It is present only when now falls inside the
    window carried here, so a question about a future day gets no ``current`` -- the
    honest answer, since the alternative is the day's *mean* presented as "now", which
    is precisely the defect this was added to close (session
    ``9c40b78b-3d1c-41ff-bd8a-ccaec8c85b7f``, where "temperatura agora" was answered
    with the daily average).
    """
    if not isinstance(block, dict):
        return None, None
    times = block.get("time")
    if not isinstance(times, list) or not times:
        return None, None

    total = len(times)
    kept = min(total, MAX_HOURLY_HOURS)
    series: dict[str, Any] = {
        "unit": "hour",
        "axis": "every array below lines up with time[] by index",
        "hours": kept,
        "time": [_field(t, 16) for t in times[:kept]],
    }
    if total > kept:
        series["truncated"] = total - kept

    for variable, key in _HOURLY_FIELDS.items():
        values = block.get(variable)
        if not isinstance(values, list):
            continue
        column: list[float | int | None] = []
        for raw in values[:kept]:
            value = _number(raw)
            if value is None:
                column.append(None)
                continue
            # Same integral-value rule as the daily row: the provider sends
            # percentages as JSON integers, which a float division turns into "84.0%".
            column.append(int(value) if float(value).is_integer() else round(value, 1))
        series[key] = column

    return series, _current_reading(series, offset_seconds, now)


def _current_reading(
    series: dict[str, Any], offset_seconds: int, now: datetime.datetime | None
) -> dict | None:
    """This hour's readings, computed here so the model never does the arithmetic.

    ``now`` is the **place's** wall clock: the provider's ``utc_offset_seconds`` is what
    makes that possible, and it is the same field the daily rows already carry. Working
    in UTC and reporting it as local time is the failure mode -- it would put "agora" on
    the wrong hour for every place whose offset is not zero, which is every place
    outside west Africa and the UK.
    """
    if now is None:
        return None
    times = series.get("time") or []
    try:
        local = now.astimezone(datetime.UTC) + datetime.timedelta(
            seconds=offset_seconds
        )
    except (OverflowError, OSError, ValueError):  # pragma: no cover - absurd offset
        return None
    # tzinfo dropped, and the minutes floored: the provider's stamps are naive local
    # times, so a `+00:00` suffix here would match nothing and silently yield no
    # reading of the present at all.
    stamp = local.replace(tzinfo=None, minute=0, second=0, microsecond=0).isoformat(
        timespec="minutes"
    )
    index = times.index(stamp) if stamp in times else None
    if index is None:
        # The window starts at the requested day and today is not in it, or the
        # provider's clock differs by an hour. Either way there is no reading of *now*
        # here and inventing one is the one answer this module must never produce.
        return None

    reading: dict[str, Any] = {
        "time": times[index],
        "hour": times[index][-5:],
        "local_date": local.date().isoformat(),
        "local_time": local.strftime("%H:%M"),
    }
    for key in _HOURLY_FIELDS.values():
        column = series.get(key)
        if isinstance(column, list) and index < len(column):
            reading[key] = column[index]
    return reading


def record_hourly_series(tool_context: Any, series: Any, current: Any, place: Any = "") -> None:
    """Note the hourly series this turn returned, so the chart can be drawn from it.

    Recorded by the tool, at the moment it returns the numbers, and keyed by
    ``invocation_id`` -- identical to ``web_search.record_returned_urls`` and for the
    same reason. The alternative, reconstructing it from ``session.events``, does not
    work in production: that list is not populated under ``adk web``'s database session
    service, which is the only place a chart can be rendered from anyway (the tool
    result is gone by the time the answer's ``after_model_callback`` runs).

    Best-effort and silent on failure, like every other state write on this path: a
    missing chart costs a curve in the answer, never a turn.
    """
    if tool_context is None or not isinstance(series, dict):
        return
    invocation_id = getattr(tool_context, "invocation_id", None)
    if not invocation_id:
        return
    try:
        state = tool_context.state
        recorded = state.get(HOURLY_STATE_KEY) or {}
        if not isinstance(recorded, dict):
            recorded = {}
        # The state object is a delta, so assign the whole key back. `place` is the name
        # the tool *resolved*, not the one it was asked for: the chart is titled with
        # the city the numbers are for, and an ambiguous match never gets this far.
        recorded[invocation_id] = {
            "hourly": series,
            "current": current if current else None,
            "place": _field(place, 80),
        }
        state[HOURLY_STATE_KEY] = recorded
    except Exception:  # pragma: no cover - never break the turn
        pass


def _day(index: int, daily: dict[str, Any], offset_seconds: int) -> dict[str, Any]:
    """One day of the provider's daily block, in this module's field names."""
    day: dict[str, Any] = {}
    times = daily.get("time") or []
    iso = times[index] if index < len(times) else ""
    day["date"] = _field(iso, 10)
    try:
        day["weekday"] = datetime.date.fromisoformat(day["date"]).strftime("%A")
    except ValueError:
        day["weekday"] = ""

    for variable, key in _DAILY_FIELDS.items():
        values = daily.get(variable)
        if not isinstance(values, list) or index >= len(values):
            continue
        raw = values[index]
        if key == "condition_code":
            code = _number(raw)
            day[key] = int(code) if code is not None else None
            # The code alone is not an answer; the phrase is, and an unrecognised code
            # reports itself rather than borrowing a neighbour's wording.
            day["condition"] = WEATHER_CONDITIONS.get(int(code or -1), "unknown code")
            continue
        if key in _CLOCK_FIELDS:
            text = _field(raw, 24)
            day[key] = text.split("T")[-1] if "T" in text else text
            continue
        value = _number(raw)
        if value is None:
            continue
        if key in _SECONDS_DIVISOR:
            value = round(value / _SECONDS_DIVISOR[key], 2)
        # An integral value is reported as an int. The provider sends percentages and
        # counts as JSON integers and this turns them into floats, which reaches the
        # answer as "84.0% humidity" -- true, and not what anybody writes.
        day[key] = int(value) if float(value).is_integer() else value
    day["utc_offset_seconds"] = offset_seconds
    return day


def forecast(
    place: Place, first: datetime.date, days: int, now: datetime.datetime | None = None
) -> tuple[list[dict], dict | None, dict | None, dict, str]:
    """The forecast for ``place``, as ``(days, hourly, current, provider_metadata, url)``.

    ``hourly`` and ``current`` are ``None`` when the provider sent no hourly block, so
    a provider that does not serve one degrades to the daily-only tool it was before
    rather than failing. ``now`` is injected rather than read here so the current-hour
    arithmetic is testable against a fixed clock.
    """
    count = max(1, min(int(days), MAX_DAYS))
    url = _forecast_query(place, first, count)
    status, document = _api_json(url, timeout=FORECAST_TIMEOUT_S, with_error_body=True)
    if status >= 400:
        detail = _reason_for(status, document)
        raise ValueError(
            f"the forecast service returned HTTP {status}" + (f": {detail}" if detail else "")
        )

    daily = document.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("the forecast service returned no daily data")
    offset = int(_number(document.get("utc_offset_seconds")) or 0)
    times = daily.get("time") or []
    rows = [_day(index, daily, offset) for index in range(len(times))][:count]
    if not rows:
        raise ValueError("the forecast service returned no days for the requested range")

    series, current = _hourly_window(document.get("hourly"), offset, now)

    metadata: dict[str, Any] = {
        "timezone": _field(document.get("timezone"), 64),
        "timezone_abbreviation": _field(document.get("timezone_abbreviation"), 24),
        "elevation_m": _number(document.get("elevation")),
        "daily_units": (
            document.get("daily_units") if isinstance(document.get("daily_units"), dict) else {}
        ),
    }
    return rows, series, current, metadata, url


def _failure(message: str, hint: str = "") -> dict[str, Any]:
    """A refusal the model can read, and act on. Never an exception.

    ``digest_tools.read_day_digest``'s shape: ``error`` plus an optional ``hint``, and
    ``hint`` omitted rather than empty when there is nothing to suggest.
    """
    payload: dict[str, Any] = {"error": _field(message, 400)}
    if hint:
        payload["hint"] = _field(hint, 400)
    return payload


def weather_forecast(
    place: str,
    date: str = "",
    days: int = 1,
    country: str = "",
    tool_context: Any = None,
) -> dict:
    """The weather forecast for a named place and day, as JSON.

    Use this for ANY question about the weather of a place -- temperature, rain, rain
    chance, humidity, wind, UV, how cold it will be, whether to take an umbrella. Never
    answer one from memory: a forecast is not in your training data, and a plausible
    guess here is indistinguishable from a real reading.

    **Resolve the place before you report anything.** The payload names the city, state
    and country it actually matched, plus its population. If that is not plainly the
    place the user meant -- a different city, a different country, or a very small
    settlement -- do not report numbers for it. Either call ask_user, or call this again
    with a more specific `place`: "Springfield, Illinois" resolves to the right one in
    a single call.

    If the payload has `candidates` instead of a forecast, the name matched several
    places and none clearly won. Do not silently take the first one. Either retry with
    the region added ("Valencia, Espana") or with `country` set ("Valencia" + `ES`),
    or call ask_user offering the candidates.

    Answer with **every** metric the payload carries for the day, each with its unit:
    max and min temperature, apparent temperature, humidity, rain chance, precipitation,
    wind and gusts, UV, sunshine hours, sunrise and sunset. The user asked for the whole
    picture, and a follow-up that should not have been needed is a turn wasted.

    **A question about *now* is answered by `current`, not by the daily row.** The daily
    figures are aggregates over the whole day, so their mean is not what the weather is
    doing at this moment -- report `current` for "agora", "right now", "at the moment",
    and for any question about a particular hour. `current` is absent when the day asked
    is not today, and then there is no reading of the present to report: say the forecast
    is for that date rather than substituting a daily average.

    `hourly` is the day asked about, one reading per hour, in the place's local time.
    Every array lines up with `hourly.time` by index. Use it for "when will it rain",
    "what will it be like tonight", and anything comparing two hours. Do not add these
    numbers up: the daily row already carries the totals.

    Do NOT save this to the second brain: a forecast goes stale, and a note holding one
    would be replayed verbatim by the vault cache tomorrow.

    Args:
        place: City or place name, as the user wrote it ("Osasco", "São Paulo"). Adding
            a region -- "Osasco, SP" -- resolves an ambiguous name in one call.
        date: "today", "tomorrow", "yesterday" (or "hoje", "amanha", "ontem") or
            YYYY-MM-DD. Defaults to today. Anything else is refused -- call
            current_datetime first for a date like "next Friday".
        days: How many consecutive days to return, from `date`, 1-14.
        country: Optional two-letter **country** code to disambiguate ("BR", "PT").
            Not a state: "IL" is Israel, not Illinois.
    """
    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    asked = _field(place, 120)
    if not asked:
        return _failure(
            "no place was given",
            "pass the city or place name the user asked about",
        )

    try:
        resolved = resolve_place(asked, country=country)
    except AmbiguousPlace as exc:
        return {
            "status": "ambiguous",
            "error": _field(str(exc), 400),
            "asked_for": asked,
            "candidates": [p.as_dict() for p in exc.candidates[:MAX_REPORTED_CANDIDATES]],
            "hint": (
                "either call ask_user offering these places, or call this tool again "
                "with the region added to `place` ('Springfield, Illinois') or with "
                "`country` set to the country you believe was meant"
            ),
        }
    except ValueError as exc:
        return _failure(
            str(exc),
            "spell the place as the user wrote it, or add the region to it",
        )

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return _failure("the weather service took too long to answer")
    try:
        first = start_date(date)
        rows, series, current, metadata, url = forecast(
            resolved, first, days, now=datetime.datetime.now(datetime.UTC)
        )
    except ValueError as exc:
        return _failure(str(exc))

    geocode_url = _geocoding_url().rstrip("/") + "?" + urlencode(
        {"name": asked, "count": 1, "format": "json", "language": _language()}
    )
    # Both URLs the model was shown, which is what makes either of them citable. The
    # forecast URL is the one to cite: it is the document the numbers came from, and it
    # carries the coordinates the answer was computed for.
    record_returned_urls(tool_context, [url, geocode_url], provider=provider_name())
    # Prefer the place's own timezone for this session, so general time questions
    # inherit it without the model guessing. The clock tool reads it from state.
    try:
        if tool_context is not None:
            state = getattr(tool_context, "state", None)
            tz = metadata.get("timezone") or resolved.timezone
            if tz and isinstance(tz, str) and tz.strip():
                if isinstance(state, dict):
                    state["user_timezone"] = tz.strip()
                else:
                    try:
                        state.set("user_timezone", tz.strip())
                    except Exception:
                        pass
    except Exception:
        pass
    # The chart is drawn from this, by ``sources.render_chart``, off the answer's own
    # callback. Recorded whether or not a tool_context exists, so the call degrades to
    # "no chart" rather than to an exception.
    record_hourly_series(tool_context, series, current, resolved.name)

    payload: dict[str, Any] = {
        "status": "ok",
        "asked_for": asked,
        "place": resolved.as_dict(),
        "forecast": rows,
        "units": dict(UNITS),
        "provider": provider_name(),
        "source_url": url,
        "geocoding_url": geocode_url,
        "timezone": metadata.get("timezone") or resolved.timezone,
        "utc_offset_seconds": metadata.get("utc_offset_seconds") or rows[0].get("utc_offset_seconds"),
        "elevation_m": metadata.get("elevation_m"),
        "provider_daily_units": metadata.get("daily_units"),
        "hint": (
            "report every metric above for the day asked, with its unit, and say which "
            "place it is for"
        ),
    }
    # Absent rather than empty, for the reason the ambiguous branch has no `forecast`
    # key: a key that exists but is empty is one the model can quote from without
    # reading the refusal beside it.
    if series:
        payload["hourly"] = series
    if current:
        payload["current"] = current
    return payload
