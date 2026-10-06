"""The hourly curve, as an SVG image served from ``/chart``.

Why this module exists
----------------------
The first version of the graph was a fenced block of block-drawing characters
(``▂▃▄▅``). It rendered correctly -- widths verified equal, the strip marker was
the fence's own info string -- and it was still *weird* to look at, for three
reasons no amount of glyph-picking fixes:

* **The glyphs are East-Asian-*ambiguous* width.** ``U+2581``–``U+2588`` are
  drawn at a single cell width by some fonts and at double width by others, so a
  reader whose monospace face disagrees with the writer's gets the columns
  drifting instead of a curve. Alignment by counting characters assumes a
  property the renderer does not control and cannot verify.
* **There is no axis.** A range printed as ``16.1..22.9`` asks the reader to do
  the interpolation the chart should be doing, and each row was scaled to fill
  its own range, so a day that moved 1 °C looked as dramatic as one that moved
  15 °C. Honest, and unreadable.
* **The answer is text that a browser renders as a phone book.** Every consumer
  of this response is a text consumer except one: the dev UI.

So the curve moved to SVG. Everything else about it is unchanged on purpose --
still drawn in code, never asked of the model -- because the reasons for drawing
it in code were never about glyph choice.

**SVG, not PNG, and that is measurable rather than aesthetic.** Rasterising needs
a font rasteriser and this container has **zero TTF files** installed, which is
the whole reason the first version was not a PNG. SVG carries its own geometry
and lets the *browser* pick a font, so it needs neither a font nor a rendering
dependency: **no new dependency at all**, which is a standing requirement of
this deployment.

**Why not a charting library -- measured, not assumed.** The obvious candidate
is Google Charts, and it cannot work here. The dev UI renders assistant text
through ngx-markdown into Angular's ``SecurityContext.HTML`` sanitizer, which
strips ``<script>`` unconditionally (verified in the bundled ``main-*.js``:
``DEFAULT_SECURITY_CONTEXT=eg.HTML``). Google Charts is a JavaScript library
loaded via ``<script src=".../gstatic.com/charts/loader.js">``; the tag is
removed, the loader never runs, and the answer shows an empty div with nothing
logged anywhere. The same applies to Chart.js or any other JS charting runtime,
and Mermaid -- which *is* bundled in that same file -- is disabled
(``mermaid:!1``) and draws diagrams, not time series. What the sanitizer *does*
permit is a plain ``<img src>``: the bundle's own policy table carries
``["IMG", Map([["src", ...RESOURCE_URL_POLICY]])]``. Hence an image, hence a
route, hence this module.

**One store, content-addressed.** The file name is a digest of the payload, not
a random token, and that is load-bearing twice over: the same forecast renders
to the same URL, so :func:`text_summarizer.sources.render_sources` stays
**idempotent** (a second pass over an already-rendered answer reproduces it byte
for byte, which the linked-``**Sources**`` re-parse depends on); and two turns
that share a forecast share one file instead of writing two.

**Nothing here can end a turn.** Every failure -- an unwritable directory, a
malformed payload, an absent ``current`` -- is caught at the boundary and
reported as "no chart", which the caller turns into a dropped placeholder. The
numbers the reader needs are in the prose regardless; a missing graph is a
worse answer, never a broken one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

__all__ = [
    "CHART_WEB_PREFIX",
    "CHART_MAX_FILES",
    "chart_dir",
    "chart_href",
    "render_svg",
    "store_svg",
    "svg_fingerprint",
]

#: Where the images are served from, and -- deliberately -- the same mount style
#: as ``sources.VAULT_WEB_PREFIX``. ``StaticFiles`` rejects paths that escape its
#: directory, so the traversal defence is the mount's and is already proven by
#: ``tests/test_serve.py``; :func:`store_svg` is the second line, and it writes
#: only names this module computed.
CHART_WEB_PREFIX = "/chart"

#: Read-only over HTTPS for a file nobody ever mutates.
CHART_MEDIA_TYPE = "image/svg+xml"

#: How many charts to keep. One weather turn writes one file, so this is roughly
#: a week of hourly forecasts; it exists because a long-lived container would
#: otherwise accumulate them forever, not because the directory is large.
CHART_MAX_FILES = 200

#: Environment override, for a deployment that keeps ``.adk`` on another volume.
CHART_DIR_ENV = "CHART_DIR"

#: The extension, and the only one. A name is a hex digest, so this is also the
#: guard that decides what ``StaticFiles`` may serve.
CHART_SUFFIX = ".svg"

_NAME_RE = re.compile(r"^[0-9a-f]{16}$")

# --- geometry ---------------------------------------------------------------
# Sized for the dev UI's ~800px message column with room for the y-axis labels.
# Fixed rather than responsive because an <img> has no viewport to measure, and
# the browser scales it down on a phone.
_WIDTH = 720
_PANEL_HEIGHT = 118
_PANEL_GAP = 26
_PAD_LEFT = 52
_PAD_RIGHT = 16
_PAD_TOP = 54
_PAD_BOTTOM = 30
_HEIGHT = _PAD_TOP + _PANEL_HEIGHT * 2 + _PANEL_GAP + _PAD_BOTTOM

# --- palette ----------------------------------------------------------------
# Fixed rather than inherited: an <img> is an isolated document and cannot read
# the page's colours. So the chart carries its own background, which is what
# makes it legible in the dev UI's dark theme instead of a black-on-black hole.
_INK = "#1c1b1f"
_MUTED = "#5f5b66"
_GRID = "#e3e0e6"
_FRAME = "#d5d1da"
_PANEL_BG = "#ffffff"
_CARD_BG = "#fbfafc"

#: The two series, and their colours. Kept apart from ``sources._CURVE_SERIES``
#: deliberately: that tuple keys the *payload*, this one keys the *drawing*, and
#: a third pair of glyph glyph constants in the same module would be the second
#: source of truth for the same idea.
_TEMPERATURE_COLOR = "#c2410c"
_RAIN_COLOR = "#1d6f9c"
_NOW_COLOR = "#7c3aed"


def chart_dir() -> str:
    """The directory charts are written to.

    ``.adk/charts`` -- inside the git-ignored ``.adk`` that ``adk web`` already
    writes its session database into, so this adds no new state location and
    nothing new to clean up. Resolved against the process working directory,
    which is what ``serve.py`` hands to ``adk web`` as its agents dir.
    """
    configured = os.environ.get(CHART_DIR_ENV, "").strip()
    if configured:
        return os.path.abspath(configured)
    return os.path.join(os.getcwd(), ".adk", "charts")


def _values(hourly: Any, key: str, count: int) -> list[float | None]:
    """One series as floats, ``None`` for a reading the provider did not serve.

    ``None`` stays ``None`` all the way to the drawing, where it breaks the line
    rather than being plotted as a number: interpolating across a gap would
    invent a reading, and dropping the point silently would misstate how many
    hours the panel covers.

    **Padded to ``count``, never truncated to what arrived.** The panel's columns
    come from ``len(values)`` while its hour labels come from ``count``, so a short
    column silently *stretched* the readings to fill the axis -- four readings across
    six hours drew a line ending on the sixth hour's column, and a reader counting
    hours on the axis would read the fourth reading as the sixth. Padding keeps every
    reading on the hour it was recorded for, and turns the shortfall into the gap it
    is.
    """
    column = hourly.get(key) if isinstance(hourly, dict) else None
    if not isinstance(column, list):
        return []
    out: list[float | None] = []
    for value in column[:count]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append(None)
        else:
            out.append(float(value))
    if len(out) < count:
        out.extend([None] * (count - len(out)))
    return out


def svg_fingerprint(
    hourly: Any,
    current: Any = None,
    place: Any = "",
    width: int = _WIDTH,
) -> str:
    """The 16-hex-character name for this exact chart.

    A digest of the *inputs*, not a random token, and both reasons matter. A
    token would make :func:`text_summarizer.sources.render_sources`
    non-idempotent -- a second pass would emit a different URL for an answer
    that already carries one -- and would make two turns describing the same
    forecast write two identical files. Width is included because it is a
    rendering input: a chart redrawn at a different size is a different image.
    """
    payload = {
        "hourly": hourly if isinstance(hourly, dict) else {},
        "current": current if isinstance(current, dict) else {},
        "place": place if isinstance(place, str) else "",
        "width": width,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chart_href(name: str) -> str | None:
    """The ``/chart`` URL for a name this module computed, else ``None``.

    Returning ``None`` for anything unexpected is the whole reason this exists
    rather than an f-string at the call site: the href goes into an answer, and
    an href built from an unvalidated name is a path the reader's browser will
    happily resolve somewhere else.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return None
    return f"{CHART_WEB_PREFIX}/{name}{CHART_SUFFIX}"


def _esc(text: str) -> str:
    """Escape for both XML and an HTML ``<img>`` consumer.

    ``&`` first, or the ampersands introduced by the later rules get escaped
    twice. The place name is a gazetteer string -- third-party text -- so this is
    not defensive decoration.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _number(value: float) -> str:
    """A short, locale-free number for an axis tick.

    A period, deliberately, and the reason is in the module docstring's sibling:
    this is an image, so it is not the reader's locale -- a Portuguese answer
    would show ``19,2`` in the prose and ``19.2`` on the axis, which reads as two
    different numbers. There is no right answer without knowing the answer's
    language, and the axis is the least-bad place to be locale-blind.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Evenly spaced tick values across a panel's range.

    ``low``/``high`` are the *data* extremes, so the curve always touches both
    ends and the tick that coincides with each extreme lands exactly on the
    curve. Rounded to the step so a tick reads as a round number.
    """
    if high <= low:
        return [low]
    step = (high - low) / count
    return [low + step * i for i in range(count + 1)]


def _position_of(stamps: list[str], current: Any) -> int | None:
    """Which column of the series is the reading of *now*, or ``None``.

    ``current`` carries both a full ``time`` and an ``hour`` label, and they are **not**
    interchangeable: ``time`` is the provider's own stamp and ``hour`` is only its last
    five characters. So a ``time`` that is present is authoritative *and exclusive* -- if
    it is not in the series the reading is outside the window and nothing is marked.
    Falling back to the label in that case would find ``00:00`` on the *wrong day* and
    put the annotation there, which is the same class of failure as plotting a missing
    reading as the minimum: a confident mark on a column nobody reported.

    The label is the fallback for when ``time`` is absent, and it has to be **joined to
    the series' own day** rather than suffixed -- ``"16:00" + ":00"`` is ``"16:00:00"``,
    which is in no series, so that branch silently never fired and every ``now`` marker
    vanished whenever only the label was present. Found by the first test run against
    this change, and it is the same shape as the "silent" half of most defects recorded
    in this repository: nothing errors, the annotation is simply gone.
    """
    if not isinstance(current, dict):
        return None
    stamp = current.get("time")
    if isinstance(stamp, str):
        return stamps.index(stamp) if stamp in stamps else None
    hour = current.get("hour")
    if isinstance(hour, str):
        for day in dict.fromkeys(value[:10] for value in stamps):
            candidate = f"{day}T{hour}"
            if candidate in stamps:
                return stamps.index(candidate)
    return None


def _x(index: int, count: int) -> float:
    """Column centre for hour ``index``, shared by every panel.

    One function rather than an inline expression repeated per panel, because the
    panels only line up if they agree -- which is the entire visual argument for
    stacking them instead of overlaying two scales on one plot.
    """
    if count <= 1:
        return _PAD_LEFT + (_WIDTH - _PAD_LEFT - _PAD_RIGHT) / 2
    span = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    return _PAD_LEFT + (span * index) / (count - 1)


def _panel(
    y_top: float,
    values: list[float | None],
    color: str,
    unit: str,
    label: str,
) -> list[str]:
    """One panel: frame, gridlines with real numbers, and the line itself.

    A **step-after** line, not a smooth spline. A spline through hourly readings
    invents intermediate extrema that were never measured -- it will draw a
    peak between two hours at both of which the temperature fell -- and a chart
    that shows a temperature nobody recorded is worse than no chart. Steps are
    also the honest reading of an hourly *forecast*, which is a value held for
    the hour.
    """
    numbers = [v for v in values if v is not None]
    if not numbers:
        return []
    low, high = min(numbers), max(numbers)
    if high == low:
        high = low + 1.0

    out: list[str] = []
    plot_h = _PANEL_HEIGHT - 22  # room for the tick labels under the frame

    def y_of(value: float) -> float:
        return y_top + plot_h - ((value - low) / (high - low)) * plot_h

    # Gridlines and their values. Behind the line.
    for tick in _ticks(low, high):
        y = y_of(tick)
        out.append(
            f'<line x1="{_PAD_LEFT}" y1="{y:.1f}" x2="{_WIDTH - _PAD_RIGHT}" '
            f'y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{_PAD_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{_esc(_number(tick))}</text>'
        )

    out.append(
        f'<rect x="{_PAD_LEFT}" y="{y_top:.1f}" width="{_WIDTH - _PAD_LEFT - _PAD_RIGHT}" '
        f'height="{plot_h:.1f}" fill="none" stroke="{_FRAME}" stroke-width="1"/>'
    )

    # The line, as step-after runs. A gap in the readings ends a run rather than
    # being bridged: the reader must be able to see which hours the provider did
    # not serve, and a bridged gap invents a temperature nobody reported.
    previous: float | None = None
    segments: list[list[tuple[float, float]]] = []
    for index, value in enumerate(values):
        x = _x(index, len(values))
        if value is None:
            previous = None
            continue
        y = y_of(value)
        if previous is None:
            segments.append([(x, y)])
        else:
            step_x = _x(index - 1, len(values))
            segments[-1].append((step_x, y))
            segments[-1].append((x, y))
        previous = y
    for segment in segments:
        if len(segment) < 2:
            # A single isolated reading is a tick, not a line: one point with no
            # extent would render as nothing at all.
            (x, y) = segment[0]
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}"/>'
            )
            continue
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in segment)
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )

    # Panel label and unit, so the axis numbers are never ambiguous.
    out.append(
        f'<text x="{_PAD_LEFT}" y="{y_top - 9:.1f}" class="label">{_esc(label)}</text>'
    )
    out.append(
        f'<text x="{_WIDTH - _PAD_RIGHT}" y="{y_top - 9:.1f}" text-anchor="end" '
        f'class="tick">{_esc(unit)}</text>'
    )
    return out


def render_svg(
    hourly: Any,
    current: Any = None,
    place: Any = "",
    width: int = _WIDTH,
) -> str:
    """The chart as a standalone SVG document.

    Returns ``""`` when there is nothing to draw -- no times, or no series with a
    single reading -- so the caller has one branch for "no chart" rather than
    two.
    """
    if not isinstance(hourly, dict):
        return ""
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        return ""
    # Only keep the prefix whose timestamps are well formed: a truncated series
    # must not have an axis longer than the data drawn on it.
    stamps: list[str] = []
    for value in times:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value):
            break
        stamps.append(value)
    if not stamps:
        return ""
    count = len(stamps)

    temperature = _values(hourly, "temperature_c", count)
    rain = _values(hourly, "rain_chance_pct", count)

    panels: list[tuple[str, list[float | None], str, str, str]] = []
    if any(v is not None for v in temperature):
        panels.append(("Temperature", temperature, _TEMPERATURE_COLOR, "°C", "temperature"))
    if any(v is not None for v in rain):
        panels.append(("Rain chance", rain, _RAIN_COLOR, "%", "chance of rain"))
    if not panels:
        return ""

    height = _PAD_TOP + _PANEL_HEIGHT * len(panels) + _PANEL_GAP * (len(panels) - 1) + _PAD_BOTTOM

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">',
        "<style>",
        f".t{{fill:{_INK};font-size:12px}}",
        f".tick{{fill:{_MUTED};font-size:11px}}",
        f".label{{fill:{_INK};font-size:12px;font-weight:600}}",
        f".title{{fill:{_INK};font-size:13px;font-weight:600}}",
        f".sub{{fill:{_MUTED};font-size:11px}}",
        f".now{{fill:{_NOW_COLOR};font-size:11px;font-weight:600}}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="{_CARD_BG}"/>',
    ]

    name = place.strip() if isinstance(place, str) else ""
    day = stamps[0][:10]
    title = f"Hourly - {name}, {day}" if name else f"Hourly - {day}"
    out.append(f'<text x="{_PAD_LEFT - 36}" y="16" class="title">{_esc(title)}</text>')
    out.append(
        f'<text x="{width - 16}" y="16" text-anchor="end" class="sub">'
        f"{_esc('local time')}</text>"
    )

    for index, (_label, values, color, unit, _slug) in enumerate(panels):
        y_top = _PAD_TOP + index * (_PANEL_HEIGHT + _PANEL_GAP)
        out.extend(_panel(y_top, values, color, unit, _label))

        # Hour axis under the last panel only, so two panels do not print the
        # same hours twice.
        if index == len(panels) - 1:
            base = y_top + _PANEL_HEIGHT - 22
            step = max(1, count // 12)
            for hour in range(0, count, step):
                out.append(
                    f'<text x="{_x(hour, count):.1f}" y="{base + 15:.1f}" '
                    f'text-anchor="middle" class="tick">{stamps[hour][-5:]}</text>'
                )

        # The "now" rule, drawn only on the first panel so it reads as one
        # annotation across the chart rather than two.
        if index == 0 and isinstance(current, dict):
            position = _position_of(stamps, current)
            if position is not None:
                x = _x(position, count)
                bottom = y_top + _PANEL_HEIGHT - 22
                out.append(
                    f'<line x1="{x:.1f}" y1="{y_top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                    f'stroke="{_NOW_COLOR}" stroke-width="1.5" stroke-dasharray="3 3"/>'
                )
                out.append(
                    f'<text x="{x + 5:.1f}" y="{y_top + 12:.1f}" class="now">'
                    f"now {_esc(str(stamps[position][-5:]))}</text>"
                )

    out.append("</svg>")
    return "\n".join(out)


def _prune(directory: str, keep: int | None = None) -> None:
    """Drop the oldest charts past ``keep``.

    Sorted by modification time, and it runs *before* the new file is written so
    the chart this turn just drew is never a candidate for deletion -- a message
    that is on screen right now must not have its image pulled out from under it.

    ``keep`` defaults to :data:`CHART_MAX_FILES` **read at call time**, not bound as a
    default argument. A default is evaluated once, at definition, so pinning it there
    would make the retention untunable by a deployment that wants a different one --
    and untestable without writing two hundred files.
    """
    limit = CHART_MAX_FILES if keep is None else keep
    try:
        entries = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.endswith(CHART_SUFFIX) and _NAME_RE.fullmatch(name[: -len(CHART_SUFFIX)])
        ]
    except OSError:
        return
    if len(entries) <= limit:
        return
    try:
        entries.sort(key=lambda path: os.path.getmtime(path))
    except OSError:
        return
    for path in entries[: len(entries) - limit]:
        try:
            os.remove(path)
        except OSError:
            pass


def store_svg(svg: str, name: str) -> str | None:
    """Write ``svg`` under ``name`` and return its href, or ``None``.

    Write-then-rename, for the same reason ``second_brain._write`` does it: a
    reader that fetches the URL while the write is in progress would otherwise
    be served a truncated SVG, which renders as a blank image rather than as an
    error -- and the temporary file is a sibling of the target because
    ``os.replace`` is only atomic *within* a filesystem.

    Returns ``None`` rather than raising on any failure. A chart is an
    enhancement; a turn that dies because a directory was read-only is a
    catastrophic trade for it.
    """
    href = chart_href(name)
    if not svg or href is None:
        return None
    directory = chart_dir()
    target = os.path.join(directory, name + CHART_SUFFIX)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None

    # Second line of defence behind the mount's own traversal check: the only
    # names that may be written are hex digests this module computed.
    if os.path.dirname(os.path.abspath(target)) != os.path.abspath(directory):
        return None

    _prune(directory)
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(svg)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    return href