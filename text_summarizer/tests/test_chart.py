"""Unit tests for the SVG chart itself -- the drawing, the store, and the name.

``tests/test_sources.py`` covers the *answer*: where the image goes in the text, what its
alt text says, and that ``summary_only`` removes it. This file covers the *image*: that
it is well-formed, that its geometry does not lie about the data, and that the store it is
written to is content-addressed and bounded.

Split that way because the two properties fail differently. A wrong answer is read by the
model's user; a malformed SVG is read by a browser as a blank box, with nothing logged and
no test in the other file able to see it.

No LLM, no network, no browser. ``render_svg`` is pure string-building, so every claim
here is checkable against the text it produces -- and well-formedness is checked by
*parsing* it with ``xml.etree`` rather than by asserting on a substring, because a broken
document and a document that renders wrongly look identical to a substring assertion.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import pytest
from text_summarizer import chart


#: Distinguishes "caller did not say" from "caller said: no such series". Passing
#: ``rain=None`` for the latter is impossible with a plain default, and getting it wrong
#: once added a second panel to every assertion that expected one -- which is the kind of
#: failure that makes a geometry test quietly test less than it reads as testing.
_NO_SERIES = object()


def _series(temps=None, rain=_NO_SERIES, hours=24, day="2026-10-05"):
    """An hourly block shaped like the live API's.

    A night near 16 °C rising to 22 °C at 13:00, and a rain chance climbing through the
    evening -- so the two panels have visibly different shapes, which is what makes the
    per-panel scaling testable rather than vacuous.
    """
    block = {
        "unit": "hour",
        "hours": hours,
        "time": [f"{day}T{hour:02d}:00" for hour in range(hours)],
        "temperature_c": temps if temps is not None else [16.1 + (h % 7) for h in range(hours)],
    }
    if rain is _NO_SERIES:
        block["rain_chance_pct"] = (
            [0] * (hours - 6) + [20, 30, 40, 44, 40, 30] if hours >= 6 else [0] * hours
        )
    elif rain is not None:
        block["rain_chance_pct"] = rain
    return block


NOW = {"time": "2026-10-05T16:00", "hour": "16:00", "temperature_c": 19.2, "rain_chance_pct": 24}


def _root(svg: str) -> ET.Element:
    """Parse the document, so well-formedness is asserted rather than assumed."""
    assert svg.startswith("<svg"), svg[:80]
    assert svg.rstrip().endswith("</svg>")
    return ET.fromstring(svg)


def _panel_geometry(svg: str) -> list[list[tuple[float, float]]]:
    """One list of (x, y) per ``polyline``, in document order."""
    points: list[list[tuple[float, float]]] = []
    for node in _root(svg).iter("{http://www.w3.org/2000/svg}polyline"):
        pairs = [tuple(float(v) for v in pair.split(",")) for pair in node.get("points").split()]
        points.append(pairs)
    return points


def _texts(svg: str) -> list[str]:
    """Every string drawn as ``<text>``, which is all of the chart's labels."""
    return ["".join(node.itertext()) for node in _root(svg).iter("{http://www.w3.org/2000/svg}text")]


# --- the document -------------------------------------------------------------


def test_the_chart_is_a_well_formed_svg_document_with_a_size():
    """The two properties an ``<img>`` cannot recover from.

    A browser handed a document it cannot parse renders nothing, and nothing else in the
    answer says so -- the alt text is there precisely because the image can vanish. So
    well-formedness is checked by parsing.

    And the size: an ``<img>`` with no intrinsic width is laid out at its container's
    width, which on the dev UI's ~800px column would scale a 720px drawing up. Fixed
    dimensions are what make it draw at the size it was drawn for.
    """
    svg = chart.render_svg(_series(), NOW, "Osasco")
    root = _root(svg)
    # The parser consumes the declaration, so the namespace is only observable as the
    # element's own tag. Asserting on the literal string in `svg` as well would be
    # true of a document that declared it and was then rendered as something else.
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    width = int(root.get("width"))
    height = int(root.get("height"))
    assert root.get("viewBox") == f"0 0 {width} {height}"
    assert width <= 800, "wider than the message column it is rendered in, so it scales down"
    assert height > 0


def test_the_image_carries_its_own_background_because_an_img_cannot_inherit_one():
    """An ``<img>`` is an isolated document and cannot read the page's colours.

    That is the whole reason the palette is fixed rather than inherited: the dev UI's dark
    theme would otherwise show black ink on a transparent background -- and since the
    drawing is the *only* thing that fails, the answer would read as complete.
    """
    root = _root(chart.render_svg(_series(), NOW, "Osasco"))
    fills = [node.get("fill") for node in root.iter("{http://www.w3.org/2000/svg}rect")]
    assert fills and fills[0], "the first rect must paint a background over the whole frame"
    assert chart._CARD_BG in fills[0]


# --- geometry -----------------------------------------------------------------


def test_each_series_is_scaled_to_its_own_range():
    """Temperature near 19 °C and rain chance near 0 % share nothing numerically.

    On one shared scale the rain panel flattens into a straight line along the bottom --
    a graph saying "it will not rain" on a day with a 44 % evening chance, drawn by the
    code whose whole job is not to say that. Stacked panels with independent ranges are
    the fix; the assertion is that the two panels' tick labels do not overlap as sets.
    """
    svg = chart.render_svg(_series(), NOW, "Osasco")
    panels = _panel_geometry(svg)
    assert len(panels) == 2, "expected one polyline per series"

    # The rain panel spans 0..44 and the temperature panel 16.1..22.1, so at least one
    # gridline label of each panel is a value the other cannot contain.
    labels = _texts(svg)
    assert "44" in labels and "16.1" in labels
    assert "0" in labels and "22.1" in labels


def test_a_flat_day_is_drawn_flat_rather_than_as_a_full_range_ramp():
    """A constant day must look constant.

    A zero-span series scaled across a whole panel would draw a wild-looking graph out of
    a day where nothing changed -- noise presented as a trend, which is the specific way a
    chart lies. The fix is a branch, not a tolerance: a span of 0 is widened by 1 °C, so
    every point lands on the same y.
    """
    svg = chart.render_svg(_series(temps=[19.2] * 24, rain=None), None, "Osasco")
    panels = _panel_geometry(svg)
    assert len(panels) == 1, "the rain series is absent, so there is one panel"
    ys = {y for _x, y in panels[0]}
    assert len(ys) == 1, f"a flat day drew {len(ys)} distinct heights: {sorted(ys)}"


def test_the_line_is_step_after_and_never_a_spline():
    """Steps, because an hourly forecast is a value *held* for the hour.

    A smooth spline through hourly readings invents intermediate extrema that were never
    measured: it draws a peak between two hours at both of which the temperature fell. The
    assertion is not "it looks jagged" -- it is that consecutive readings sit on a
    horizontal run, so a value is held until the hour changes.
    """
    svg = chart.render_svg(_series(temps=[16.0, 20.0, 16.0] + [16.0] * 21, rain=None), None)
    points = _panel_geometry(svg)[0]
    horizontal = sum(1 for (x1, y1), (x2, y2) in zip(points, points[1:]) if y1 == y2 and x1 != x2)
    assert horizontal >= 3, f"{horizontal} horizontal runs in {len(points)} points"


def test_a_missing_reading_breaks_the_line_rather_than_being_bridged():
    """The provider returns ``null`` for hours it cannot serve.

    Bridging those would draw a temperature nobody reported, and on a curve the lowest
    stretch is exactly the shape a reader would believe -- a cold dip that is really just
    an absence. So a gap ends the run, and the two sides are separate polylines.
    """
    # Three hours, a two-hour gap, three hours: two runs of real length, so the assertion
    # is about the *bridge* and not about a lone dot (which is drawn, as a circle).
    temps = [16.0, 17.0, 18.0, None, None, 20.0, 21.0, 22.0]
    svg = chart.render_svg(_series(temps=temps, rain=None, hours=8), None)
    runs = _panel_geometry(svg)
    assert len(runs) == 2, f"the gap did not split the line: {len(runs)} run(s)"

    first, second = runs
    assert max(x for x, _y in first) == pytest.approx(chart._x(2, 8), abs=0.6)
    assert min(x for x, _y in second) == pytest.approx(chart._x(5, 8), abs=0.6)

    # The property, stated as the thing a reader would notice: no drawn segment spans
    # either missing column. A bridged gap is exactly that.
    missing = [chart._x(i, 8) for i in (3, 4)]
    for run in runs:
        for (x1, _y1), (x2, _y2) in zip(run, run[1:]):
            assert not (min(x1, x2) < missing[0] and max(x1, x2) > missing[1]), run


def test_a_single_isolated_reading_is_a_dot_rather_than_nothing():
    """One point with no extent would render as an empty panel.

    ``polyline`` with a single coordinate draws nothing at all, so a series that is one
    reading beside two gaps needs a shape with area: a ``circle``. A ``null`` cell for
    every other hour is what makes this reachable in practice.
    """
    temps: list[float | None] = [None] * 8
    temps[4] = 21.5
    svg = chart.render_svg(_series(temps=temps, rain=None, hours=8), None)
    root = _root(svg)
    circles = list(root.iter("{http://www.w3.org/2000/svg}circle"))
    assert len(circles) == 1, f"{len(circles)} dots for one reading"
    assert not list(root.iter("{http://www.w3.org/2000/svg}polyline")), "a lone point drew a line"
    assert float(circles[0].get("cx")) == pytest.approx(chart._x(4, 8), abs=0.1)


def test_the_two_panels_share_one_x_axis():
    """Stacked panels only line up if every panel computes the same column.

    This is the entire visual argument for stacking them rather than overlaying two scales
    on one plot, and it is the reason ``_x`` is one function: a per-panel inline
    expression would agree today and drift the first time a panel was given a different
    column count.
    """
    svg = chart.render_svg(_series(), NOW, "Osasco")
    panels = _panel_geometry(svg)
    assert len(panels) == 2
    left_temperature = min(x for x, _y in panels[0])
    left_rain = min(x for x, _y in panels[1])
    assert left_temperature == left_rain == pytest.approx(chart._PAD_LEFT, abs=0.6)

    # Both panels have the frame at the same x and the same width.
    frames = [
        node.get("x")
        for node in _root(svg).iter("{http://www.w3.org/2000/svg}rect")
        if node.get("stroke")
    ]
    assert len(set(frames)) == 1, f"the panels do not share a left edge: {frames}"


def test_a_truncated_series_draws_an_axis_only_as_long_as_the_data():
    """Timestamps are validated before they become an axis.

    The series is whatever the provider sent, and a payload cut short must not leave an
    axis running past the last drawn point -- a reader counts the hours on the axis and
    compares them with the curve, and the two disagreeing is the chart failing at its only
    job.
    """
    block = _series(hours=8)
    block["time"].append("not-a-timestamp")
    block["temperature_c"].append(99.0)
    svg = chart.render_svg(block, None)
    labels = _texts(svg)
    assert "07:00" in labels, labels
    assert not any(label.endswith(":00") and int(label[:2]) >= 8 for label in labels), labels


def test_axis_numbers_use_a_period_because_the_image_is_not_the_readers_locale():
    """``19.2``, not ``19,2``.

    A Portuguese answer shows ``19,2`` in the prose; a comma on the axis reads as two
    different numbers. There is no right answer without knowing the answer's language,
    and the axis is the least-bad place to be locale-blind.
    """
    svg = chart.render_svg(_series(temps=[19.25] * 24, rain=None), None)
    labels = _texts(svg)
    assert "19.2" in labels or "19.25" in labels, labels
    assert not any("," in label for label in labels), labels


# --- the "now" annotation ------------------------------------------------------


def test_now_is_a_dashed_rule_on_the_first_panel_only():
    """Once, because two would read as two annotations.

    A dashed vertical rule is affordable here in a way a caret was not in the ASCII
    version: the hour is a real x-coordinate, so it cannot drift with the reader's
    monospace face. ``tools/scenarios.py`` asserts the *other* channel -- the alt text --
    because a text consumer never sees the drawing.
    """
    svg = chart.render_svg(_series(), NOW, "Osasco")
    root = _root(svg)
    dashed = [node for node in root.iter("{http://www.w3.org/2000/svg}line") if node.get("stroke-dasharray")]
    assert len(dashed) == 1
    assert float(dashed[0].get("x1")) == pytest.approx(chart._x(16, 24), abs=0.6)
    assert any(label.startswith("now 16:00") for label in _texts(svg))


def test_the_hour_label_finds_its_column_when_the_full_timestamp_is_absent():
    """The fallback joins the label to the series' day -- it does not append a suffix.

    ``"16:00" + ":00"`` is ``"16:00:00"``, which is in no series, so that branch never
    fired and the ``now`` marker vanished whenever only the label was present: no error,
    no chart, simply no annotation. Found by the first test run of this change, and the
    test is here rather than in ``test_weather.py`` because it is a property of the
    drawing, not of which place the tool resolved.
    """
    by_label = chart.render_svg(_series(), {"hour": "16:00"}, "Osasco")
    by_time = chart.render_svg(_series(), NOW, "Osasco")
    assert "now 16:00" in _texts(by_label)
    assert _panel_geometry(by_label) == _panel_geometry(by_time)

    # And across midnight, where the label is the only way to find the column.
    overnight = chart.render_svg(
        _series(day="2026-10-05"), {"hour": "03:00"}, "Osasco"
    )
    assert "now 03:00" in _texts(overnight)


def test_a_reading_outside_the_series_is_not_marked():
    """No column, no rule -- the other half of the same test.

    A rule drawn at a clamped or nearest column would put the annotation on a reading the
    provider did not serve, which is the same class of failure as plotting a missing
    reading as the minimum.
    """
    for absent in (
        None,
        "not a dict",
        {},
        {"time": "1999-01-01T00:00", "hour": "00:00"},
        {"time": 17, "hour": 17},
        {"hour": ""},
    ):
        svg = chart.render_svg(_series(), absent, "Osasco")
        assert "stroke-dasharray" not in svg, absent
        assert not any(label.startswith("now") for label in _texts(svg)), absent


# --- hostile and malformed input -----------------------------------------------


def test_a_hostile_place_name_is_escaped_rather_than_parsed():
    """The place is third-party text from a gazetteer, so this is not decoration.

    An unescaped ``<`` in the title does not merely look wrong: it starts a tag the XML
    parser never closes, and ``<img>`` hands the document to a real renderer. The
    assertion is that the document still parses and that no element named after the
    payload exists in it -- checking for the absence of the literal text would pass
    against an escaped ``&lt;`` too, which is not the same claim.
    """
    svg = chart.render_svg(
        _series(),
        None,
        '</text></svg><script>alert("x")</script><text>&"\'<>&',
    )
    root = _root(svg)
    assert not list(root.iter("script")), "the payload became an element"
    # The name survived as text, escaped.
    assert "&lt;" in svg and "<script" not in svg
    assert any("alert" in label for label in _texts(svg)), "the name was dropped instead of escaped"


def test_there_is_nothing_to_draw_rather_than_a_something():
    """One branch for "no chart", so the caller has one decision to make.

    Every one of these is a payload the code cannot draw: no times, no well-formed
    timestamps, no series with a single reading, or not a mapping at all. Returning an
    empty string is what lets ``sources.render_chart`` drop the placeholder rather than
    emit an empty image.
    """
    for absent in (
        None,
        "a string",
        {},
        {"time": []},
        {"time": None},
        {"time": ["garbage", "more garbage"]},
        {"time": ["2026-10-05T00:00"]},
        {"time": _series()["time"]},
        {"time": _series()["time"], "temperature_c": [None] * 24, "rain_chance_pct": [None] * 24},
        {"time": _series()["time"], "temperature_c": "not a list", "rain_chance_pct": 7},
    ):
        assert chart.render_svg(absent) == "", absent


def test_a_series_that_is_shorter_than_its_own_axis_is_padded_not_drawn_short():
    """Parallel arrays are the payload's contract; a short one is still drawable.

    ``weather`` builds parallel arrays and caps them at ``MAX_HOURLY_HOURS``, so they
    agree in production. A hand-written payload that does not is rendered over its own
    timestamps -- drawing fewer points than hours would silently drop the tail of a
    forecast.
    """
    block = _series(hours=6, rain=None)
    block["temperature_c"] = block["temperature_c"][:4]
    runs = _panel_geometry(chart.render_svg(block, None))
    assert len(runs) == 1, f"the tail is a gap, not a second run: {len(runs)}"

    # The property: the fourth reading sits on the *fourth* column. Scaling the four
    # readings across the six-hour axis would put it on the sixth -- and an axis a reader
    # can count says the last point is the last hour.
    assert max(x for x, _y in runs[0]) == pytest.approx(chart._x(3, 6), abs=0.6), runs[0][-2:]


def test_a_non_string_place_is_dropped_rather_than_printed():
    """A title is either a place's name or nothing.

    ``render_svg`` is public, so f-stringing its ``place`` unguarded would print
    ``Hourly - 17`` -- which reads as a place rather than as a bug.
    """
    for absent in (None, 17, 0.5, ["Osasco"], {"name": "Osasco"}, (1, 2), b"Osasco"):
        labels = _texts(chart.render_svg(_series(), NOW, absent))
        assert any(label == "Hourly - 2026-10-05" for label in labels), absent
        assert "None" not in " ".join(labels), absent

    # The empty string and a blank one are the same "no place", not a stray separator.
    for blank in ("", "   ", "\n", "\t"):
        labels = _texts(chart.render_svg(_series(), NOW, blank))
        assert any(label == "Hourly - 2026-10-05" for label in labels), repr(blank)

    # A real name is kept, and a *stripped* one: the payload's own whitespace is not the
    # title's business.
    assert any("Hourly - Osasco" in label for label in _texts(chart.render_svg(_series(), NOW, "  Osasco ")))


# --- the name, which is also the store's only input -----------------------------


def test_the_name_is_a_digest_of_the_inputs_and_changes_when_any_of_them_changes():
    """Content-addressing, which is what makes rendering idempotent.

    ``sources.render_sources`` runs over an already-rendered answer, so a chart's URL must
    be reproducible from the payload or the second pass emits a *different* image than
    the first -- and the whole rendered block stops being byte-stable, which the linked-
    ``Sources`` re-parse depends on.

    Every input is varied, because a digest that ignored one of them would pass a test
    that only checks two.
    """
    base = chart.svg_fingerprint(_series(), NOW, "Osasco")
    assert re.fullmatch(r"[0-9a-f]{16}", base), base
    assert chart.svg_fingerprint(_series(), NOW, "Osasco") == base, "not a pure function"

    variants = {
        "hourly": chart.svg_fingerprint(_series(temps=[20.0] * 24), NOW, "Osasco"),
        "current": chart.svg_fingerprint(_series(), dict(NOW, temperature_c=21.4), "Osasco"),
        "place": chart.svg_fingerprint(_series(), NOW, "Londres"),
        "width": chart.svg_fingerprint(_series(), NOW, "Osasco", width=800),
    }
    for label, other in variants.items():
        assert other != base, f"the name ignores {label}"

    # And it is a digest of *values*, not of their order in the payload: a provider that
    # serialises its keys differently must not produce a second file for one forecast.
    reordered = {key: _series()[key] for key in reversed(list(_series()))}
    assert chart.svg_fingerprint(reordered, NOW, "Osasco") == base


def test_the_name_coerces_rather_than_crashes_on_a_payload_that_is_not_a_mapping():
    """Both reasons for the guards are visible in one assertion each.

    A non-mapping is hashed as empty rather than raising -- the caller is
    ``render_chart``, which is best-effort by construction. And a non-string place is
    coerced the *same way* ``render_svg`` coerces it, or the two would disagree about
    which chart a given payload is and the name would name a file nobody wrote.
    """
    assert chart.svg_fingerprint(None, None, None) == chart.svg_fingerprint({}, {}, "")
    assert chart.svg_fingerprint(_series(), NOW, 17) == chart.svg_fingerprint(_series(), NOW, None)
    assert chart.svg_fingerprint(_series(), "not a dict") == chart.svg_fingerprint(_series(), None)


def test_chart_href_accepts_only_a_name_this_module_could_have_produced():
    """The href goes into an answer, so an unvalidated one is a path the browser resolves.

    Not a hypothetical: the name is a path *segment* in the URL the dev UI's browser then
    requests, and ``..`` segments are exactly what a traversal attempt looks like from the
    server's side. Returning ``None`` rather than raising means the caller emits no image
    at all -- which is the safe outcome for an enhancement.
    """
    assert chart.chart_href("0123456789abcdef") == "/chart/0123456789abcdef.svg"

    for bad in (
        "../etc/passwd",
        "..",
        "../../../etc/passwd",
        "/etc/passwd",
        "0123456789ABCDEF",
        "0123456789abcde",
        "0123456789abcdefg",
        "0123456789abcdef.svg",
        "0123456789abcde/",
        "",
        " ",
        "0123456789abcdef\n",
        "\n0123456789abcdef",
        None,
        17,
        ["0123456789abcdef"],
    ):
        assert chart.chart_href(bad) is None, bad


# --- the store ------------------------------------------------------------------


def test_storing_writes_the_bytes_and_returns_the_href(tmp_path, monkeypatch):
    monkeypatch.setenv(chart.CHART_DIR_ENV, str(tmp_path / "charts"))
    href = chart.store_svg(chart.render_svg(_series(), NOW, "Osasco"), "0123456789abcdef")
    assert href == "/chart/0123456789abcdef.svg"
    written = tmp_path / "charts" / "0123456789abcdef.svg"
    assert written.is_file()
    assert _root(written.read_text(encoding="utf-8")) is not None
    # No temporary file survives the rename.
    assert [p.name for p in written.parent.iterdir()] == ["0123456789abcdef.svg"]


def test_storing_is_a_no_op_rather_than_a_failure_on_every_way_it_cannot_proceed(
    tmp_path, monkeypatch
):
    """A chart is an enhancement; a turn that dies over one is a catastrophic trade.

    Each of these is a way ``store_svg`` can be unable to write, and every one of them
    must come back as ``None`` so the caller drops the placeholder instead of emitting an
    image whose file is not there -- which renders as a broken-image icon, not as an error.
    """
    monkeypatch.setenv(chart.CHART_DIR_ENV, str(tmp_path / "charts"))
    svg = chart.render_svg(_series(), NOW, "Osasco")

    assert chart.store_svg("", "0123456789abcdef") is None, "an empty document"
    assert chart.store_svg(svg, "../evil") is None, "a name that is not a digest"
    assert chart.store_svg(svg, "") is None, "no name"

    # A directory the process cannot create.
    unwritable = tmp_path / "locked"
    unwritable.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(chart.CHART_DIR_ENV, str(unwritable / "charts"))
    assert chart.store_svg(svg, "0123456789abcdef") is None, "an unwritable directory"
    assert not (unwritable / "charts").exists(), "it was created anyway"


def test_a_reader_never_sees_a_half_written_image(tmp_path, monkeypatch):
    """Write-then-rename, for the reason ``second_brain._write`` does it.

    The temporary file is a **sibling** of the target: ``os.replace`` is only atomic
    *within* a filesystem, so staging in ``TMPDIR`` would silently degrade the guarantee
    to a non-atomic copy -- and a reader that fetched the URL mid-write would be served a
    truncated SVG, which renders as a blank image rather than as an error.
    """
    store = tmp_path / "charts"
    monkeypatch.setenv(chart.CHART_DIR_ENV, str(store))
    seen: list[str] = []
    real_replace = os.replace

    chart.store_svg("<svg/>", "0123456789abcdef")
    assert (store / "0123456789abcdef.svg").read_text(encoding="utf-8") == "<svg/>"

    def spy(src, dst, **kwargs):
        # Mid-write, the target holds its *previous* contents -- never a partial new
        # document. That is the property, asserted from the writer's own side.
        seen.extend(sorted(p.name for p in store.iterdir()))
        assert str(src).endswith(".tmp"), src
        assert str(dst).endswith(".svg"), dst
        assert (store / "0123456789abcdef.svg").read_text(encoding="utf-8") == "<svg/>"
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(chart.os, "replace", spy)
    chart.store_svg(chart.render_svg(_series(), NOW, "Osasco"), "fedcba9876543210")
    assert seen, "the rename never happened, so nothing was observed"
    assert not list(store.glob("*.tmp")), "a temporary file was left behind"


def test_pruning_drops_the_oldest_and_never_the_chart_the_turn_just_drew(tmp_path, monkeypatch):
    """Bounded retention, and the ordering is the load-bearing half.

    ``store_svg`` prunes *before* it writes, so the file being written this turn is never
    a candidate. Pruning after would delete, on a full store, the image a message that is
    on screen right now points at -- a blank image under a correct answer.
    """
    store = tmp_path / "charts"
    monkeypatch.setenv(chart.CHART_DIR_ENV, str(store))
    store.mkdir()

    names = [f"{i:016x}" for i in range(5)]
    for index, name in enumerate(names):
        target = store / f"{name}.svg"
        target.write_text(f"<svg id='{index}'/>", encoding="utf-8")
        os.utime(target, (1_700_000_000 + index, 1_700_000_000 + index))

    # Not a chart: a chart-shaped name that is not one, and an unrelated file. Neither may
    # be deleted -- the store is only ever pruned by a writer that computed the names, so
    # a stray file the operator put there is theirs.
    (store / "notes.svg").write_text("mine", encoding="utf-8")
    (store / "README").write_text("mine", encoding="utf-8")

    chart._prune(str(store), keep=2)
    left = sorted(p.name for p in store.iterdir())
    assert left == ["0000000000000003.svg", "0000000000000004.svg", "README", "notes.svg"], left

    # And the newest file survives even when the store is exactly at the limit.
    chart._prune(str(store), keep=2)
    assert (store / "0000000000000004.svg").is_file()


def test_pruning_takes_the_limit_from_the_module_constant_at_call_time(tmp_path):
    """A default argument binds once, at definition.

    ``_prune(keep=CHART_MAX_FILES)`` would make the retention permanently 200 for every
    deployment -- no environment can change it and no test can reach it without writing
    two hundred files. Asserted through the constant rather than by reading the signature,
    because that is the property a future edit would take away.
    """
    import inspect

    source = inspect.getsource(chart._prune)
    assert "CHART_MAX_FILES if keep is None" in source, (
        "the limit is bound in the signature, where nothing can change it"
    )

    store = tmp_path / "c"
    store.mkdir()
    for index in range(3):
        (store / f"{index:016x}.svg").write_text("<svg/>", encoding="utf-8")
    monkey = chart.CHART_MAX_FILES
    try:
        chart.CHART_MAX_FILES = 1
        chart._prune(str(store))
        assert len(list(store.glob("*.svg"))) == 1, "the constant did not take effect"
    finally:
        chart.CHART_MAX_FILES = monkey


def test_pruning_a_directory_that_is_not_there_is_not_an_error(tmp_path):
    """``_prune`` runs before the write, on a directory that may not exist yet."""
    chart._prune(str(tmp_path / "absent"))


def test_chart_dir_is_the_adk_directory_by_default_and_the_override_wins(tmp_path, monkeypatch):
    """``.adk/charts`` is inside the git-ignored state ``adk web`` already writes to.

    Nothing new to clean up and nothing new to mount. And a deployment that keeps ``.adk``
    on another volume needs the override, so the resolution is one place.
    """
    monkeypatch.delenv(chart.CHART_DIR_ENV, raising=False)
    assert chart.chart_dir() == os.path.join(os.getcwd(), ".adk", "charts")

    monkeypatch.setenv(chart.CHART_DIR_ENV, f"  {tmp_path / 'elsewhere'}  ")
    assert chart.chart_dir() == str(tmp_path / "elsewhere"), "the value is not stripped or absolutised"


def test_a_blank_override_falls_back_rather_than_writing_to_the_cwd(tmp_path, monkeypatch):
    """An empty ``CHART_DIR`` is a misconfiguration, and the safe reading is the default.

    Taking it literally would resolve ``.adk/charts`` *relative to a stray empty string*,
    which is the default anyway -- so the interesting half is that it must not become
    ``/charts`` or the root of the filesystem.
    """
    monkeypatch.setenv(chart.CHART_DIR_ENV, "   ")
    assert chart.chart_dir() == os.path.join(os.getcwd(), ".adk", "charts")
