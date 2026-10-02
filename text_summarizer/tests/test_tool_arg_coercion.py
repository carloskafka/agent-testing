"""Unit tests for MCP tool-argument coercion.

Live failure, session ``2fe0d9d0-a73b-4d9b-bc13-452658c2d585``. Asked *"what's the
São Paulo weather in Celsius degree?"*, the agent called ``search_text`` with:

    {'query': 'São Paulo weather', 'max_results': '5', 'context_length': '50'}

Both numbers arrived **in quotes**, and ``obsidian-mcp`` is Rust:

    failed to deserialize parameters: invalid type: string "50", expected usize

The turn then died on an unrelated fault -- the next model call went back to Gemini
with that unsigned ``functionCall`` in the history and was rejected with a 400 --
so the type error never got a retry. It is worth fixing on its own account anyway:
it costs a round trip, and it is the kind of thing that ends a turn the moment the
other bug is fixed.

The shape that matters is the *union*. ``obsidian-mcp`` declares every optional
parameter as ``["integer", "null"]``, so a parser that only understood
``{"type": "integer"}`` would coerce nothing at all, and the schemas below are
verbatim copies of what that server serves.

No MCP server, network or model is involved: coercion is a pure function over a
JSON Schema and an argument dict.
"""

from __future__ import annotations

import asyncio

from text_summarizer.obsidian_tools import (
    _coercing_run_async,
    coerce_tool_args,
    sanitize_tool_schema,
)

#: Verbatim from obsidian-mcp's search_text, minus the descriptions. Every optional
#: parameter is a nullable union, which is the shape that makes this worth a test.
SEARCH_TEXT = {
    "properties": {
        "query": {"description": "Text to search for.", "type": "string"},
        "max_results": {"default": None, "type": ["integer", "null"]},
        "context_length": {"default": None, "type": ["integer", "null"]},
        "fuzzy": {"default": None, "type": ["boolean", "null"]},
        "fields": {
            "default": None,
            "type": ["array", "null"],
            "items": {"enum": ["title", "body"], "type": "string"},
        },
    },
    "type": "object",
}


# --- the live failure ----------------------------------------------------------


def test_the_live_call_is_repaired():
    """The exact arguments from the failed turn, coerced to what the server declared.

    Asserted as one call because the interesting property is that *every* quoted
    number is fixed, not that a single one can be: a fix that handled
    ``max_results`` and missed ``context_length`` would pass a one-field test and
    return the same live error.
    """
    args = {"query": "São Paulo weather", "max_results": "5", "context_length": "50"}

    coerced = coerce_tool_args(SEARCH_TEXT, args)

    assert coerced == {"query": "São Paulo weather", "max_results": 5, "context_length": 50}
    assert isinstance(coerced["max_results"], int)
    assert isinstance(coerced["context_length"], int)


def test_the_original_dict_is_not_mutated():
    """The caller may still want the raw arguments for a log line.

    Also a quiet trap: mutating in place would mean the *model's* record of what it
    called is rewritten, so a trace would show a call the model never made.
    """
    args = {"max_results": "5"}

    coerce_tool_args(SEARCH_TEXT, args)

    assert args == {"max_results": "5"}


# --- what must not be touched --------------------------------------------------


def test_an_already_correct_argument_is_unchanged():
    """The overwhelmingly common case: Gemini sends real integers.

    If this ever started failing, the coercion would be rewriting valid calls --
    which is a far worse failure than the one it prevents, because it would be
    invisible on exactly the turns that work today.
    """
    args = {"query": "dogs", "max_results": 5, "context_length": 100, "fuzzy": False}

    assert coerce_tool_args(SEARCH_TEXT, args) == args


def test_a_string_parameter_keeps_a_numeric_looking_string():
    """``{"type": "string"}`` is a string, even when the value is ``"50"``.

    Without this, a free-tier model searching for the literal text "50" would have
    that argument silently turned into the number 50 and searched for something
    else -- a wrong answer with nothing in the trace to say why.
    """
    args = {"query": "50"}

    assert coerce_tool_args(SEARCH_TEXT, args) == {"query": "50"}


def test_a_union_that_still_allows_string_is_left_alone():
    """``["string", "integer", "null"]`` genuinely permits ``"50"``.

    A parameter declared this way is exactly the case where the server has told us
    a quoted number is acceptable, so parsing it would be second-guessing the
    schema rather than helping it. ``search_metadata.value`` is the real example:
    its description says a JSON-encoded string is compared as a literal string.
    """
    schema = {"properties": {"value": {"type": ["string", "integer", "null"]}}}

    assert coerce_tool_args(schema, {"value": "50"}) == {"value": "50"}


def test_an_unparseable_number_is_left_for_the_server_to_reject():
    """Turning ``"fifty"`` into ``0`` would be a silent wrong answer.

    The alternative -- coercing what we can and inventing what we cannot -- makes
    the failure land somewhere it can no longer be seen. Left alone, the server's
    error names the argument, which is the whole point of it being strict.
    """
    assert coerce_tool_args(SEARCH_TEXT, {"max_results": "many"}) == {"max_results": "many"}


def test_an_empty_string_is_not_zero():
    """``int("")`` raises, so this is covered -- pinned because it is the boundary.

    A parser written as ``value or default`` would turn ``""`` into the default,
    which for ``max_results`` means "ten results" for a query the model asked to
    return none of.
    """
    assert coerce_tool_args(SEARCH_TEXT, {"max_results": ""}) == {"max_results": ""}


# --- booleans, arrays, objects, nesting ----------------------------------------


def test_booleans_quoted_by_the_model_are_parsed():
    """``"true"`` is what a model that treats every scalar as a string produces."""
    assert coerce_tool_args(SEARCH_TEXT, {"fuzzy": "true"}) == {"fuzzy": True}
    assert coerce_tool_args(SEARCH_TEXT, {"fuzzy": "False"}) == {"fuzzy": False}


def test_a_quoted_boolean_under_an_integer_parameter_is_not_guessed():
    """``"true"`` is not an integer, and the declared type is the only evidence.

    Left as a string so the server rejects it -- rather than parsed as ``1``
    because some other branch would have accepted it.
    """
    assert coerce_tool_args(SEARCH_TEXT, {"max_results": "true"}) == {"max_results": "true"}


def test_a_number_declared_as_float_keeps_its_fraction():
    """``"1.5"`` under a number is a float; under an integer it is not one."""
    schema = {"properties": {"ratio": {"type": ["number", "null"]}, "count": {"type": ["integer", "null"]}}}

    coerced = coerce_tool_args(schema, {"ratio": "1.5", "count": "2"})

    assert coerced["ratio"] == 1.5
    assert isinstance(coerced["ratio"], float)
    assert coerced["count"] == 2 and isinstance(coerced["count"], int)


def test_array_elements_are_coerced_through_items():
    """An array of integers arrives as an array of quoted integers."""
    schema = {"properties": {"ids": {"type": ["array", "null"], "items": {"type": "integer"}}}}

    assert coerce_tool_args(schema, {"ids": ["1", "2", "3"]}) == {"ids": [1, 2, 3]}


def test_array_elements_under_a_string_items_type_are_untouched():
    """The array parameter in the real schema holds strings (``title``, ``body``)."""
    args = {"fields": ["1", "2"]}

    assert coerce_tool_args(SEARCH_TEXT, args) == {"fields": ["1", "2"]}


def test_a_nested_object_is_coerced_against_its_own_properties():
    """``note_create.frontmatter`` is an object whose values are user-chosen.

    Recursion is what makes the rule apply to a real schema rather than only to the
    flat top level, and it stops at the declared type: a value under a parameter
    with no subschema is untouched.
    """
    schema = {
        "properties": {
            "frontmatter": {
                "type": ["object", "null"],
                "properties": {"year": {"type": "integer"}, "note": {"type": "string"}},
            }
        }
    }

    assert coerce_tool_args(schema, {"frontmatter": {"year": "2026", "note": "2026"}}) == {
        "frontmatter": {"year": 2026, "note": "2026"}
    }


# --- shapes that are not ours to touch -----------------------------------------


def test_a_schema_with_no_properties_leaves_args_alone():
    """A tool that declares no parameters must not have its arguments rewritten."""
    assert coerce_tool_args({"type": "object"}, {"anything": "1"}) == {"anything": "1"}


def test_a_non_dict_argument_is_returned_unchanged():
    """Defensive: ADK hands this whatever the ``functionCall`` carried.

    A malformed call that is not an object has nothing to coerce against, and
    raising here would turn a recoverable bad call into a dead turn.
    """
    assert coerce_tool_args(SEARCH_TEXT, "not-a-dict") == "not-a-dict"
    assert coerce_tool_args(SEARCH_TEXT, None) is None


def test_an_argument_the_schema_does_not_declare_is_kept_as_is():
    """ADK does not validate arguments against the schema before dispatch.

    So a parameter the model invented arrives here, and dropping it would silently
    change the call the server sees. Kept verbatim, which lets the server be the one
    to complain.
    """
    args = {"query": "dogs", "limit": "5"}

    assert coerce_tool_args(SEARCH_TEXT, args) == args


# --- the wrapper that puts it on the tool --------------------------------------


class _RecordingTool:
    """A stand-in for an ``McpTool``: records the args it was finally called with."""

    def __init__(self):
        self.seen: list[dict] = []

    async def run_async(self, *, args, tool_context):
        self.seen.append(args)
        return "ok"

    # The wrapper uses functools.wraps, which reads __name__ off the original.
    run_async.__name__ = "run_async"


def test_the_wrapper_coerces_before_dispatch():
    """The integration point, and the reason the function is not dead code.

    Asserted on what the *original* received, because a wrapper that coerced and
    then called the original with the original would satisfy any check that only
    looked at the return value.
    """
    tool = _RecordingTool()
    wrapped = _coercing_run_async(tool.run_async, SEARCH_TEXT)

    asyncio.run(wrapped(args={"max_results": "5"}, tool_context=object()))

    assert tool.seen == [{"max_results": 5}]


def test_the_wrapper_passes_a_correct_call_straight_through():
    """The same call the agent makes on every ordinary turn, unchanged.

    Compared by value, not identity: ``coerce_tool_args`` always builds a new dict,
    so pinning identity would assert an optimisation rather than a property. What
    matters is that no *value* changed -- and the type assertions are what catch a
    coercion that turned ``5`` into ``"5"`` or ``False`` into ``0``.
    """
    tool = _RecordingTool()
    wrapped = _coercing_run_async(tool.run_async, SEARCH_TEXT)
    args = {"query": "dogs", "max_results": 5, "fuzzy": False}

    asyncio.run(wrapped(args=args, tool_context=object()))

    assert tool.seen == [args]
    assert isinstance(tool.seen[0]["max_results"], int)
    assert tool.seen[0]["fuzzy"] is False


def test_the_wrapper_forwards_tool_context_untouched():
    """``tool_context`` is framework-supplied and carries state the tool needs.

    Dropping it, or passing it positionally, would break every MCP tool that reads
    it -- and ``save_summary_to_second_brain`` reads the live model from exactly
    that object.
    """
    seen = {}

    async def original(*, args, tool_context):
        seen["ctx"] = tool_context
        return "ok"

    wrapped = _coercing_run_async(original, SEARCH_TEXT)
    marker = object()

    asyncio.run(wrapped(args={}, tool_context=marker))

    assert seen["ctx"] is marker


def test_a_positional_call_is_carried_through_without_coercing():
    """ADK calls tools by keyword, so there is no positional form to handle.

    Pinned as a *decision* rather than an accident, and the reason is which failure
    each behaviour produces. Coercing positionally means guessing the signature;
    guess wrong and the wrapper raises ``TypeError`` inside tool dispatch, where the
    traceback blames the schema and mentions nothing about coercion -- a dead turn
    from a defensive-looking feature. Skipping the coercion instead degrades to what
    the server already does with a quoted number: one loud, retryable rejection.
    """
    tool = _RecordingTool()
    wrapped = _coercing_run_async(tool.run_async, SEARCH_TEXT)

    # The stub is keyword-only, so this is what ADK would have to change to.
    marker = object()
    captured: list = []

    async def positional(*args, **kwargs):
        captured.append(args)
        return "ok"

    wrapped_positional = _coercing_run_async(positional, SEARCH_TEXT)
    asyncio.run(wrapped_positional({"max_results": "5"}, marker))

    # Carried through intact, args un-coerced -- the degradation, asserted.
    assert captured == [({"max_results": "5"}, marker)]
    assert captured[0][0]["max_results"] == "5"

    # And the keyword form -- the one ADK uses -- still coerces.
    asyncio.run(wrapped(args={"max_results": "5"}, tool_context=object()))
    assert tool.seen == [{"max_results": 5}]


def test_the_wrapper_keeps_the_original_name():
    """``functools.wraps``, asserted because a lost ``__name__`` shows up as an
    anonymous tool in the dev UI's debug trace and nowhere else."""
    tool = _RecordingTool()

    assert _coercing_run_async(tool.run_async, SEARCH_TEXT).__name__ == "run_async"


# --- it reads the sanitised schema, not the raw one ----------------------------


def test_coercion_is_driven_by_the_sanitised_schema():
    """The two run in sequence, and the order is load-bearing.

    ``sanitize_tool_schema`` rewrites ``items`` and inlines ``$ref``; coercion reads
    whatever schema it is handed. Wired the other way round it would coerce against
    the server's original ``$ref``s, which name types the wrapper has already
    replaced, and quietly fix nothing.

    The observable difference: the array parameter's element type only exists
    *after* sanitising, because before it is a dangling ``$ref``.
    """
    raw = {
        "properties": {
            "fields": {
                "type": ["array", "null"],
                "items": {"$ref": "#/$defs/SearchField"},
            },
            "max_results": {"type": ["integer", "null"]},
        },
        "$defs": {"SearchField": {"type": "integer"}},
    }
    sanitised = sanitize_tool_schema(raw)

    # Un-sanitised: `fields` elements are behind a $ref, and coercing them would
    # have to guess. Sanitised: the type is right there.
    assert coerce_tool_args(raw, {"fields": ["1"], "max_results": "5"}) == {
        "fields": ["1"],
        "max_results": 5,
    }
    assert coerce_tool_args(sanitised, {"fields": ["1"], "max_results": "5"}) == {
        "fields": [1],
        "max_results": 5,
    }
