"""Unit tests for the MCP tool-schema sanitiser.

The bug these guard against is not theoretical: with the real ``obsidian-mcp``
server, ``search_metadata`` declares ``value`` as a ``type`` union that includes
``"array"`` but carries no ``items``. The primary Gemini path tolerates that, so
turns pass all day. The ``FallbackModel`` path routes the same schema through
LiteLLM to OpenRouter, which may hand it to a Google AI Studio upstream; that
upstream lowers the JSON Schema to a native ``Schema`` message, splitting the
union into ``any_of`` and then rejecting the array branch:

    GenerateContentRequest.tools[0].function_declarations[6]
        .parameters.properties[value].any_of[0].items: missing field

which surfaces as ``litellm.BadRequestError`` / HTTP 400. Because that only
happens on the *fallback* path, it fires precisely when the primary model has
already failed, so a transient quota error turns into a dead turn.

No MCP server or network is involved: the sanitiser is a pure function over a
JSON Schema, and the fixtures below are verbatim copies of the schemas
``obsidian-mcp`` actually serves.
"""

from __future__ import annotations

from text_summarizer.obsidian_tools import sanitize_tool_schema

# Verbatim from obsidian-mcp's list_tools(): a union type whose "array" member
# has no "items", which is the exact shape that produced the 400 above.
SEARCH_METADATA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "properties": {
        "type": {"description": "tag or frontmatter", "type": "string"},
        "value": {
            "default": None,
            "description": "Value to compare against.",
            "type": ["array", "boolean", "null", "number", "object", "string"],
        },
    },
    "required": ["type"],
    "type": "object",
}

# Verbatim from obsidian-mcp's search_text: an array parameter whose element type
# is a $ref into $defs. Gemini reports "reference to undefined schema" for a
# dangling ref, so the target has to be inlined before $defs goes away.
SEARCH_TEXT = {
    "$defs": {
        "SearchField": {
            "description": "Fields available for targeted full-text search.",
            "enum": ["title", "headings", "tags", "body", "frontmatter"],
            "type": "string",
        }
    },
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "properties": {
        "query": {"description": "Search query.", "type": "string"},
        "fields": {
            "default": None,
            "description": "Restrict search to specific note fields.",
            "items": {"$ref": "#/$defs/SearchField"},
            "type": ["array", "null"],
        },
    },
    "required": ["query"],
    "type": "object",
}


def _walk(node):
    """Yield every dict in a schema, so assertions cover nested subschemas too."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_a_multi_type_union_loses_its_array_member_and_gains_no_items():
    """The change, on the exact parameter that killed a turn.

    ``search_metadata.value`` used to be left as a six-way union carrying a
    top-level ``items``. OpenRouter's ModelRun provider refuses exactly that:

        parameter "value": more than one JSON reading of the same emitted value

    A top-level ``items`` on a parameter that is also a plain ``string`` gives it
    two valid JSON readings, which is what it is complaining about. Removing
    ``items`` is not the fix -- a Google AI Studio upstream then answers
    ``any_of[0].items: missing field``, the fault this sanitiser exists to prevent
    -- and neither is expressing the array branch as ``anyOf``, which the same
    provider also refuses.

    So the union is narrowed instead. Four of the five alternatives survive, and
    the array case is not a capability loss: the server compares a JSON-encoded
    string as a literal string, so an array was already only reachable quoted.
    """
    value = sanitize_tool_schema(SEARCH_METADATA)["properties"]["value"]

    assert value["type"] == ["boolean", "null", "number", "object", "string"]
    assert "array" not in value["type"]
    # The absence of `items` is the point, not an oversight: with no array member
    # there is no array branch for an `any_of` lowering to leave without one.
    assert "items" not in value


def test_a_nullable_array_parameter_still_gains_items():
    """The common case, and the one the original fix was for.

    ``["array", "null"]`` is how ``obsidian-mcp`` writes every optional array --
    there is exactly one non-null member, so ``items`` is unambiguous and every
    upstream wants it. This is the shape the narrowing must *not* touch, and it is
    asserted on the real ``search_text`` schema rather than a synthetic one.
    """
    fields = sanitize_tool_schema(SEARCH_TEXT)["properties"]["fields"]

    assert fields["type"] == ["array", "null"]
    # Spelled out rather than compared whole, so a change to the element enum is
    # visible here instead of buried in a dict diff.
    assert fields["items"]["type"] == "string"
    assert fields["items"]["enum"] == ["title", "headings", "tags", "body", "frontmatter"]


def test_a_plain_array_parameter_still_gains_items():
    """``{"type": "array"}`` with no ``items`` at all."""
    out = sanitize_tool_schema({"properties": {"tags": {"type": "array"}}})

    assert out["properties"]["tags"]["items"] == {"type": "string"}


def test_every_array_typed_subschema_has_items():
    """The invariant, restated after the narrowing.

    A parameter is expected to carry ``items`` exactly when it can *be* an array
    and nothing else -- so the check is on the sanitised type rather than on the
    original, which is what makes it catch a narrowing that forgot to remove
    ``items`` as well as an injection that never happened.
    """
    for schema in (SEARCH_METADATA, SEARCH_TEXT):
        for node in _walk(sanitize_tool_schema(schema)):
            declared = node.get("type")
            concrete = (
                [t for t in declared if t != "null"]
                if isinstance(declared, list)
                else ([declared] if declared else [])
            )
            if concrete == ["array"]:
                assert "items" in node, node
            else:
                assert "items" not in node, f"items on a non-array parameter: {node}"


def test_narrowing_is_idempotent():
    """Sanitising twice must reach the same schema as sanitising once.

    The narrowing rewrites ``type``, so a second pass sees a union it has already
    handled. Were the two passes to disagree, the rewritten schema would depend on
    how many times the toolset had been listed -- and ``get_tools`` runs on every
    turn.
    """
    once = sanitize_tool_schema(SEARCH_METADATA)
    twice = sanitize_tool_schema(once)

    assert twice == once
    assert sanitize_tool_schema(twice) == once


def test_refs_are_inlined_and_defs_removed():
    out = sanitize_tool_schema(SEARCH_TEXT)
    fields = out["properties"]["fields"]

    assert "$defs" not in out
    assert "$schema" not in out
    assert fields["items"]["enum"] == [
        "title",
        "headings",
        "tags",
        "body",
        "frontmatter",
    ]


def test_no_dangling_refs_remain():
    for schema in (SEARCH_METADATA, SEARCH_TEXT):
        for node in _walk(sanitize_tool_schema(schema)):
            assert "$ref" not in node


def test_sibling_keys_win_over_inlined_target():
    """A $ref's own siblings must not be lost when the target is expanded."""
    schema = {
        "$defs": {"Base": {"type": "string", "description": "from target"}},
        "type": "object",
        "properties": {
            "x": {"$ref": "#/$defs/Base", "description": "from sibling"},
        },
    }
    node = sanitize_tool_schema(schema)["properties"]["x"]

    assert node["type"] == "string"
    assert node["description"] == "from sibling"


def test_existing_items_are_preserved():
    """A schema that already declares element types must not be overwritten."""
    schema = {
        "type": "object",
        "properties": {
            "paths": {
                "items": {"type": "string"},
                "maxItems": 100,
                "type": ["array", "null"],
            }
        },
    }
    out = sanitize_tool_schema(schema)

    assert out["properties"]["paths"]["items"] == {"type": "string"}


def test_descriptions_and_required_survive():
    out = sanitize_tool_schema(SEARCH_METADATA)

    assert out["required"] == ["type"]
    assert out["properties"]["value"]["description"] == "Value to compare against."


def test_sanitizing_is_idempotent():
    """The toolset rewrites the raw schema in place, so re-running must be a no-op."""
    once = sanitize_tool_schema(SEARCH_METADATA)
    twice = sanitize_tool_schema(once)

    assert once == twice


def test_unknown_ref_is_left_alone_rather_than_dropped():
    """A server-side typo must not silently weaken the schema it typo'd."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/DoesNotExist"}},
    }
    node = sanitize_tool_schema(schema)["properties"]["x"]

    assert node == {"$ref": "#/$defs/DoesNotExist"}


def test_non_dict_input_is_returned_unchanged():
    assert sanitize_tool_schema(None) is None
    assert sanitize_tool_schema("not-a-schema") == "not-a-schema"


def test_self_referential_defs_terminate():
    """$defs can point at itself; expansion must be depth-capped, not infinite."""
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }

    out = sanitize_tool_schema(schema)  # must terminate

    assert out["properties"]["root"]["type"] == "object"
