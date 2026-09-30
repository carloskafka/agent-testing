# The gate. `make check` is what CI runs and what you should run before you push:
# format-check, lint, types, test, in that order, so the cheapest and most
# mechanical failure comes first.
#
# There is no root pyproject.toml -- text_summarizer/ is the only Python project --
# so every tool is invoked either with `uv run --project text_summarizer` or with
# that project's own interpreter, and every path is written relative to it.
#
# No target here needs PYTHONPATH, and none of them assume you have run `uv sync`:
# `pick` below resolves each tool to something that works right now.

.DEFAULT_GOAL := check

VENV    := text_summarizer/.venv
VENV_PY := $(VENV)/bin/python
PKG     := text_summarizer
TESTS   := $(PKG)/tests

# $(call pick,<module to import>,<command name>)
#
# Resolve a dev tool to a runnable command, cheapest first:
#   1. the project venv's interpreter, if it can already import the tool -- no
#      network, no lockfile write, instant;
#   2. `uv run --project text_summarizer <tool>`, which is where the
#      `[dependency-groups] dev` group lands, and the runner CI uses;
#   3. bare `python3 -m <tool>`, so a machine with neither uv nor a built venv gets
#      a real error from the tool rather than "command not found".
#
# The probe is a real import rather than a `-x` test: a `pytest` entry point on
# PATH that cannot import its own package is the failure mode step 1 rules out, and
# it is precisely the state this tree is in today.
#
# `uv` installs the dev group on demand, so a fresh clone needs no `uv sync` first.
UV_BIN := $(shell command -v uv 2>/dev/null || ls -1 $(HOME)/.local/bin/uv 2>/dev/null)

# TEST_ARGS lets CI add flags without a second target -- `make test
# TEST_ARGS="--junitxml=report.xml --cov=text_summarizer"` -- so the three steps CI
# runs are these three targets and not a parallel set of hand-written commands that
# can drift from what a developer runs.
TEST_ARGS ?=

pick = $(shell \
	if [ -x '$(VENV_PY)' ] && '$(VENV_PY)' -c 'import $(1)' >/dev/null 2>&1; then \
		echo '$(VENV_PY) -m $(2)'; \
	elif [ -n '$(UV_BIN)' ]; then \
		echo '$(UV_BIN) run --project $(PKG) $(2)'; \
	else \
		echo 'python3 -m $(2)'; \
	fi)

RUFF    := $(call pick,ruff,ruff)
MYPY    := $(call pick,mypy,mypy)
PYTEST  := $(call pick,pytest,pytest)

# The agent is not a dev tool, it is the project: the venv console script is the
# right default and `uv run` the fallback. Same shape as `pick`, but probed by
# path because `adk` is an entry point rather than an importable module.
ADK := $(shell \
	if [ -x '$(VENV)/bin/adk' ]; then echo '$(VENV)/bin/adk'; \
	elif [ -n '$(UV_BIN)' ]; then echo '$(UV_BIN) run --project $(PKG) adk'; \
	else echo 'adk'; fi)

.PHONY: check fmt format-check lint types test coverage run eval clean help

## check: the whole gate, in the order a reviewer wants to read it
check: format-check lint types test

## fmt: rewrite the tree with `ruff format`
fmt:
	$(RUFF) format $(PKG)

## format-check: the first step of `check`. Not gating yet -- see the body.
format-check:
	@echo "==> format-check"
	@echo "    NOT GATING. 'ruff format --check' is not run yet on purpose: the tree"
	@echo "    has never been through the formatter, and enabling it now would demand a"
	@echo "    ~6,700-line mechanical reindent in a commit whose job is the gate."
	@echo "    'make fmt' is wired up and runnable; the check joins 'check' in the same"
	@echo "    commit that adopts the formatted tree. Until then this step is a"
	@echo "    documented no-op rather than a permanently red gate."

## lint: pyflakes, pycodestyle, import order, bugbear, pyupgrade, simplifications
lint:
	@echo "==> lint (ruff check)"
	$(RUFF) check $(PKG)

## types: mypy, non-strict (see the overrides in pyproject.toml)
types:
	@echo "==> types (mypy)"
	$(MYPY) --config-file $(PKG)/pyproject.toml

## test: the suite. No API cost, no vault writes.
test:
	@echo "==> test (pytest)"
	$(PYTEST) $(TESTS) -q $(TEST_ARGS)

## coverage: the same suite under coverage, writing .coverage and a term report
coverage:
	@echo "==> coverage"
	$(PYTEST) $(TESTS) -q \
		$(TEST_ARGS) \
		--cov=$(PKG) \
		--cov-report=term-missing:skip-covered \
		--cov-report=xml

## run: the ADK dev UI on http://127.0.0.1:8000
run:
	$(ADK) web $(PKG)

## eval: the eval set, against the real model
#
# CACHE_ENABLED=false is mandatory, not a default. `adk eval` calls the real agent,
# which calls save_summary_to_second_brain against the real vault, so the first run
# writes a note per case; every later run on the same set then hits the cache, skips
# the model *and* the tools, and reports a high response_match_score while testing
# nothing. See AGENTS.md gotcha 10.
EVAL_SET    ?= $(PKG)/tests/eval/summarizer_eval_set.evalset.json
EVAL_CONFIG ?= $(PKG)/tests/eval/test_config.json

eval:
	CACHE_ENABLED=false $(ADK) eval $(PKG) $(EVAL_SET) \
		--config_file_path $(EVAL_CONFIG) --print_detailed_results

## clean: generated caches only
##
## Deliberately does not touch text_summarizer/.venv (expensive to rebuild and it is
## what `pick` probes), .adk/, optimization_history.json, or anything in the vault --
## `adk eval` writes into the real vault, so a `clean` that reached into it would
## delete notes, not artefacts.
clean:
	find . -path './$(VENV)' -prune -o -name '__pycache__' -type d -exec rm -rf {} +
	find . -path './$(VENV)' -prune -o -name '*.pyc' -type f -exec rm -f {} +
	rm -rf .pytest_cache $(PKG)/.pytest_cache .ruff_cache $(PKG)/.ruff_cache \
	       .mypy_cache $(PKG)/.mypy_cache .coverage coverage.xml pytest-report.xml
	@echo "cleaned caches; the venv, .adk/ and the vault are untouched"

## help: list the targets
help:
	@grep -E '^## [a-z-]+:' $(MAKEFILE_LIST) | sed 's/^## //'
