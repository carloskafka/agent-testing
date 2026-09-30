# PLAN — 19 independent MVPs for `text_summarizer`

> **Status:** proposal. Nothing here is implemented.
>
> **Every MVP below ships on its own.** There are no dependencies between them — no MVP
> waits for another, and any single one can be built, merged, reviewed and reverted in
> isolation. Where two touch the same file, that is called out as *file contention*, not a
> dependency: they can still be built in either order, one after the other.
>
> **Produced by:** a full read of the repository (3,088 lines of agent source, 3,679 lines of
> tests, 3,190 of tooling, 902 of docs), a live `pytest` run, a live probe of the note
> writer, and a static sweep for documentation drift. Claims marked **[verified]** were
> executed, not inferred.
>
> **How to read it:** §1–3 are the assessment, the evidence and the rules. **§4 is the
> catalogue** — one block per MVP with what ships, how it is proven, and its cost. §5 is the
> target layout. §6 is the independence audit. §7 is the approval table. §8 is what I did
> not verify.

---

## §1 — Assessment

| Layer | Size | State |
|---|---|---|
| `text_summarizer/*.py` (12 modules) | 3,088 lines | Mature, heavily documented, **no DI, no linter, no types, no CI** |
| `text_summarizer/tests/` (10 modules) | 3,679 lines | **219 tests, all passing** in 6.4 s — the strongest asset here |
| `tools/` + shell/Docker infra | 3,190 lines | Excellent, 0 % tested, coupled to one machine |
| `docs/*.md` | 902 lines | Good prose, **~20 verified drift items** against the code |

**Keep — do not disturb:**
- The **`**Sources**` provenance design** — the model emits a *shape* with two sentinel
  tokens, code substitutes the real vault + served model, the renderer is idempotent and
  cannot produce a duplicate block. Deterministic, testable, backend-agnostic.
- The **vault-resolution contract** — mount the *parent* never the vault; refuse rather than
  guess; mirrored in POSIX shell and covered by one pytest module.
- The **tool-schema sanitiser** — fixes a real Gemini 400 that only fires on the fallback
  path, and is shared by both MCP toolsets so a third server cannot regress it.
- The **cache-key invariant** and the four ADK facts pinned by `test_adk_wiring.py` against
  a real `InMemoryRunner` at zero API cost.

**The three structural problems:**

1. **No automated gate.** `.github/workflows/` contains only `pages.yml`. `AGENTS.md:429`
   calls the pytest run "the only automated gate" — it is not a gate, and `pytest` is not
   even a declared dependency (it is installed out-of-tree into `/tmp`). 219 tests that
   nobody runs are 219 tests that will rot.
2. **Configuration resolved at *import* time into module globals.** `MODEL_PROVIDER`,
   `CACHE_ENABLED`, `VAULT_ROOT` and both tool lists are module constants. That is the
   direct cause of `tests/conftest.py` having to mutate `os.environ` at module scope,
   *before the first package import*, with a 15-line comment explaining why. That file is a
   smell detector, not a fixture.
3. **Callbacks reach into ADK by `getattr` shotgun.** `agent.py` and `second_brain.py`
   between them probe ~9 attributes off `CallbackContext` / `ToolContext` / `Event` with
   defensive `try/except`. `test_agent_callback.py` therefore builds `SimpleNamespace` doubles
   carrying only the attributes the code happens to read — the tests are coupled to
   attribute *access*, not to a contract.

---

## §2 — Findings

### 2.1 Defects — reproduced where marked

| # | Sev | Finding | Evidence |
|---|---|---|---|
| B1 | **High** | **Path traversal in `save_summary_to_second_brain`.** `topics` is split on `,`, stripped, and interpolated straight into `os.path.join(VAULT_ROOT, TOPICS_DIR, f"{topic}.md")` with **no sanitiser**. A model-supplied topic escapes the vault. | **[verified]** topic `../../../../tmp/sec/PWNED` created `/tmp/sec/PWNED.md` |
| B2 | **High** | **Frontmatter injection.** Unquoted interpolation of `tags` and `aliases` lets a newline in a model-supplied `title` close the YAML block and inject keys. | **[verified]** produced a note whose frontmatter is structurally broken |
| B3 | Med | `_slug()` strips punctuation but not `..`, and does not bound separator depth. | code read |
| B4 | Med | `second_brain.py:315` — `if note_title not in index_body or index_body:` is true whenever `index_body` is non-empty, so `created` is always `[note_path]`. Dead branch. | code read |
| B5 | Med | **`run.sh` accepts one choice past the advertised range, then dies silently.** With *N* vaults the prompt says `Choice [1-(N+2)]` but validation rejects only `> N+3`; choosing `N+3` falls through both handlers, `sed -n "Np"` yields empty, exit 1 **with no message**. | traced `run.sh:183-243` |
| B6 | Low | **`run.sh --help` prints five lines of shell source** — `sed -n '2,25p'`, but the comment block ends at line 20. | **[verified]** |
| B7 | Low | `run.sh:176` offers "import it by symlink" in a path that only implements copy — and copy is correct, since a symlink dangles inside the container. The message teaches the wrong thing. | code read |
| B8 | Low | **The published walkthrough teaches the bug it was fixed for.** `tools/films.py:128` renders `after_agent_callback → sources.render_sources`; `AGENTS.md` gotcha 16 and `agent.py:287-311` say it *cannot* live there. | code read |
| B9 | Low | **Non-atomic writes / lost updates.** `_write` truncates then writes; both `save_summary_to_second_brain` and `log_conversation` do read-modify-write on the shared index. `AGENTS.md` gap 9. | code read |
| B10 | Low | **`auto_optimize.py` optimises against a proxy that destroys the graded behaviour.** The prompt asks for "under 300 words"; the live block is ~490 words / 11 rules. Trimming to 300 likely deletes rules 7–11, which `tool_trajectory_avg_score` grades at threshold **1.0**, while ROUGE-1 *improves* and the loop accepts it. | `auto_optimize.py:112-122` vs `agent.py:460-473` + `tests/eval/test_config.json` |

### 2.2 Real behaviour with no test

| Module / function | Why it matters |
|---|---|
| `eval_scoring.py` — **whole module** | Posts `response_match_score`; `AGENTS.md` devotes a section to it being protocol-identical. Its silent-degradation mode is unguarded. |
| `observability.report_cache_outcome` | The `cache-hit`/`cache-miss` tags and `cache.hit` score — the only way to separate the cohorts. A whole documented feature. |
| `second_brain.log_conversation` | A **mandatory** eval criterion (`tool_trajectory_avg_score`, `IN_ORDER`, threshold 1.0). |
| `obsidian_tools.build_obsidian_tools` | Gotcha 11 (`[]` when unconfigured) and the 8-name `tool_filter` — asymmetric with Gmail, whose filter *is* pinned. |
| `agent.get_model` / `OPENROUTER_MODELS` | The `FallbackModel` chain is the reason `**Sources**` exists at all. |
| `agent._score_generation` | Emits three Langfuse scores; `quality.bullet_count` is documented as *not a score*. |
| `agent.cache_hit_before_model` | The `CACHE_ENABLED=false` branch. All ~5 call sites set it `true`. |
| `gmail_oauth.py` | `_exchange_authorization_code` exists purely to work around a library bug. |
| `tools/` — all 8 scripts | `verify_gif.py` proves the decoder matches Pillow. Nothing proves `verify_gif.py`. |
| `docker-compose.yml` topology | The shared-namespace invariant `AGENTS.md` says cost a recording. |

### 2.3 Brittle tests

- `test_adk_wiring.py:555` — asserts `name in {"vault", basename(abspath(root))} or name == "unknown"`. The `or` arm makes it pass for every plausible regression. **It cannot fail.**
- `test_sources.py:356` — builds a `legacy` string, asserts on the *literal*, never passes it to any function under test. **Vacuous**, and demonstrates the opposite of its name.
- `_final_model_text` returns `texts[-1]` — "the last event", which `AGENTS.md` gotcha 13 names as *the exact blind spot* that hid the duplicated-answer bug.
- `test_run_script.py:297` calls `subprocess.run(["script", …])` with **no `skipif`**, unlike its two siblings.
- The three pty tests build `env` as `{**os.environ, …}` unstripped — the one place the "same on every machine" claim fails.
- Eight copy-pasted helpers; `_make_vault` exists with **two different signatures** in two files.
- `parametrize` over the constant under test (`test_adk_devui_patch.py:230` iterates `patcher.EXPECTED`; empty it and the guard passes vacuously) — and it does *not* assert `.chat-input-container` or `.assistant-panel`, the two names the patch's own weakness is about.

### 2.4 Documentation drift — all **[verified]**

| # | Drift |
|---|---|
| D1 | `AGENTS.md:350` — "scaffolds `.env.vaults/`". No such file has ever existed. |
| D2 | `README.md:40-41` — "creates `.env` … **and stops**". `run.sh:49-54` falls straight through to `docker compose up -d --build`. |
| D3 | `AGENTS.md` gotcha 6 — "the `.venv` does **not** have `rouge_score`". **[verified] it is installed.** Gap closed, doc not. |
| D4 | `OBSIDIAN_MCP_PORT` is documented in three places and is **dead in the compose deployment** — `env_file` applies to `agent-testing` only, and `obsidian-mcp` receives only `VAULT_NAME`. It can only ever be `37842`. |
| D5 | `docs/TESTING.md:3` says `:8000`; every other current source says `8001`. |
| D6 | `EVALUATION.md` + `AGENTS.md` use lowercase `model_provider=gemini`; the code reads `MODEL_PROVIDER`. Works only because the default is already `gemini`. |
| D7 | `uv run …` from the repo root appears in four docs — there is no `pyproject.toml` at the root. |
| D8 | `README.md:45` — `adk web text_summarizer` from inside the package; the argument should be `.`. |
| D9 | `INTEGRATIONS.md:53-54` says run the OAuth helper from *inside* `text_summarizer/`; the other three references need the root. |
| D10 | `AGENTS.md:175`'s copy-pasteable `resolve-vault.sh` example exits 1 — the directory is not created first. |
| D11 | `docs/ARCHITECTURE.md:5-50` omits `patch-adk-devui-mobile.py`, all of `tools/`, `getting-started.md`, **4 of 10** test modules, and `before_agent_callback`. It also renders two bogus top-level `--` entries. |
| D12 | `MODELS.md:23-31` shows `get_model()` returning a bare `LiteLm`; the code returns a `FallbackModel` in **both** branches. The doc contradicts itself. |
| D13 | `EVALUATION.md:127` — "change the bullet rule from `3-7` to `3-5`". `agent.py:466` already says 3-5. The experiment is a no-op. |
| D14 | **Gotcha 10 mandates `CACHE_ENABLED=false`, but neither eval helper sets it.** The documented learning loop measures a cached replay on the second run. |
| D15 | `OBSIDIAN_CONFIG_DIR` is read (`vaults.py:256`) and documented nowhere. |
| D16 | `.env.example:19` — `docker compose --env-file langfuse.env up -d` in `text_summarizer/`. No such file; `git log -S` shows it never existed. |
| D17 | `.env.example:32-34` says "set exactly one" but ships **both** Obsidian vars with placeholders; `OBSIDIAN_MCP_URL` wins at `obsidian_tools.py:167`, so a fresh local `.env` points at a server that is not running. |
| D18 | Three files reference `tools/verify_gif.js` (one suggests `verify_gif.js_check.py`). The file is `verify_gif.py`. |
| D19 | `README.md:54-96` duplicates `.env.example`'s variable list with different comments. Two hand-maintained sources, already drifted. |
| D20 | `eval_exercise.py:134` prints a hardcoded `cd C:\Users\carlo\Desktop\agents`. |

---

## §3 — Design rules

The answer to "best practices", stated so they can be *checked* rather than agreed with.

**P1 — One source of truth per fact.** A constant, env-var name, threshold or path lives in
one place. `DEFAULT_VAULT_NAME` is currently in three files; `source_fingerprint` is
reimplemented byte-for-byte in `tools/films.py:35-43`.

**P2 — Resolve configuration once, explicitly, and pass it in.** No module reads
`os.environ` at import time to build a module-level constant. ADK needs a zero-arg
`root_agent`, so one module may hold a default; nothing else may.

**P3 — Depend on behaviour, not framework objects.** Callbacks take a narrow protocol rather
than probing ADK contexts with `getattr`. Duck-typed probes are how two live cache bugs got in.

**P4 — Untrusted input is untrusted.** Anything the model produces — title, topic, summary
body — is data. Never a path segment, never YAML, never a glob.

**P5 — Degrade loudly, never silently.** `except Exception: pass` becomes
`logging.exception` under a named policy. A skipped tool, a failed score, an ambiguous vault
and a fallback backend each leave an observable trace.

**P6 — One command runs everything.** `make check` = format + lint + types + tests + drift.
If it is not one command it will not be run.

**P7 — Every documented command is executed by a test.** Fenced `bash` blocks in the docs
get parsed and their paths and flags asserted. Documentation that is executed cannot rot.

**P8 — Cost of change is a design metric.** Prefer additive changes behind a seam over edits
needing coordinated multi-file changes. `gotcha 1` (the `instruction="""…"""` regex both eval
scripts depend on) is a coupling that has already caused one accidental-corruption class.

---

## §4 — The MVP catalogue

**Independence rule:** every MVP below is a complete, mergeable, independently revertable
slice. *Depends on: nothing.* Where an MVP needs a facility, it builds it locally rather
than assuming another MVP delivered it.

**Effort:** XS <1 h · S 1–3 h · M ½–1½ d · L 2–4 d.
**Risk:** Low (additive) · Med (behaviour change) · High (touches the graded eval path).

---

### TIER A — zero behaviour change. Land any of these first.

---

#### MVP 1 — The Gate
**Value.** The 219 tests start running on every push. Until this lands, every other MVP
ships unverified — and 219 unrun tests are worse than 150 run tests, because they create a
false sense of coverage.

**Ships**
1. `text_summarizer/pyproject.toml`: `[dependency-groups] dev = ["pytest", "pytest-cov", "ruff", "mypy"]` (PEP 735 — no build-backend change). Retires the out-of-tree `/tmp` install in `AGENTS.md:412-423`.
2. `.github/workflows/ci.yml` — `checkout` + `astral-sh/setup-uv` + `uv sync --frozen --group dev`, then `ruff check` → `mypy` → `pytest -q`. `paths:` filter covers source, tests, Docker and infra so it *does* run on agent commits. `uv` cache keyed on `uv.lock`, `timeout-minutes: 15`, summary to `$GITHUB_STEP_SUMMARY`, `.coverage` uploaded.
3. `Makefile` — `check` (the P6 command), plus `run`, `eval`, `fmt`, `lint`, `types`, `test`, `drift`.
4. `ruff` config (`E,F,I,B,UP,SIM`) + `mypy` non-strict with a per-module allowlist for `agent.py` / `second_brain.py`, so the gate lands green on day one and tightens later. `ruff format` on the existing tree is a **separate, mechanical commit**.
5. **Test hygiene** (the §2.3 list): delete the two unfailable tests and re-aim them; hoist the 8 duplicated helpers into `tests/_helpers.py`; unify the two colliding `_make_vault` signatures; add the missing `skipif`; strip `os.environ` in the three pty tests; delete the dead assertion at `test_run_script.py:242`.

**Done when** CI is green on a push; `make check` reproduces it locally; the two replaced
tests fail when their regression is re-introduced (*proven by temporarily reverting the
`run.sh` menu bound and seeing the test go red*).

**Effort** S · **Risk** Low · **Depends on:** nothing
**Decision: approve MVP 1? (Two sub-choices: `make` or `just`; and whether to `ruff format`
6,700 lines in a dedicated commit.)**

---

#### MVP 2 — Documentation Truth
**Value.** The docs currently teach a bug the code fixed, a `.env` behaviour that does not
happen, a port that is dead, and an eval loop that silently measures nothing.

**Ships**
- **D1–D20** prose corrections. Two are *not* prose: **D4** — fix `docker-compose.yml` to
  actually pass `OBSIDIAN_MCP_PORT` to the `obsidian-mcp` service (the doc is right, the code
  is wrong); **D17** — ship only one Obsidian var in `.env.example`.
- **D14** (behaviour, not prose): `eval_exercise.py` and `auto_optimize.py` set
  `CACHE_ENABLED=false` in the `subprocess` env. Today both inherit the ambient
  environment, so the documented learning loop measures a cached replay on run 2.
- **B10** guard: after each rewrite, `auto_optimize.py` asserts rules 7–11 still exist in the
  new instruction block; abort and revert the iteration if not. Without this the loop
  optimises ROUGE-1 and destroys the `tool_trajectory_avg_score` gate.
- **B8**: re-label `tools/films.py:128` so the published walkthrough stops claiming the
  rewrite lives in `after_agent_callback`. *(Alternative to re-recording the scene: fix the
  label and re-run `build_gifs.py`, which spends quota and needs a live stack.)*
- **D19**: pick one source for the env block — delete one of the two hand-maintained copies.

**Done when** a `grep` for `.env.vaults/`, `model_provider=`, `127.0.0.1:8000` and
`C:\Users\carlo` returns nothing outside history; and `auto_optimize.py` provably reverts an
iteration that drops rule 8.

**Effort** S (or M with the B10 guard) · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 2? And for B8 — re-label only, or re-record the GIF?**

---

#### MVP 3 — The Doc-Drift Ratchet
**Value.** Makes MVP 2 permanent. Each check corresponds to a specific drift item, so this
converts a one-off cleanup into a gate.

**Ships** `tests/test_drift.py` with 9 checks:
1. **Env inventory** — every `os.environ` read in `text_summarizer/`, `run.sh`,
   `resolve-vault.sh` is documented in `.env.example` → catches **D15**.
2. **Docker reachability** — every Docker-only var in `.env.example` is actually delivered
   to the service that reads it (asserts compose `env_file` / `environment` coverage) →
   catches **D4**.
3. **Port consistency** — `8001` / `37842` literals across compose, `resolve-vault.sh`,
   `run.sh`, `check_stack.sh` and every doc agree → catches **D5**.
4. **Command smoke test** — extract fenced `bash` blocks from `README.md`, `AGENTS.md`,
   `docs/*.md`; assert paths exist, the implied cwd is valid, and flags exist on the target's
   `argparse` → catches **D2, D6, D7, D8, D9, D10, D20**.
5. **Generated trees** — the `docs/ARCHITECTURE.md` and `AGENTS.md` layout blocks match
   `git ls-files` → catches **D11**.
6. **Threshold single-source** — the two criteria in `tests/eval/test_config.json` are
   exactly the two documented in `EVALUATION.md` → catches **D13**.
7. **Rule survival** — the live `instruction="""…"""` block still contains every rule cited
   by number in `AGENTS.md` / `tools/films.py` → the B10 tripwire.
8. **Fingerprint parity** — `films.fingerprint() == second_brain.source_fingerprint` over a
   table of prompts (P1 — the duplication is currently exact but unguarded).
9. **Generated-artifact integrity** — running `tools/inject_player.py` leaves `git diff`
   empty; the four icon files match `make_favicon.py`'s output → catches **D18**.

**Independence note (this is why the MVP is mergeable today).** Checks 1–9 would *fail*
against the current tree. So the file ships with a **`tests/fixtures/drift_baseline.json`**
allowlisting today's known drift. The suite is green on arrival; **MVP 2 removes entries from
the allowlist** one at a time and the ratchet refuses to let them come back. Ratchet, not a
wall.

**Done when** the suite passes with the baseline populated, and deleting one baseline entry
turns exactly one check red.

**Effort** M · **Risk** Low** · **Depends on:** nothing — *it ships with a baseline that
records the current state; it is stricter the moment MVP 2 lands, and useful before that.*
**Decision: approve MVP 3? (All 9 checks, or start with 1, 2, 8, 9 — the highest value, the
lowest noise?)**

---

#### MVP 4 — Tools Pipeline Tests
**Value.** `tools/` is 3,190 lines that record the agent and produce the published site, with
zero tests. `verify_gif.py` exists to prove the decoder matches Pillow; nothing proves
`verify_gif.py`.

**Ships**
- `tests/test_tools_gif.py`: build a 3-frame GIF from synthetic frames in `tmp_path`, extract
  the decoder by the same slicing `verify_gif.py` uses, run it, compare SHA-256 per frame
  against Pillow — the pipeline's own contract, on a fixture instead of a live Firefox.
- Parametrised coverage of `build_gifs.distinct()` (near-identical frame dropping) and the
  255-colour shared-palette path, with the hardcoded `FONTDIR` made overridable.
- `films.py`: assert the panel strings are actually read from the vault — replace the
  `vault.notes[-1]` fallbacks with a hard failure, so "every string in a panel is real" stays
  true when a named note is absent.
- Stop hardcoding `MODEL = "gemini-3.5-flash-lite"` and the `2026-09-26` chat-log date in
  `films.py`; read them from the capture.

**Done when** the synthetic GIF round-trips through the extracted decoder byte-for-byte, with
no browser and no network.

**Effort** M · **Risk** Low** · **File contention:** `tools/films.py` only.
**Decision: approve MVP 4, or declare `tools/` "run it on this machine" and say so in
`AGENTS.md`?**

---

### TIER B — closes live defects.

---

#### MVP 5 — Untrusted Input  ⭐ *live security bug*
**Value.** Closes two reproducible defects: model output escaping the vault, and model
output breaking the YAML frontmatter.

**Ships** (`text_summarizer/second_brain.py` + new `tests/test_second_brain_write.py`)
1. Generalise `_slug` into `_safe_name(value) -> str`: strips `/`, `\`, `..`; collapses to
   `[A-Za-z0-9 _-]`; returns `""` for `.` / `..` / empty; caps length.
2. Apply it to **topics** — today only `title` is slugged. The topic *body* keeps the
   human-readable original so `## Related` still links `[[Mobile App]]`; only the *filename*
   is slugged. Safe file, human link.
3. One containment assert before any write: `Path(p).resolve().is_relative_to(Path(VAULT_ROOT).resolve())`
   for note, topic and index paths. Catches every future regression, including symlinked parents.
4. Frontmatter: route every scalar through `_yaml_str()` (`json.dumps` — valid YAML 1.2
   flow-scalar syntax, and `json` is already imported by `sources.py`). A value containing
   `\n` then cannot escape the block. Applied to `title` in `aliases`, every topic in `tags`,
   and the `## Related` wikilinks.
5. Delete the dead branch at `second_brain.py:315` (B4).

**Tests** — the exact B1 and B2 probes as regression tests: `test_a_traversing_topic_cannot_escape_the_vault`;
`test_a_topic_filename_is_slugged_but_the_link_is_not`; `test_a_title_with_a_newline_cannot_break_the_frontmatter`
(parses the note and asserts one intact YAML block); `test_yaml_values_with_quotes_round_trip`;
`test_every_written_path_is_inside_the_vault_root` (parametrised over payloads);
`test_a_note_twice_on_the_same_day_is_updated_not_duplicated` (also covers B4).

**Done when** the B1 and B2 probes from §2.1 are re-run and produce no file outside the vault
and no malformed note.

**Effort** M · **Risk** Low** (additive; valid existing vaults unaffected) · **Depends on:** nothing
**Decision: approve MVP 5? (Sub-choice: sanitise all unsafe filename characters — which
**renames existing on-disk topic files containing `/` or `:`**, a one-way change — or block
only traversal sequences and leave everything else alone?)**

---

#### MVP 6 — Shell & Bootstrap
**Value.** Fixes a silent failure and a `--help` that prints source code.

**Ships** (`run.sh` + `tests/test_run_script.py`)
- **B6** — move the usage text into a `usage()` function and `printf` it, so the `sed` range
  can never drift again.
- **B5** — introduce `LAST_VAULT` / `LAST_OPTION` explicitly; validate against `LAST_OPTION`;
  make the final `sed -n "${answer}p"` fail **loudly** with the same "expected 1-N" message, so
  a future arithmetic slip prints a diagnostic instead of `exit 1` in silence.
- **B7** — `run.sh:176` → "or `--vault NAME` to select a vault Obsidian already knows". Drop
  "by symlink".
- `mkdir -p` the parent for **absolute** `OBSIDIAN_VAULT_PARENT_HOST` too, or fail with a
  clear message.
- Tests: `test_the_menu_rejects_the_choice_past_the_advertised_range` — the **boundary**
  (`N+3` with 2 vaults, i.e. exactly B5, which the current `99` test does not cover);
  `test_help_prints_usage_and_no_source`; `test_an_absolute_vault_parent_is_created`.

**Done when** `./run.sh --help` contains no line of the file, and choosing `N+3` prints
`out of range: 'N+3' (expected 1-N)` instead of exiting silently.

**Effort** S · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 6?**

---

#### MVP 7 — Durable Writes
**Value.** Closes lost updates under concurrent sessions, and replaces truncate-then-write
with an atomic replace.

**Ships** (`second_brain.py` + new `tests/test_second_brain_io.py`)
- `_write(path, content)` → temp file in the same directory + `os.replace` (+ fsync). Every
  caller benefits for free.
- `_append_once(path, line, header)` → `fcntl.flock` on a sidecar `.lock` + read + conditional
  append + atomic replace. Replaces the duplicated index logic at `second_brain.py:292-297`
  and `:383-388` — one implementation, two call sites (SRP + DRY).
- Chat log: today's file is append-only by construction, so its read-modify-write becomes a
  plain append.
- `find_cached_summary`: read **frontmatter only** (open + read to the closing `---`, ~2 KB)
  instead of every file's full body — a pure win that also retires the
  "regexes the whole note body" note in the docstring. The O(1) *index* is deliberately
  deferred to MVP 12.
- Tests: a raise mid-write leaves no partial file; a `multiprocessing` test proves N
  concurrent appends all land; frontmatter-only read ignores a `source_fingerprint` string
  appearing in the *body*.

**Done when** a 20-process concurrent-append test loses zero entries, and the cache lookup
reads a bounded number of bytes per note.

**Effort** M · **Risk** Med** (behaviour change on shared files) · **File contention:**
`second_brain.py`, shared with **MVP 5** and **MVP 12** — land these one at a time, any order.
**Decision: approve MVP 7? (Sub-choice: POSIX `fcntl.flock` only, or add an `msvcrt` path for
Windows?)**

---

### TIER C — new capability, purely additive.

---

#### MVP 8 — The Untested Half
**Value.** Five modules with real, graded behaviour and zero tests — including
`eval_scoring` and `log_conversation`, which is a **hard eval criterion**.

**Ships** (test-only; no source change required — monkeypatch the module constants)
- `tests/test_eval_scoring.py` — both fixture shapes; `load_eval_cases` keys on the
  normalised prompt; `rouge1_fmeasure` in [0,1]; **`test_the_eval_extra_is_installed` fails
  loudly** (does not skip) when `_calculate_rouge_1_scores` is `None`, because a silently
  missing metric is worse than a red test; the `_normalize` contract that matches prompt to golden.
- `tests/test_second_brain_tools.py` — `log_conversation` end to end: dated note, timestamp
  header, index append, and that a cache hit skips it. Plus `save_summary_to_second_brain`'s
  topic stubs, `## Related` block, and the `created_topics` message.
- `tests/test_obsidian_toolset.py` — `build_obsidian_tools`: both transports, the
  mutual-exclusion rule, the relative-path rejection at `obsidian_tools.py:170`, the `[]`
  return when unconfigured (gotcha 11) and when `google-adk[mcp]` is missing, and the exact
  8-name `tool_filter` — the shape `test_gmail.py` already pins for Gmail.
- `tests/test_model_selection.py` — the `FallbackModel` chain order, `MODEL_ALIAS` selection,
  and that both branches produce a chain starting with the configured primary.
- `tests/test_scoring.py` — `_score_generation` for all three metrics, the `CACHE_ENABLED=false`
  branch, and the empty-input paths.
- `tests/test_observability.py` — `report_cache_outcome` against a capturing double: the
  `cache-hit`/`cache-miss`/`cache-disabled` tag, both scores, and the never-raises contract.
- `tests/test_gmail_oauth.py` — `_load_client_config` in all three shapes;
  `_exchange_authorization_code` against a stubbed `requests.post`, pinning the
  scope-mismatch workaround that exists purely because the library raises on it.

**Done when** each of the five modules has a test that fails when the behaviour regresses
(*proven by temporarily breaking each and observing red*).

**Effort** M · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 8? (Sub-choice: should a missing `rouge_score` **fail** the suite —
recommended — or skip?)**

---

#### MVP 9 — `doctor`
**Value.** Four places currently answer "is this deployment actually wired up?" —
`tools/check_stack.sh` (shell, hardcoded ports, greps an upstream Rust log format),
`tests/conftest.py` (the env-pinning ritual), the `AGENTS.md` gap list (prose), and trial and
error. This replaces all four with one command that uses the same code the agent uses.

**Ships** `text_summarizer/doctor.py`, run as `python -m text_summarizer.doctor [--json] [--no-network]`
— exit 0 all-ok, 1 on any `fail`, each check printing a one-line fix.

| # | Check | Reuses |
|---|---|---|
| 1 | Python / ADK / `rouge_score` / `mcp` / `google-api-python-client` present | — |
| 2 | Every env var: set, plausible, **documented in `.env.example`** | the MVP 3 check 1 logic, inlined |
| 3 | `MODEL_PROVIDER` resolves; print the exact fallback chain and each backend's served name | `get_model` |
| 4 | Vault root, vault name, `VaultIdentity.source`, and the ambiguity refusal | `vaults.select_vault` |
| 5 | **Vault parity** — run `resolve-vault.sh` with a stub `obsidian-mcp` on `PATH` and diff its choice against Python's. The invariant `AGENTS.md` calls load-bearing, today covered only indirectly | both implementations |
| 6 | MCP reachable; `vault_info` returns a name **matching** the Python side | the real server |
| 7 | `CACHE_ENABLED` state, note count, measured lookup ms | `find_cached_summary` |
| 8 | Langfuse: keys, base URL reachable within `LANGFUSE_AUTH_CHECK_TIMEOUT`, instrumenting or not | `observability` |
| 9 | Gmail: all three vars, refresh token valid, `gmail.readonly` scope | `gmail_mcp_server` |
| 10 | Dev UI: patch marker, `100dvh` count, viewport meta | `patch-adk-devui-mobile.py`'s `EXPECTED` |
| 11 | Write probe — create/delete a scratch note **inside** the vault, asserting containment | the writer |
| 12 | Free OpenRouter fallback: a one-token probe per `:free` model, reporting which are 429-ing right now (`AGENTS.md` gap 7) | `get_model` |

Then **delete `tools/check_stack.sh`** and repoint `AGENTS.md`'s recording gate at
`doctor --json` — one implementation instead of two.

**Done when** `doctor --json` reports the same verdict as the four places it replaces, and
`check_stack.sh` is gone with `AGENTS.md` updated.

**Effort** L · **Risk** Low** · **Depends on:** nothing — every check calls existing code;
none needs a refactor first.
**Decision: approve MVP 9? And may I delete `tools/check_stack.sh` in its favour?**

---

#### MVP 10 — Turn Metrics
**Value.** Makes the cache's return *provable* rather than asserted. Today `cache.hit` and
`cache.lookup_ms` exist with no paired latency number to compare against — the whole
`AGENTS.md` cache story rests on "first ask 23.5 s, repeat 0.04 s", which was measured by
hand, once.

**Ships** — per-trace scores in `after_agent_callback`, all deterministic and free:
`turn.model_calls`, `turn.tokens_in`, `turn.tokens_out`, `turn.latency_ms`,
`turn.cache_saved_ms`; plus the served model as a trace tag (already available on the
response). Extend `report_scores_after_agent` — a new additive score set, and it **keeps
returning `None`** (gotcha 16). Tests: the scores are emitted, are numeric, are absent on a
cache hit, and the callback still returns `None`.

**Done when** cache-hit and cache-miss cohorts are separable by latency and token count in
Langfuse without a manual measurement.

**Effort** S · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 10?**

---

#### MVP 11 — Degradation Banner
**Value.** Converts three currently-**silent** failure modes into visible ones. `AGENTS.md`
records that an unreachable MCP server reads exactly like a retrieval bug and "cost a
recording".

**Ships** a machine-readable turn banner rendered below the answer, from real state that is
already in hand (`VaultIdentity.source`, `LlmResponse.model_version`, MCP reachability):
```
⚠ served by qwen/qwen3.8-27b:free (Gemini quota exhausted)
⚠ vault selection ambiguous — 2 candidates, set OBSIDIAN_VAULT_NAME
```
Rendered by `sources.py` (which already owns response rendering) so it is deterministic, not
model-authored. **The banner must be excluded from `summary_only` and from ROUGE-1** or it
depresses the golden scores — that exclusion is a test, not a hope.

**Done when** each of the three states produces the right banner in a real turn, and a test
asserts ROUGE-1 is unchanged when a banner is present.

**Effort** S · **Risk** Med** (changes response text → moves ROUGE-1) · **Depends on:** nothing
**Decision: approve MVP 11? (It touches the graded output path, so it needs an eval run
afterwards — real quota.)**

---

#### MVP 12 — Cache Index
**Value.** Closes `AGENTS.md` gap 10: `find_cached_summary` is O(n) full-file reads on the
path of **every single turn**.

**Ships** `Second Brain/.index.json`, `fingerprint → note filename`, written by
`save_summary_to_second_brain`, read by `find_cached_summary`, rebuilt on demand when
missing or when the directory mtime post-dates it (so a hand-edited vault self-heals).
**Includes** tests for the three states — hit, miss, stale-index rebuild — and a test that a
corrupt index file degrades to the O(n) scan rather than failing the turn.

**Recommendation: defer.** MVP 7's frontmatter-only read already removes most of the cost,
and this was measured at 0.5 ms against 19 notes. The index earns its place at ~500 notes, or
once MVP 7's concurrency work makes it load-bearing. **If you would rather not carry the
debt, say so and I will note the deferral in `AGENTS.md` instead of building it.**

**Effort** M · **Risk** Med** (a stale index is a silent wrong answer) · **File contention:**
`second_brain.py`, shared with MVP 5 / MVP 7 · **Depends on:** nothing
**Decision: build MVP 12 now, or defer (recommended)?**

---

#### MVP 13 — Streaming Guard
**Value.** Closes `AGENTS.md` gap 12. `render_sources_after_model` returns early for streamed
chunks, so a `StreamingMode.SSE` turn would show the user raw `@@ADK_VAULT@@` tokens. Inert
as deployed — the dev UI posts `streaming: false` — and therefore invisible until someone
turns streaming on.

**Ships — recommended scope:** replace the silent `return None` with a **loud** refusal: a
`logging.warning` naming `sources-render-skipped: streaming` and a trace tag, plus a startup
warning if `StreamingMode` is ever non-NONE. Makes the limitation *observable* for ~1 h.
**Full scope** (the alternative): buffer chunks per `invocation_id` in session state and
substitute on the final non-partial chunk, which actually *supports* streaming.

**Done when** an SSE turn cannot silently leak a sentinel token.

**Effort** XS (guard) / M (full) · **Risk** Low / Med** · **Depends on:** nothing
**Decision: approve MVP 13 — the guard, or the full buffering? (I recommend the guard unless
streaming is on your roadmap, because the buffering is only worth its cost if it is.)**

---

#### MVP 14 — Vault Semantic Retrieval
**Value.** The cheapest quality win available. `obsidian-mcp` already ships a semantic search
daemon — its Dockerfile even reserves `/root/.local` for it — and rule 7 tells the agent to
use `search_text`, which is **lexical only**. The strongest available retrieval is exposed
and unused.

**Ships** — expose `search_semantic` in the `obsidian_tools` filter; extend rule 7 to prefer
it with `search_text` as the lexical fallback; add `search_semantic` to the `MVP 8` filter
test so it cannot silently disappear. **Run a full eval before and after** — this changes the
model-visible tool set, so it is the one Tier-C MVP that touches the graded path.

**Done when** a real turn cites a note that shares no keywords with the prompt, and
`tool_trajectory_avg_score` stays 1.0.

**Effort** S · **Risk High** (changes the graded output shape; needs two real eval runs, real quota)
**Decision: approve MVP 14? (Sub-choice: ship it behind `VAULT_SEMANTIC=opt-in` so the
current behaviour stays the default until the eval proves the gain.)**

---

#### MVP 15 — Digest Mode
**Value.** `adk web` is a chat box; "what did the agent learn today?" is a different product,
and this one needs **no agent turn and no quota** — it just reads `Second Brain/` for today.

**Ships** `python -m text_summarizer.digest [--date YYYY-MM-DD] [--format md|text]`,
summarising the day's notes: titles, topics, backlinks, which of them cite which sources.
Reads the existing vault layout, uses `strip_sources_block` from `sources.py`, and writes
nothing. Tests: date selection, empty day, and the `--format` switch.

**Effort** S · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 15?**

---

### TIER D — refactors. Do these last; the payoff is testability, not behaviour.

---

#### MVP 16 — Metric Truth
**Value.** Closes `AGENTS.md` gaps 3, 4 and 13 — the three documented cases where a metric
name lies about what it measures.

**Ships**
- **Gap 3** — `quality.bullet_count` is documented as 0..1 but returns the raw count, and the
  computed `bullet_score` is discarded. Emit **both** (`quality.bullet_count_raw` +
  `quality.bullet_score`) so existing dashboards keep working.
- **Gap 4** — `quality.fidelity` is lexical recall, not faithfulness. **Rename** to
  `quality.lexical_recall` and add a real, deterministic, free groundedness signal:
  **novel-content-word rate** — the fraction of content words in the summary absent from the
  source. High = hallucination risk. This pushes *against* the copy-verbosity bias (gap 5)
  rather than with it.
- **Gap 13** — make "assert on the last event" a lint rule: grep the test tree for
  `texts[-1]` / `events[-1]` with an allowlist. The cheapest possible guard against the exact
  blind spot that cost the duplicated-answer bug.
- **Gap 6** — `AGENTS.md` gotcha 6 is stale (`rouge_score` is installed).

**Done when** no metric's name contradicts its value, and a regression to a last-event
assertion fails lint.

**Effort** S · **Risk** Low** · **Depends on:** nothing
**Decision: approve MVP 16? (Sub-choice: rename `quality.fidelity` — that **breaks existing
Langfuse dashboards**. Keep it as an alias for a release, or cut over?)**

---

#### MVP 17 — One Turn-Text Function
**Value.** "What did the user actually ask this turn?" is answered **four** times, in two
modules, from three different ADK fields — and they are not equivalent. That difference *is*
the cache bug documented in `AGENTS.md` ("The vault cache — one key, computed in code"). Four
copies, with the write/lookup invariant maintained by prose and one test.

**Ships**
1. New `text_summarizer/turn.py`: a structural `TurnText` protocol (`user_content`), plus
   `turn_text(source, *, events=())`, `last_model_text(events)`, `service_model(events)`,
   `parts_text(content)`.
2. `turn_text` reads `user_content` first — **the same field on both sides** — and the event
   scan only as an explicit, *logged* fallback, because that silent fallback is what gotcha
   12 is about.
3. Delete `agent._last_user_text`, `agent._user_content_text`, `agent._last_session_user_text`,
   `agent._last_session_model_text`, `second_brain.user_text_from_context`,
   `second_brain.cache_key_text`. `service_model` moves from `sources.py` unchanged.
4. The invariant becomes structural: the write key and the lookup key are now literally the
   same function call on the same protocol, so they **cannot** drift.
5. Tests: the three context shapes agree; `user_content` beats a tool result in
   `llm_request.contents` (the gotcha that made `_last_user_text` wrong); a pre-turn event
   snapshot is *detected* rather than silently used; `service_model` returns `None` for a
   cached replay (no `model_version` — the signal nothing was generated).

**Done when** the four implementations are gone, the 219 existing tests still pass unchanged
(except the ones that referenced the deleted helpers), and a new test proves the write and
lookup paths are the same code.

**Effort** M · **Risk Med** (this is the code path two live bugs lived in) · **Depends on:** nothing
**Decision: approve MVP 17? I rate it the highest-value refactor here, because it converts a
comment-maintained invariant into a type-level one.**

---

#### MVP 18 — Ports & Factories
**Value.** Makes the callbacks testable without ADK, without `SimpleNamespace`, and without a
filesystem — which is what makes the four `except Exception: print(...)` blocks in `agent.py`
testable at all (P5 is unenforceable while the code is welded to ADK objects).

**Ships**
1. `text_summarizer/ports.py` — three `Protocol`s plus frozen value objects:
   `SummaryCache.find(text) -> CacheLookup(hit, body, elapsed_ms)`;
   `SourcesRenderer.render(text, *, vault, model) -> str`; `TurnScorer.score(user, model) -> Mapping[str, float]`.
2. `agent.py` gains `make_cache_callback(cache, reporter)` and
   `make_render_callback(renderer, identity)` factory functions closing over the dependency.
   `cache_hit_before_model` / `render_sources_after_model` become thin module-level singletons
   built from the production wiring — so **every existing test keeps passing unchanged**.
3. `observability.CacheReporter` as a Protocol, which also makes `report_cache_outcome`
   trivially testable via a capturing double.
4. Centralise "must never break the turn" into one `@never_break_turn(reason=…)` decorator
   that logs and returns `None`, replacing four hand-rolled try/excepts with a single named
   policy.
5. `tests/test_ports.py` + `tests/test_cache_callback.py` driven entirely by doubles — no env,
   no filesystem, no `SimpleNamespace`.

**Done when** every callback test constructs its collaborators as one-line lambdas.

**Effort** M · **Risk Low** (purely additive; the module-level names are preserved) · **Depends on:** nothing
**Decision: approve MVP 18? (Sub-choice: factory functions closing over dependencies, or ports
passed as ADK-injected state? The factories are more testable but move the wiring point.)**

---

#### MVP 19 — `Settings` Object
**Value.** The root cause of the `conftest.py` import ritual, of five modules each reading
`os.environ` independently, and of `README.md`/`.env.example` being two hand-maintained
copies of the same variable list. **This is the biggest change in the catalogue and the one
with the least user-visible payoff** — it buys testability, and it is deliberately last.

**Ships**
1. New `text_summarizer/config.py`: a frozen `Settings` dataclass with a single
   `from_env(env)` factory that validates and coerces **once** — `CACHE_ENABLED` → `bool`
   (`ValueError` on a typo, so a misspelling cannot silently mean "on"), and the three Google
   vars behind a redaction wrapper so a key cannot land in a `repr`.
2. `obsidian: ObsidianSettings | None` and `gmail: GmailSettings | None` — `None` *is* the
   "tools disabled" case, so gotcha 11 becomes unrepresentable rather than checked.
3. `agent.py` keeps exactly one module-level `settings` + `build_model(settings)`, because
   ADK needs a zero-arg `root_agent`. Everything else takes settings as a parameter.
4. `build_obsidian_tools(settings)` / `build_gmail_tools(settings)` — the `os.environ` reads
   move out of both modules. `build_model` extracted from `get_model()` becomes pure.
5. `second_brain.set_vault_root(path)` so tests stop monkeypatching the constant.
6. `tools/gen_env_example.py` **generates** `.env.example` and the `README.md` env block from
   the declarations in `config.py` — which is what finally closes D19 and D15 and D17 (P1).
7. `tests/conftest.py` shrinks from 76 lines to ~12.

**Done when** importing the package with an empty environment works, `conftest.py` is a
dozen lines, and regenerating `.env.example` leaves `git diff` empty.

**Effort** L · **Risk Med** (touches every module's entry point) · **File contention:**
`agent.py`, `obsidian_tools.py`, `gmail_tools.py`, `second_brain.py`, `observability.py`,
`.env.example`, `README.md` — **the broadest MVP in the catalogue** · **Depends on:** nothing,
but MVP 3's check 1 becomes a two-way equality assertion once this lands.

**Decision: approve MVP 19? I would not build it before MVP 8 and MVP 17 — those extract the
parts that would otherwise be the riskiest part of this refactor.**

---

## §5 — Target folder structure

**Constraints that shape it:** ADK discovers the agent package by directory, and
`agent.py`'s `root_agent` must stay importable as `text_summarizer.agent.root_agent` because
`eval_exercise.py`, `auto_optimize.py`, `eval_scoring.py` and the Dockerfile all depend on
that path. `gotcha 1` additionally pins the literal `instruction="""…"""` in `agent.py`.
**The package name and the module name do not move.**

```
agent-testing/
├── AGENTS.md  README.md  PLAN.md  Makefile          # make check (MVP 1)
├── run.sh  resolve-vault.sh                           # MVP 6
├── Dockerfile  Dockerfile.obsidian-mcp  docker-compose.yml
├── patch-adk-devui-mobile.py
│
├── .github/workflows/
│   ├── ci.yml                                        # MVP 1
│   └── pages.yml
│
├── text_summarizer/
│   ├── __init__.py           # load_dotenv → observability → export root_agent
│   ├── agent.py              # ★ wiring: settings, model, callbacks, tools, root_agent
│   │                         #   instruction="\"\"\"...\"\"\"" stays here (gotcha 1)
│   ├── config.py             # NEW  MVP 19 — Settings; the one place env is read
│   ├── ports.py              # NEW  MVP 18 — the protocols the agent depends on
│   ├── turn.py               # NEW  MVP 17 — TurnText + the ONE turn-text function
│   ├── doctor.py             # NEW  MVP 9
│   ├── digest.py             # NEW  MVP 15
│   │
│   ├── sources.py  second_brain.py  vaults.py
│   ├── obsidian_tools.py     # MVP 14 adds search_semantic
│   ├── gmail_tools.py  gmail_mcp_server.py  gmail_oauth.py
│   ├── observability.py      # MVP 10 adds turn.* scores
│   ├── eval_scoring.py
│   ├── auto_optimize.py  eval_exercise.py  check_free_models.py   # CLIs, stay at top level
│   ├── pyproject.toml        # + dev group, [tool.ruff], [tool.mypy]
│   └── tests/
│       ├── conftest.py       # SHRINKS 76 → ~12 lines (MVP 19)
│       ├── _helpers.py       # NEW  MVP 1 — the 8 de-duplicated helpers
│       ├── test_drift.py     # NEW  MVP 3
│       ├── fixtures/drift_baseline.json  # NEW  MVP 3 — makes the ratchet land green
│       ├── unit/             # NEW  pure: no ADK, no filesystem
│       ├── contract/         # NEW  hand-built ADK-shaped doubles
│       ├── integration/      # NEW  real runner, real filesystem, zero API cost
│       └── eval/             # ⚠ PATH UNCHANGED — the CLI hard-codes it
│           ├── simple_test.test.json
│           ├── summarizer_eval_set.evalset.json
│           └── test_config.json
│
├── tools/                    # not in the image; records the agent
│   ├── check_stack.sh        # DELETE (MVP 9)
│   ├── gen_env_example.py    # NEW  MVP 19
│   ├── capture_ui.py  firefox_marionette.py  build_gifs.py
│   ├── films.py  gif_player.js  inject_player.py  verify_gif.py  make_favicon.py
│   └── test_tools_gif.py     # NEW  MVP 4
│
└── docs/
```

**Why not a bolder restructure** (no `domain/`, `adapters/`, `application/`):
1. ADK resolves the agent package by path, and every script, the Dockerfile and the eval
   config assume `text_summarizer.agent`.
2. The package is **3,088 lines**. Sub-packaging for its own sake is ceremony. The win that
   matters — separating *where config comes from* from *what the agent does* — is MVP 19,
   and that is one file, not a tree.
3. The three boundaries that genuinely need a seam (`config`, `ports`, `turn`) each get one.

The **test tree does** get split, because that is where the real confusion lives: today
"which file does this go in" has four answers (`test_sources`, `test_agent_callback`,
`test_adk_wiring` all touch `sources.py`). The split is by *what you need* — pure / ADK-shaped
/ real — not by *what you test*.

**This layout is a union of every MVP above, not a prerequisite for any of them.** Each MVP
can ship into the current flat layout and the move to `unit/ contract/ integration/` happens
whenever, as a `git mv` series.

---

## §6 — Independence audit

The user requirement was that no MVP depends on another. Audited:

| MVP | Reads from other MVPs | Builds what it needs locally | Verdict |
|---|---|---|---|
| 1 The Gate | — | its own dev group, workflow, Makefile | **independent** |
| 2 Doc Truth | — | its own corrections; the B10 guard is self-contained | **independent** |
| 3 Drift Ratchet | — | **ships a baseline that records today's drift**, so it is green on arrival | **independent** |
| 4 Tools Tests | — | its own synthetic fixtures | **independent** |
| 5 Untrusted Input | — | its own `_safe_name` / `_yaml_str` | **independent** |
| 6 Shell | — | its own `usage()`; replaces the `sed` range | **independent** |
| 7 Durable Writes | — | its own `_append_once` with its own lock | **independent** |
| 8 Untested Half | — | monkeypatches module constants; **needs no source change** | **independent** |
| 9 doctor | — | calls the existing `vaults`, `get_model`, `resolve-vault.sh`, patcher | **independent** |
| 10 Turn Metrics | — | additive score set; keeps returning `None` | **independent** |
| 11 Banner | — | `sources.py` already owns response rendering | **independent** |
| 12 Cache Index | — | its own index format + rebuild rule | **independent** |
| 13 Streaming Guard | — | its own warning | **independent** |
| 14 Semantic RAG | — | adds one filter entry; MVP 8's filter test is *strengthened*, not required | **independent** |
| 15 Digest | — | reads the existing vault layout; reuses `strip_sources_block` | **independent** |
| 16 Metric Truth | — | emits both old and new names | **independent** |
| 17 One Turn-Text | — | its own `turn.py`; the four deletions are self-contained | **independent** |
| 18 Ports | — | its own protocols; preserves every existing public name | **independent** |
| 19 Settings | — | its own `config.py`; does not require 17/18 to have landed | **independent** |

**Ordering is therefore a *preference*, not a constraint.** My recommendation, on
risk-reduction per hour:

1. **MVP 5** — the live security bug. Highest urgency, additive, M.
2. **MVP 1** — the gate, so that 6 and 5 are protected by 219 running tests. S.
3. **MVP 6 + 2** — the two user-visible defects and the doc that teaches them. S each.
4. **MVP 8** — the untested half, before the refactors, so the refactors are guarded. M.
5. **MVP 9** — `doctor`, which is the payoff for everything already instrumented. L.
6. Then whichever of 3 / 4 / 7 / 10 / 16 / 17 appeals.
7. **MVP 19 last**, and only if 8 and 17 are already in — they extract the riskiest part of it.

**File contention** (not a dependency — these cannot be merged simultaneously, but any order
works): `second_brain.py` is touched by **5, 7, 12**; `agent.py` by **10, 11, 13, 14, 16,
17, 18, 19**; `run.sh` by **2** (docs only) and **6**; `tools/films.py` by **2** and **4**.

---

## §7 — Approval

Each MVP is a separate decision. Approve any subset; I build only what is approved, one at a
time, and stop for the next approval after each.

| # | MVP | Tier | Effort | Risk | Reversible | Approve? |
|---|---|---|---|---|---|---|
| 1 | The Gate | A | S | Low | yes | ☐ |
| 2 | Documentation Truth | A | S–M | Low | yes | ☐ |
| 3 | The Doc-Drift Ratchet | A | M | Low | yes | ☐ |
| 4 | Tools Pipeline Tests | A | M | Low | yes | ☐ |
| 5 | **Untrusted Input** | B | M | Low | yes | ☐ |
| 6 | Shell & Bootstrap | B | S | Low | yes | ☐ |
| 7 | Durable Writes | B | M | Med | yes | ☐ |
| 8 | The Untested Half | C | M | Low | yes | ☐ |
| 9 | `doctor` | C | L | Low | yes | ☐ |
| 10 | Turn Metrics | C | S | Low | yes | ☐ |
| 11 | Degradation Banner | C | S | Med | yes | ☐ |
| 12 | Cache Index | C | M | Med | yes | ☐ *(recommend defer)* |
| 13 | Streaming Guard | C | XS / M | Low | yes | ☐ |
| 14 | Vault Semantic RAG | C | S | **High** | yes | ☐ |
| 15 | Digest Mode | C | S | Low | yes | ☐ |
| 16 | Metric Truth | D | S | Low | yes | ☐ |
| 17 | One Turn-Text Function | D | M | Med | yes | ☐ |
| 18 | Ports & Factories | D | M | Low | yes | ☐ |
| 19 | `Settings` Object | D | L | Med | yes | ☐ |

**Sub-choices waiting on you** (only for the MVP you approve):

- **1** — `make` or `just`? And `ruff format` all 6,700 lines in a dedicated commit, or `ruff check` only?
- **2** — for B8, re-label the GIF scene only, or re-record it (spends quota, needs a live stack)?
- **3** — all 9 drift checks, or start with 1, 2, 8, 9 (highest value, lowest noise)?
- **4** — or declare `tools/` "run it on this machine" in `AGENTS.md` instead?
- **5** — sanitise all unsafe filename characters (**renames existing topic files** containing `/` or `:`, a one-way change), or block only traversal sequences?
- **7** — POSIX `fcntl.flock` only, or an `msvcrt` path for Windows?
- **8** — should a missing `rouge_score` **fail** the suite (recommended) or skip?
- **9** — may I delete `tools/check_stack.sh` in its favour?
- **11** — needs a real eval run afterwards (real quota). Confirm.
- **13** — the guard (XS), or full buffering (M)?
- **14** — ship behind `VAULT_SEMANTIC=opt-in` so current behaviour stays default until the eval proves the gain?
- **16** — renaming `quality.fidelity` breaks existing Langfuse dashboards. Keep an alias for a release, or cut over?
- **18** — factory functions closing over dependencies, or ports passed as ADK-injected state?
- **19** — confirm you want this at all; I would not build it before 8 and 17.

**Explicitly recommended against:**

- **Structured output / `output_schema`** (would let `render_sources` lose its 400-line
  tolerant parser). `sources.py`'s leniency is a *feature* — it survives a model that ignores
  the format — and free OpenRouter backends support structured output unevenly, so the
  realistic outcome is **two** rendering paths instead of one.
- **A bolder package restructure** into `domain/ adapters/ application/` — see §5.
- **Web-fetch integration** — widens the trust boundary, and a fetched page is untrusted input
  (P4) and a prompt-injection surface. Not proposed as an MVP.
- **A Slack sink** — defensible, but a new external integration is a bigger commitment than
  the value it adds to a demo. Not proposed as an MVP; ask if you want it and I will scope it.

---

## §8 — What I did not verify

- **My findings describe the working tree, not `main`.** `git status` shows **12 unmerged
  files** (`agent.py`, `sources.py`, `test_adk_wiring.py`, `test_agent_callback.py`,
  `test_sources.py`, `AGENTS.md`, `docs/ARCHITECTURE.md`, `patch-adk-devui-mobile.py`, the
  eval set, `test_adk_devui_patch.py`). Some findings may already be addressed there.
  **B8 and D3 sit in `tools/films.py` and `AGENTS.md`'s gotcha list — `AGENTS.md` *is* in the
  modified set, so confirm B8/D3 against `git diff` before approving MVP 2.** Review the diff
  before approving anything.
- **Not executed:** the agent, the MCP server, Docker, Langfuse. No model call was made.
- **`tools/` at runtime:** no browser, no Pillow, no `node`, no vault on this host. My read
  of those scripts is static only.
- **The published site** — I did not fetch `carloskafka.github.io/agent-testing/`.

**Executed:** the full 219-test suite (all green, 6.4 s); `./run.sh --help`; the B1 path
traversal and B2 frontmatter-injection probes against a real `save_summary_to_second_brain`
write; the `rouge_score` import; the `run.sh` menu arithmetic traced line by line.
