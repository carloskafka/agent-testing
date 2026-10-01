"""Durable-write tests: atomic replacement, and appends that survive concurrency.

``second_brain`` writes into a vault that is read by three other parties at once --
``obsidian-mcp`` indexing the same directory, the host's Obsidian, and a second
agent session running the same two tools. The filesystem code was written as if it
were the only writer, which is ``AGENTS.md`` known gap 9 and two live defects:

* **B9a, non-atomic writes.** ``_write`` was ``open(path, "w")`` followed by a
  write, so a crash or a competing writer between the two left a note missing its
  tail. Truncation is visible to the other two readers immediately, which is worse
  than the write simply not happening: ``obsidian-mcp`` indexes the truncated
  version and the user reads it in Obsidian.
* **B9b, lost updates.** Two tools did read-modify-write on
  ``Second Brain Index.md`` -- :func:`save_summary_to_second_brain` and
  :func:`log_conversation` -- so two sessions adding a line at the same moment
  each read the old body and the second write discarded the first. The chat log
  had the same read-rstrip-rewrite shape for a file that never needs a rewrite.

The atomicity test is the interesting one to read: it does not check that a
partial write *would* be visible, it makes one happen and asserts the file is
untouched. The concurrency tests are built so that removing the lock makes them
fail **by construction** rather than by luck -- the read is widened to hold the
critical section open long enough that an unsynchronised writer is guaranteed to
overlap another, because a test that passes unless a race happens to land is not a
guard on anything.

:meth:`find_cached_summary` is here too, because the same file is read on the path
of every single turn: it now reads a bounded prefix of each note rather than all
of it, and the test asserts the bound with the kernel's own read counter rather
than with a timing threshold.
"""

from __future__ import annotations

import builtins
import concurrent.futures
import contextlib
import multiprocessing
import os
import shutil
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pytest
from _helpers import _point_vault_at
from text_summarizer import second_brain
from text_summarizer.second_brain import (
    INDEX_HEADER,
    find_cached_summary,
    log_conversation,
    save_summary_to_second_brain,
    source_fingerprint,
)


def _vault(monkeypatch, tmp_path) -> Path:
    """Point the module at a temp vault that looks like one, and return it.

    ``_helpers._point_vault_at`` rather than a fourth copy: the globals it moves
    and the env var it sets are the whole reason a test's assertions land on the
    vault it created, and that has now been written three times by hand.

    ``Second Brain`` is created because ``_point_vault_at`` only makes the
    directory, and both the cache lookup and a real save need the marker.
    """
    vault = Path(_point_vault_at(monkeypatch, tmp_path))
    (vault / second_brain.BRAIN_DIR).mkdir(exist_ok=True)
    return vault


def _index_path(vault: Path) -> Path:
    return vault / f"{second_brain.INDEX_NAME}.md"


def _files(directory: Path) -> list[str]:
    """The names of the *files* in a directory, so a fixture directory is ignored."""
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def _chat_path(vault: Path) -> Path:
    return vault / second_brain.CHAT_LOG_DIR / f"{date.today().isoformat()}.md"


# --- B9a: a write that dies half way through -----------------------------------

#: Enough content that a write split across two calls is a visible difference, and
#: short enough that the fixture stays readable if the assertion ever prints it.
LONG_BODY = "- dogs are loyal\n" * 64


def _open_that_dies(monkeypatch, budget: int = 7) -> None:
    """Make every file this process opens for writing fail after ``budget`` bytes.

    Both openers are patched, not just the one the code under test happens to use
    today: ``tempfile.mkstemp`` hands back a descriptor that :func:`_write` wraps
    with ``os.fdopen``, while truncate-then-write went through ``open`` directly.
    Patching only one would make this test pass for the wrong reason under the
    implementation it is meant to fail against -- the fault would never be injected
    and the assertion would be about nothing.
    """
    real_open = builtins.open
    real_fdopen = os.fdopen

    class _DiesHalfway:
        """A write handle that truncates its first write and then dies."""

        def __init__(self, real, remaining: int) -> None:
            self._real = real
            self._remaining = remaining

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return self._real.__exit__(*exc_info)

        def write(self, data):
            self._real.write(data[: self._remaining])
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            self._real.flush()

        def fileno(self) -> int:
            return self._real.fileno()

    monkeypatch.setattr(builtins, "open", lambda *a, **k: _DiesHalfway(real_open(*a, **k), budget))
    monkeypatch.setattr(os, "fdopen", lambda *a, **k: _DiesHalfway(real_fdopen(*a, **k), budget))


def test_a_write_that_raises_midway_leaves_no_partial_file(monkeypatch, tmp_path):
    """A write that dies half way through must not touch the target at all.

    The fault is injected into the file objects :func:`_write` actually gets: each
    writes its first seven bytes and then raises, which is what a full disk or a
    killed process does. Under the staged write the target is never opened for
    writing, so it cannot be truncated and nothing is left behind; under
    ``open(path, "w")`` the truncation has already happened by the time the write
    starts, and the note is left holding seven bytes of the new content.
    """
    vault = _vault(monkeypatch, tmp_path)
    target = vault / "note.md"
    second_brain._write(str(target), LONG_BODY)
    before = target.read_bytes()

    _open_that_dies(monkeypatch)

    with pytest.raises(OSError):
        second_brain._write(str(target), LONG_BODY + "- and this never lands\n")

    assert target.read_bytes() == before
    # Nothing staged is left behind either: a dot-prefixed sibling is a temp file
    # whose replace never happened, and it is exactly what a reader walking the
    # vault would trip over on the next run.
    assert _files(vault) == ["note.md"]


def test_a_write_replaces_the_file_rather_than_truncating_it(monkeypatch, tmp_path):
    """The mechanism itself, with no fault injected: the inode changes.

    An atomic replace is a rename, so the target stops being the file that was
    there and becomes a different inode entirely. That is what makes the swap safe
    for a reader -- ``obsidian-mcp`` and Obsidian both hold notes open, and a reader
    with the old descriptor keeps reading a whole, consistent note rather than a
    file being emptied underneath it. A truncating write keeps the inode and does
    neither.
    """
    vault = _vault(monkeypatch, tmp_path)
    target = vault / "note.md"
    second_brain._write(str(target), LONG_BODY)
    before = target.stat().st_ino

    second_brain._write(str(target), "- replaced\n")

    assert target.stat().st_ino != before
    assert target.read_text(encoding="utf-8") == "- replaced\n"
    assert _files(vault) == ["note.md"]


def test_a_failed_write_does_not_delete_the_previous_note(monkeypatch, tmp_path):
    """Truncate-then-write also loses the old content, which is the worse half.

    Even where a partial write is not observable, ``"w"`` empties the file before
    the first byte of the new content is ready. A note is the only record of a
    turn the user had; "the tool errored" and "the note is gone" are different
    outcomes, and only the first is recoverable.
    """
    vault = _vault(monkeypatch, tmp_path)
    target = vault / "note.md"
    second_brain._write(str(target), LONG_BODY)

    # Fail the commit rather than the write, so the scenario is "every byte was
    # written and staged, and the swap did not happen".
    monkeypatch.setattr(
        second_brain.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("EPERM"))
    )

    with pytest.raises(OSError):
        second_brain._write(str(target), "- replacement\n")

    assert target.read_text(encoding="utf-8") == LONG_BODY
    assert _files(vault) == ["note.md"]


def test_a_write_keeps_the_permissions_of_the_file_it_replaces(monkeypatch, tmp_path):
    """``mkstemp`` creates at 0600; a plain ``open`` would have used the umask.

    Under Docker the container writes as one user and the host's Obsidian reads as
    another, so staging every note through a 0600 temp file and renaming that onto
    the target would quietly make the whole vault unreadable. The target's own mode
    wins, so a note someone deliberately tightened is not loosened by a rewrite.
    """
    vault = _vault(monkeypatch, tmp_path)
    target = vault / "note.md"
    second_brain._write(str(target), "- x\n")
    assert target.stat().st_mode & 0o777 == 0o644

    os.chmod(target, 0o640)
    second_brain._write(str(target), "- y\n")
    assert target.stat().st_mode & 0o777 == 0o640


# --- B9b: the index, which two tools append to ---------------------------------


def _pool_context():
    """``fork`` where it exists, and the platform default where it does not.

    The workers are correct either way -- :func:`_append_in_child` is a module
    function and sets ``VAULT_ROOT`` itself rather than inheriting it -- so this is
    purely about cost. Python 3.14 defaults to ``forkserver`` on Linux, which starts
    every worker as a *fresh interpreter*: eight of them re-importing this module
    and the whole ADK took 90 seconds. ``fork`` reuses the warm parent and takes
    milliseconds.
    """
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else None)


def _append_in_child(directory: str, line: str, barrier, header: str) -> bool:
    """One worker's whole job, at module level so ``spawn`` can pickle it.

    ``VAULT_ROOT`` is assigned rather than inherited: it is resolved at import time
    from the environment, and under ``spawn`` the child imports the module fresh
    instead of forking a parent that already had it repointed. Assigning it is the
    same move ``_point_vault_at`` makes in the parent, and it is what makes this
    test independent of the start method.
    """
    second_brain.VAULT_ROOT = directory
    barrier.wait(30)
    return second_brain._append_once(os.path.join(directory, "index.md"), line, header=header)


def _log_in_child(directory: str, marker: int, barrier) -> None:
    """One worker logging an exchange through the real entry point.

    Goes through ``log_conversation`` rather than calling ``_append_chat_entry``
    directly: the defect guarded here is in the create-or-append *decision*, which
    the entry point reaches by asking whether the day's note exists yet, and
    invoking the private helper directly would skip the very code under test.
    """
    second_brain.VAULT_ROOT = directory
    barrier.wait(30)
    second_brain.log_conversation(f"question {marker}", f"- answer {marker}")


def test_concurrent_appends_all_land(tmp_path):
    """Eight processes, one shared index, no lost entries.

    The target file is pre-loaded with 40,000 lines so each worker's read and its
    fsynced rewrite take milliseconds rather than microseconds. That is not a
    performance choice: it widens the read-modify-write window to a length the
    scheduler cannot miss on a two-core box, which is what turns "no entries lost"
    into a result rather than a hope. ``Manager().Barrier`` (rather than
    ``multiprocessing.Barrier``) because the latter can only be inherited, not
    pickled through a pool's task queue.
    """
    directory = tmp_path / "vault"
    directory.mkdir()
    target = directory / "index.md"
    target.write_text("# Index\n\n" + ("filler line\n" * 40_000), encoding="utf-8")

    workers = 8
    lines = [f"- [[2026-09-30 - worker {index}]]\n" for index in range(workers)]
    context = _pool_context()
    manager = context.Manager()
    barrier = manager.Barrier(workers)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            results = [
                future.result(timeout=120)
                for future in [
                    pool.submit(_append_in_child, str(directory), line, barrier, INDEX_HEADER)
                    for line in lines
                ]
            ]
    finally:
        manager.shutdown()

    assert results == [True] * workers
    body = target.read_text(encoding="utf-8")
    for line in lines:
        assert body.count(line) == 1, f"lost or duplicated {line!r}"
    assert body.startswith("# Index\n\n")
    assert len(body.splitlines()) == 2 + 40_000 + workers


def _run_concurrent_logs(directory: str, workers: int) -> str:
    """Eight processes each log one exchange into the same not-yet-created day."""
    context = _pool_context()
    manager = context.Manager()
    barrier = manager.Barrier(workers)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            for future in [
                pool.submit(_log_in_child, directory, index, barrier)
                for index in range(workers)
            ]:
                future.result(timeout=120)
    finally:
        manager.shutdown()
    day = datetime.now().strftime("%Y-%m-%d")
    return Path(directory, "Chat Log", f"{day}.md").read_text(encoding="utf-8")


def test_the_first_exchange_of_a_day_survives_every_concurrent_writer(tmp_path):
    """Eight sessions starting a fresh day must not clobber each other.

    **This is the case that made the lock necessary, and it is the first entry of
    the day rather than a later one.** ``_append_chat_entry`` decides between
    *creating* the day's note and *appending* to it by reading the tail, and the
    create branch is a full-file rewrite. Without a lock, all eight processes read
    the same empty tail, all eight concluded "create", and each replaced the file
    with only its own entry: measured at 2--5 of 8 surviving, varying per run,
    and the file always well-formed. Nothing looks wrong with the result, which is
    why this is a data-loss bug and not a crash.

    ``test_concurrent_appends_all_land`` cannot catch this. It exercises the
    *index* via ``_append_once``, which always locks, and it pre-creates its
    target file -- so the append branch is taken from the first line and the
    create branch is never reached.
    """
    directory = tmp_path / "vault"
    directory.mkdir()
    workers = 8

    body = _run_concurrent_logs(str(directory), workers)

    missing = [index for index in range(workers) if f"- answer {index}" not in body]
    assert not missing, (
        f"{len(missing)} of {workers} exchanges were lost: {missing}. Each writer "
        f"replaced the note with only its own entry, so the file is valid and short."
    )
    for index in range(workers):
        assert body.count(f"question {index}") == 1, f"duplicated entry {index}"
    # One header, written once -- not eight competing headers.
    assert body.count("# Chat Log -- ") == 1


def test_a_later_exchange_concurrent_with_the_first_still_all_land(tmp_path):
    """The same race on the *append* branch, which is the common case in production.

    Once the day's note exists every later exchange takes the pure-``O_APPEND``
    path, whose indivisibility does not depend on the lock -- so this is the case
    that was already safe, and it is asserted separately so that loosening the lock
    for the append branch would be a deliberate, visible act rather than a
    consequence of editing the create branch.
    """
    directory = tmp_path / "vault"
    directory.mkdir()
    day = datetime.now().strftime("%Y-%m-%d")
    seed = directory / "Chat Log" / f"{day}.md"
    seed.parent.mkdir(parents=True)
    seed.write_text("# Chat Log -- 2026-10-01\n\n**User:**\n\nseed\n", encoding="utf-8")
    workers = 8

    body = _run_concurrent_logs(str(directory), workers)

    assert "seed" in body
    missing = [index for index in range(workers) if f"- answer {index}" not in body]
    assert not missing, f"{len(missing)} of {workers} exchanges were lost: {missing}"
    # No interleaving: every exchange's own agent reply follows its own question.
    for index in range(workers):
        question = body.index(f"question {index}")
        answer = body.index(f"- answer {index}")
        assert question < answer, f"entry {index} was spliced out of order"


def test_the_index_gets_one_entry_per_note_however_many_calls_repeat_it(monkeypatch, tmp_path):
    """Idempotence is unchanged by the rewrite of this logic.

    Saving the same title twice is one note, so it is one index line -- the
    membership test is what makes a repeated save safe, and it now lives in one
    helper rather than in each of the two callers.
    """
    vault = _vault(monkeypatch, tmp_path)
    for _ in range(3):
        save_summary_to_second_brain(title="Repeat Probe", summary_content="- x", topics="Dogs")

    note_title = f"{date.today().isoformat()} - repeat-probe"
    index = _index_path(vault).read_text(encoding="utf-8")
    assert index.count(f"- [[{note_title}]]") == 1
    assert index.startswith("# Second Brain Index\n\nNone yet.\n")


def test_an_unchanged_index_is_not_rewritten(monkeypatch, tmp_path):
    """A save that adds nothing leaves the note's mtime alone.

    ``obsidian-mcp`` watches this directory. Rewriting an identical index on every
    call would wake it to reindex a file that did not change, on every turn of
    every session -- so the helper reports whether it appended and skips the write
    when it did not.
    """
    vault = _vault(monkeypatch, tmp_path)
    save_summary_to_second_brain(title="Mtime", summary_content="- x", topics="")
    first = _index_path(vault).stat().st_mtime_ns

    save_summary_to_second_brain(title="Mtime", summary_content="- y", topics="")
    assert _index_path(vault).stat().st_mtime_ns == first

    save_summary_to_second_brain(title="Other", summary_content="- z", topics="")
    assert _index_path(vault).stat().st_mtime_ns != first


def test_two_writers_racing_on_the_index_do_not_lose_an_entry(monkeypatch, tmp_path):
    """The read-modify-write is serialised, and the test proves it by construction.

    Threads, so the two writers are the two *tools* of one process: a real pair of
    sessions, minus the process startup. The read of the index is widened to 40 ms,
    which is what makes the assertion meaningful. Without the lock every thread is
    still inside its read when the first one commits, so the last write wins and
    all but one entry is gone -- deterministically, not sometimes. With the lock
    the wait happens *inside* the critical section, so the test pays the latency
    and the entries survive.

    A second assertion, on the sequence of writes rather than the final state: every
    write to the index must be a superset of the line set of the write before it.
    That is the property the lock buys, and it is checked directly rather than
    inferred from the outcome.
    """
    vault = _vault(monkeypatch, tmp_path)
    index_path = str(_index_path(vault))
    real_read = second_brain._read

    def _slow_index_read(path: str) -> str:
        body = real_read(path)
        if path == index_path:
            time.sleep(0.04)
        return body

    monkeypatch.setattr(second_brain, "_read", _slow_index_read)

    committed: list[set[str]] = []
    real_write = second_brain._write

    def _recording_write(path: str, content: str) -> None:
        if path == index_path:
            committed.append({line for line in content.splitlines() if line.startswith("- [[")})
        real_write(path, content)

    monkeypatch.setattr(second_brain, "_write", _recording_write)

    workers = 6
    titles = [f"Racer {index}" for index in range(workers)]
    start = threading.Barrier(workers)
    errors: list[BaseException] = []

    def _save(title: str) -> None:
        try:
            start.wait(30)
            save_summary_to_second_brain(title=title, summary_content="- x", topics="")
        except BaseException as exc:  # a worker dying must not look like a lost entry
            errors.append(exc)

    threads = [threading.Thread(target=_save, args=(title,)) for title in titles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, errors
    index = _index_path(vault).read_text(encoding="utf-8")
    for title in titles:
        line = f"- [[{date.today().isoformat()} - {second_brain._safe_name(title)}]]"
        assert index.count(line) == 1, f"{line!r} missing or duplicated"
    assert len(committed) == workers
    for earlier, later in zip(committed, committed[1:], strict=False):
        assert earlier <= later, f"a write dropped lines: {earlier - later}"


def test_an_index_append_waits_for_a_lock_another_writer_holds(monkeypatch, tmp_path):
    """Mutual exclusion, asserted on the thing the lock is for.

    The holder takes :func:`_exclusive`, *says so once it is inside*, and keeps the
    lock while a second thread calls :func:`_append_once`; the second writer's line
    must not appear until the holder lets go. This is the only assertion here that
    depends on timing -- the blocked thread cannot report anything before it is
    released, so all the test can do is wait and then insist nothing has happened --
    and it is deliberately redundant with the test above, which fails by
    construction without the lock. This one says why it matters.
    """
    vault = _vault(monkeypatch, tmp_path)
    target = str(_index_path(vault))
    second_brain._write(target, INDEX_HEADER)

    held = threading.Event()
    release = threading.Event()
    done = threading.Event()
    started = threading.Event()

    def _blocked_writer() -> None:
        started.set()
        second_brain._append_once(target, "- [[blocked]]\n", header=INDEX_HEADER)
        done.set()

    holder_done = threading.Event()

    def _holder() -> None:
        with second_brain._exclusive(target):
            held.set()
            release.wait(30)
        holder_done.set()

    holder = threading.Thread(target=_holder)
    holder.start()
    assert held.wait(30), "the holder never got the lock, so nothing was being tested"
    writer = threading.Thread(target=_blocked_writer)
    writer.start()
    assert started.wait(30)
    time.sleep(0.3)
    assert not done.is_set(), "the append completed while the lock was held"
    assert "- [[blocked]]" not in _index_path(vault).read_text(encoding="utf-8")

    release.set()
    assert holder_done.wait(30)
    assert done.wait(30)
    writer.join(timeout=30)
    holder.join(timeout=30)
    assert "- [[blocked]]" in _index_path(vault).read_text(encoding="utf-8")


# --- the chat log -------------------------------------------------------------


def test_the_chat_log_appends_rather_than_rewrites(monkeypatch, tmp_path):
    """Two exchanges in one day are two entries in one note.

    The old implementation read the whole day's note and wrote it back, so the
    cost of a turn grew with the length of the day and two sessions could each
    discard the other's entry. It also re-derived the ``# Chat Log`` heading from
    whatever it read, which is how a note could end up with two.
    """
    vault = _vault(monkeypatch, tmp_path)
    day = date.today().isoformat()

    log_conversation("first question", "first answer")
    log_conversation("second question", "second answer")

    body = _chat_path(vault).read_text(encoding="utf-8")
    assert body.count(f"# Chat Log -- {day}") == 1
    assert body.count("## ") == 2
    for expected in ("first question", "first answer", "second question", "second answer"):
        assert body.count(expected) == 1
    assert body.count("**User:**") == 2
    assert body.count("**Agent:**") == 2
    # The shape the format is for: the two entries are separated by exactly one
    # blank line, which is what ``rstrip() + "\n\n"`` used to produce and what a
    # plain append would drift away from.
    assert body.split("**Agent:**\n\nfirst answer")[1].startswith("\n\n## ")


def test_the_chat_log_is_appended_without_being_read_whole(monkeypatch, tmp_path):
    """The append path does not read the file it appends to.

    Reading a growing file to append one entry is the defect, so the assertion is
    on the bytes read rather than on the result: :func:`_read` is replaced with a
    tripwire for the day's note. The bounded tail read is a different function and
    is allowed -- reading the last 4 KiB to learn how the previous entry ended is
    not the read-modify-write.
    """
    vault = _vault(monkeypatch, tmp_path)
    chat = str(_chat_path(vault))
    real_read = second_brain._read
    reads: list[str] = []

    def _recording_read(path: str) -> str:
        if path == chat:
            reads.append(path)
        return real_read(path)

    monkeypatch.setattr(second_brain, "_read", _recording_read)
    for index in range(3):
        log_conversation(f"question {index}", f"answer {index}")

    assert reads == []


def test_the_chat_log_note_is_not_recreated_per_call(monkeypatch, tmp_path):
    """A second call must not re-derive the heading from a note it just wrote.

    The old code seeded the heading whenever the body it read was empty and then
    rewrote the file, so the two conditions were never independent. The heading is
    written by the create path only; this pins that the append path never emits
    one.
    """
    vault = _vault(monkeypatch, tmp_path)
    log_conversation("q", "a")
    path = _chat_path(vault)
    inode = path.stat().st_ino

    log_conversation("q2", "a2")
    assert path.stat().st_ino == inode
    assert path.read_text(encoding="utf-8").count("\n# Chat Log") == 0


# --- the cache lookup's read bound --------------------------------------------


def _rchar() -> int | None:
    """Bytes this process has read, per the kernel, or ``None`` if unavailable."""
    try:
        with open("/proc/self/io", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def test_a_cache_lookup_does_not_match_a_fingerprint_in_a_note_body(monkeypatch, tmp_path):
    """The body is not frontmatter, and a note that says otherwise is not a hit.

    The fingerprint line is the cache key. If the search ran over the whole note,
    any note that quoted one -- a summary *about* this system, a pasted log line --
    would claim a turn it never served, and the user would be shown a note that has
    nothing to do with what they asked.
    """
    vault = _vault(monkeypatch, tmp_path)
    prompt = "Summarize: a question about the vault itself."
    decoy = source_fingerprint(prompt)
    (vault / second_brain.BRAIN_DIR / "2026-09-29 - decoy.md").write_text(
        '---\ntags:\n  - "Meta"\ndate: 2026-09-29\n---\n\n'
        f"- The fingerprint for this turn is source_fingerprint: {decoy}\n",
        encoding="utf-8",
    )

    assert find_cached_summary(prompt) is None


def test_a_cache_lookup_reads_a_bounded_prefix_of_each_note(monkeypatch, tmp_path):
    """The scan's cost is bounded by note *count*, not by note *size*.

    The lookup runs on the first model call of every turn, so a note that grew
    without bound would tax every turn in the vault.

    The budget is **calibrated against the filesystem in the same test** rather
    than hard-coded, because ``/proc/self/io``'s ``rchar`` does not count the bytes
    a read asked for: it charges a whole readahead window per syscall, and on this
    box that is 128 KiB whether the call requests one byte or 64 KiB. So the two
    reference costs are measured here -- one bounded read of the huge note, and one
    read of all of it -- and the scan has to cost about the first, not the second.
    A fixture whose whole-file read is not measurably dearer than its prefix read
    cannot tell a correct implementation from a broken one, so it skips rather than
    passing for the wrong reason.
    """
    if _rchar() is None:
        pytest.skip("no /proc/self/io on this platform")

    vault = _vault(monkeypatch, tmp_path)
    brain = vault / second_brain.BRAIN_DIR
    prompt = "Summarize: a question nothing in the vault answers."

    small = brain / "2026-09-28 - small.md"
    small.write_text('---\ntags:\n  - "Size"\ndate: 2026-09-28\n---\n\n- short\n', encoding="utf-8")
    huge = brain / "2026-09-29 - huge.md"
    huge.write_text(
        '---\ntags:\n  - "Size"\ndate: 2026-09-29\n---\n\n- filler\n' + ("x" * 79 + "\n") * 52_000,
        encoding="utf-8",
    )
    assert huge.stat().st_size > 4_000_000

    bounded_read = _read_cost(lambda: second_brain._read_prefix(str(huge), 8 * 1024))
    whole_read = _read_cost(lambda: second_brain._read(str(huge)))
    if whole_read < 8 * bounded_read:
        pytest.skip(f"read accounting cannot separate {bounded_read} from {whole_read}")

    with _parked(small):
        scan = _read_cost(lambda: find_cached_summary(prompt))

    assert scan <= 2 * bounded_read + 16 * 1024, (
        f"scanning a {huge.stat().st_size} byte note cost {scan} bytes, against "
        f"{bounded_read} for a bounded read of it and {whole_read} for the whole of it"
    )


@contextlib.contextmanager
def _parked(note: Path):
    """Move a note out of the scanned directory for the duration of the block.

    Moved into a sibling directory rather than renamed, and the reason is a bug this
    test would otherwise have shipped: the scan is a non-recursive ``os.listdir``
    filtered on ``.md``, so a note renamed to ``hidden-x.md`` is still read, the
    measurement would be of the same thing either way, and the ratio would be 1.0
    for a correct *and* a broken implementation.
    """
    brain = Path(second_brain.VAULT_ROOT) / second_brain.BRAIN_DIR
    parked = brain.parent / "parked"
    parked.mkdir(exist_ok=True)
    shutil.move(str(note), str(parked / note.name))
    try:
        yield
    finally:
        shutil.move(str(parked / note.name), str(note))


def _read_cost(what) -> int:
    """Bytes the kernel charged the process for running ``what``."""
    before = _rchar()
    result = what()
    cost = _rchar() - before
    assert result is None or cost >= 0
    return cost


def test_a_cache_hit_still_replays_the_whole_summary(monkeypatch, tmp_path):
    """Bounding the *scan* must not bound what a hit returns.

    The prefix is the filter; the note that matched is read in full to produce the
    summary, because the summary is what the user is shown and a truncated one
    would be a silently wrong answer. One extra read, on one file, on a turn that
    ends there.
    """
    vault = _vault(monkeypatch, tmp_path)
    prompt = "Summarize: a long answer."
    bullets = "\n".join(f"- point {index}" for index in range(2000))
    (vault / second_brain.BRAIN_DIR / "2026-09-29 - long.md").write_text(
        '---\ntags:\n  - "Long"\ndate: 2026-09-29\n'
        f"source_fingerprint: {source_fingerprint(prompt)}\n---\n\n{bullets}\n\n## Related\n",
        encoding="utf-8",
    )

    replayed = find_cached_summary(prompt)

    assert replayed is not None
    assert replayed == bullets
    assert "## Related" not in replayed


def test_a_note_whose_fingerprint_is_far_into_a_long_frontmatter_is_not_missed(
    monkeypatch, tmp_path
):
    """The bound is generous, and this is the shape that would break it.

    Every frontmatter this module writes puts the fingerprint in the first few
    hundred bytes, so 8 KiB has ample room. A hand-written note could order its keys
    differently, and the failure would be a silent cache miss -- the turn simply
    re-summarises -- which is why the size of the window is pinned rather than
    assumed.
    """
    vault = _vault(monkeypatch, tmp_path)
    prompt = "Summarize: the last key in the block."
    padding = "".join(f'  - "topic {index}"\n' for index in range(400))
    (vault / second_brain.BRAIN_DIR / "2026-09-29 - reordered.md").write_text(
        f"---\ntags:\n{padding}date: 2026-09-29\n"
        f"source_fingerprint: {source_fingerprint(prompt)}\n---\n\n- the answer\n\n## Related\n",
        encoding="utf-8",
    )

    assert find_cached_summary(prompt) == "- the answer"
