#!/usr/bin/env python3
"""SICA Stop Hook - Self-Improving Coding Agent loop controller.

Intercepts Claude's exit attempts to:
1. Run benchmark command
2. Archive results
3. Analyze failures
4. Continue with improvement prompt if not complete
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from debug import dbg
from paths import (
    get_state_file,
    is_sica_session,
    list_active_configs,
    make_runtime_marker,
)


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def read_state() -> SicaState | None:
    """Read active SICA state if exists."""
    active = list_active_configs()
    if not active:
        return None
    if len(active) > 1:
        log(f"SICA: Warning - multiple active configs: {', '.join(active)}")
        log(f"SICA: Using first: {active[0]}")
    config_name = active[0]
    try:
        return SicaState.load(get_state_file(config_name))
    except (json.JSONDecodeError, OSError, FileNotFoundError, ValueError):
        return None


def save_state(state: SicaState) -> None:
    """Save SICA state."""
    state.save(get_state_file(state.config_name))


def complete_run(state: SicaState) -> None:
    """Mark run as complete by setting status and timestamp."""
    state.status = "complete"
    state.completed_at = datetime.now(timezone.utc).isoformat()
    save_state(state)


def run_benchmark(
    command: str,
    timeout: int = 120,
) -> dict[str, str | int | float]:
    """Run benchmark command and parse results.

    Args:
        command: Shell command to run (already interpolated)
        timeout: Max seconds to wait

    Returns:
        Dict with exit_code, stdout, stderr, duration, passed, failed, errors, score.
    """
    start = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

    except subprocess.TimeoutExpired as e:
        duration = timeout
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        stderr += f"\n[TIMEOUT after {timeout}s]"
        exit_code = -1
    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        stdout = ""
        stderr = str(e)
        exit_code = -1

    passed, failed, errors = parse_test_output(stdout + stderr)
    total = passed + failed + errors
    score = passed / total if total > 0 else (1.0 if exit_code == 0 else 0.0)

    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration": duration,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "score": score,
    }


def parse_test_output(output: str) -> tuple[int, int, int]:
    """Parse test output for pass/fail/error counts.

    Supports: pytest, jest, mocha, go test, cargo test
    """
    passed = failed = errors = 0

    # pytest summary line
    passed_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
    failed_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
    error_match = re.search(r"(\d+)\s+error", output, re.IGNORECASE)

    if passed_match:
        passed = int(passed_match.group(1))
    if failed_match:
        failed = int(failed_match.group(1))
    if error_match:
        errors = int(error_match.group(1))

    if passed > 0 or failed > 0 or errors > 0:
        return passed, failed, errors

    # Jest: "Tests: X passed, Y failed"
    jest_match = re.search(
        r"Tests:\s*(?:(\d+)\s+passed)?(?:,\s*(\d+)\s+failed)?", output, re.IGNORECASE
    )
    if jest_match:
        passed = int(jest_match.group(1) or 0)
        failed = int(jest_match.group(2) or 0)
        return passed, failed, errors

    # Go test: "ok" or "FAIL"
    go_ok = len(re.findall(r"^ok\s+", output, re.MULTILINE))
    go_fail = len(re.findall(r"^FAIL\s+", output, re.MULTILINE))
    if go_ok or go_fail:
        return go_ok, go_fail, 0

    # Cargo test: "X passed; Y failed"
    cargo_match = re.search(r"(\d+)\s+passed;\s+(\d+)\s+failed", output)
    if cargo_match:
        return int(cargo_match.group(1)), int(cargo_match.group(2)), 0

    return passed, failed, errors


def archive_iteration(
    state: SicaState,
    benchmark_result: dict[str, str | int | float],
    transcript_path: str,
) -> Path:
    """Save iteration results to archive."""
    run_dir = Path(state.run_dir)
    iteration = state.iteration
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark results
    benchmark_file = iter_dir / "benchmark.json"
    benchmark_file.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": state.benchmark_cmd,
                **benchmark_result,
            },
            indent=2,
        )
    )

    # Save stdout/stderr
    stdout = benchmark_result.get("stdout", "")
    stderr = benchmark_result.get("stderr", "")
    if stdout:
        (iter_dir / "stdout.txt").write_text(str(stdout))
    if stderr:
        (iter_dir / "stderr.txt").write_text(str(stderr))

    # Save git diff
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, timeout=10)
        if diff.stdout:
            (iter_dir / "changes.diff").write_text(diff.stdout)
    except Exception as e:
        log(f"SICA: Warning - could not save git diff: {e}")

    # Extract last assistant message from transcript
    try:
        if transcript_path and Path(transcript_path).exists():
            with open(transcript_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if '"role":"assistant"' in line:
                    data = json.loads(line)
                    texts = [
                        c.get("text", "")
                        for c in data.get("message", {}).get("content", [])
                        if c.get("type") == "text"
                    ]
                    if texts:
                        (iter_dir / "summary.txt").write_text("\n".join(texts)[:5000])
                    break
    except Exception:
        pass

    return iter_dir


def get_archive_summary(state: SicaState, top_n: int = 10) -> str:
    """Generate compact CSV summary of top N iterations by score."""
    run_dir = Path(state.run_dir)
    entries: list[tuple[float, int, int, int, int]] = []

    for i in range(state.iteration):
        iter_dir = run_dir / f"iteration_{i}"
        benchmark_file = iter_dir / "benchmark.json"

        if benchmark_file.exists():
            try:
                d = json.loads(benchmark_file.read_text())
                score = d.get("score", 0)
                entries.append(
                    (
                        score,
                        i,
                        d.get("passed", 0),
                        d.get("failed", 0),
                        d.get("errors", 0),
                    )
                )
            except Exception:
                pass

    if not entries:
        return "No previous iterations."

    entries.sort(reverse=True)
    entries = entries[:top_n]
    rows = ["#,score,pass,fail,err"]
    for score, i, p, f, e in entries:
        rows.append(f"{i},{score:.2f},{p},{f},{e}")

    return "\n".join(rows)


def extract_failures(benchmark_result: dict[str, str | int | float]) -> str:
    """Extract failure details from benchmark output."""
    stdout = str(benchmark_result.get("stdout", ""))
    stderr = str(benchmark_result.get("stderr", ""))
    output = stdout + "\n" + stderr

    failures: list[str] = []

    # pytest FAILED lines
    for match in re.finditer(r"FAILED\s+(.+?)(?:\s+-|$)", output, re.MULTILINE):
        failures.append(f"- {match.group(1)}")

    # pytest short test summary
    summary_match = re.search(r"=+ short test summary info =+\n(.*?)(?:=+|$)", output, re.DOTALL)
    if summary_match:
        failures.append(summary_match.group(1).strip())

    # AssertionError details
    for match in re.finditer(r"AssertionError:.*", output):
        failures.append(f"- {match.group(0)[:200]}")

    # Error tracebacks
    for match in re.finditer(r"^E\s+(.+)$", output, re.MULTILINE):
        failures.append(f"- {match.group(1)[:200]}")

    if failures:
        return "\n".join(failures[:20])

    # Fallback: last 50 lines
    exit_code = benchmark_result.get("exit_code", 0)
    if exit_code != 0:
        lines = output.strip().split("\n")[-50:]
        return "\n".join(lines)

    return "No specific failures extracted."


def generate_improvement_prompt(
    state: SicaState,
    benchmark_result: dict[str, str | int | float],
    iter_dir: Path,
) -> str:
    """Generate compact improvement prompt with archive analysis."""
    failures = extract_failures(benchmark_result)
    top_n = 10
    archive_summary = get_archive_summary(state, top_n=top_n)
    original = state.prompt
    run_dir = state.run_dir

    passed = benchmark_result.get("passed", 0)
    failed = benchmark_result.get("failed", 0)

    journal_path = f"{run_dir}/journal.md"
    parts = [
        f"## Results: {passed}✓ {failed}✗",
        "",
    ]

    # Only show journal reminder if this iteration had strategy code changes (not meta files)
    changes_file = iter_dir / "changes.diff"
    has_strategy_changes = False
    if changes_file.exists():
        try:
            diff_content = changes_file.read_text()
            # Exclude meta files: only count changes to strategy code
            lines = diff_content.split("\n")
            for line in lines:
                if line.startswith("diff --git"):
                    # Extract filename
                    if (
                        "state.json" not in line
                        and ".sica" not in line
                        and "plugins/sica" not in line
                    ):
                        has_strategy_changes = True
                        break
        except Exception:
            pass

    if has_strategy_changes:
        parts.append(f"**JOURNAL:** Update {journal_path} NOW (plan → result)")
        parts.append("")

    parts.extend(
        [
            f"## Top {top_n} Iterations",
            archive_summary,
            "",
            "## Failures",
            failures,
            "",
            "## Archive",
            f"If stuck, read {run_dir}/iteration_N/changes.diff to restore a better approach.",
        ]
    )

    if original:
        parts.extend(["", "## Task", original])

    return "\n".join(parts)


def main() -> None:
    # Fast exit: no active config = not a SICA session
    state = read_state()
    if not state:
        sys.exit(0)

    # Read hook input
    # https://code.claude.com/docs/en/hooks#stop
    # https://code.claude.com/docs/en/hooks#stop-and-subagentstop-input
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    hook_event_name = hook_input.get("hook_event_name", "")
    session_id = hook_input.get("session_id", "")
    permission_mode = hook_input.get("permission_mode", "")
    stop_hook_active = hook_input.get("stop_hook_active", "")
    cwd = hook_input.get("cwd", "")
    transcript_path = hook_input.get("transcript_path", "")
    dbg(f"hook event: {hook_event_name}, session_id: {session_id}, permission_mode: {permission_mode}, stop_hook_active: {stop_hook_active}, cwd: {cwd}, transcript: {transcript_path}")

    # Also constrained via "matcher" in hooks.json
    if hook_event_name != "Stop":
        sys.exit(0)

    # Fast exit: wrong session (user running CC for other work)
    if not is_sica_session(transcript_path, state.config_name, state.run_id):
        sys.exit(0)

    # Now we know it's a SICA session - start logging
    dbg()
    dbg("=== STOP HOOK ===")
    dbg(f"CWD: {os.getcwd()}")
    dbg(f"State: iter={state.iteration}, max={state.max_iterations}")

    # Validate state
    if not isinstance(state.iteration, int):
        dbg("State corrupted - exiting")
        log("SICA: State corrupted")
        get_state_file(state.config_name).unlink(missing_ok=True)
        sys.exit(0)

    max_iter = max(0, state.max_iterations)

    # Check max iterations
    if state.iteration >= max_iter:
        dbg(f"Max iter reached: {state.iteration} >= {max_iter}")
        log(f"SICA COMPLETE: Max iterations ({max_iter}) reached")
        log(f"  Config: {state.config_name} | Run: {state.run_id}")
        log(f"  Last score: {state.last_score} | Target: {state.target_score}")
        log(f"  Run dir: {state.run_dir}")
        complete_run(state)
        sys.exit(0)

    # Check transcript for promise
    promise_detected = False
    if transcript_path and Path(transcript_path).exists():
        try:
            content = Path(transcript_path).read_text()
            promise = state.completion_promise
            if promise and f"<promise>{promise}</promise>" in content:
                promise_detected = True
                dbg("Promise detected in transcript")
        except Exception as e:
            dbg(f"Transcript read error: {e}")

    dbg("Running benchmark...")

    # Interpolate params into benchmark command (shell-escaped)
    benchmark_cmd = state.interpolate(state.benchmark_cmd, shell_escape=True)
    dbg(f"Cmd: {benchmark_cmd[:80]}...")

    log(f"SICA: Running benchmark (iteration {state.iteration})...")
    benchmark_result = run_benchmark(benchmark_cmd, timeout=state.benchmark_timeout)

    score = benchmark_result.get("score", 0.0)
    passed = benchmark_result.get("passed", 0)
    failed = benchmark_result.get("failed", 0)
    errors = benchmark_result.get("errors", 0)

    dbg(f"Benchmark done: score={score}, p={passed}, f={failed}")
    log(f"SICA: Score={score:.2f} (passed={passed}, failed={failed}, errors={errors})")

    # Archive results
    iter_dir = archive_iteration(state, benchmark_result, transcript_path)

    # Track recent scores for convergence
    state.recent_scores.append(round(float(score), 4))
    state.recent_scores = state.recent_scores[-10:]

    # Check convergence (5 consecutive identical scores)
    if len(state.recent_scores) >= 5 and len(set(state.recent_scores[-5:])) == 1:
        log(f"SICA COMPLETE: Converged at score {state.recent_scores[0]:.2f}")
        log(f"  Config: {state.config_name} | Run: {state.run_id}")
        log(f"  Iterations: {state.iteration} | Target: {state.target_score}")
        log(f"  Run dir: {state.run_dir}")
        complete_run(state)
        sys.exit(0)

    # Check target reached
    if float(score) >= state.target_score:
        log(f"SICA COMPLETE: Target score reached! ({score:.2f} >= {state.target_score})")
        log(f"  Config: {state.config_name} | Run: {state.run_id}")
        log(f"  Iterations: {state.iteration}")
        log(f"  Run dir: {state.run_dir}")
        complete_run(state)
        sys.exit(0)

    # Promise detected but tests failing
    if promise_detected:
        log("SICA: Promise detected but tests still failing! Continuing...")

    # Update state
    state.iteration += 1
    state.last_score = float(score)
    save_state(state)

    # Generate improvement prompt
    improvement_prompt = generate_improvement_prompt(state, benchmark_result, iter_dir)

    # Build system message with marker at end (persists across iterations)
    run_dir = state.run_dir
    promise = state.completion_promise
    marker = make_runtime_marker(state.config_name, state.run_id)
    system_msg = (
        f"SICA iter {state.iteration} | "
        f"Score: {score:.2f}/{state.target_score} | "
        f"RULES: ONE fix→exit immediately. NO manual tests. NO test changes. "
        f"Read/update {run_dir}/journal.md. Done: <promise>{promise}</promise> "
        f"{marker}"
    )

    # Block exit and continue
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": improvement_prompt,
                "systemMessage": system_msg,
            }
        )
    )


if __name__ == "__main__":
    main()
