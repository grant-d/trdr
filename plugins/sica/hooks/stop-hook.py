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
from paths import find_active_config, get_state_file


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def read_state() -> SicaState | None:
    """Read active SICA state if exists."""
    config_name = find_active_config()
    if not config_name:
        return None
    try:
        return SicaState.load(get_state_file(config_name))
    except (json.JSONDecodeError, OSError, FileNotFoundError, ValueError):
        return None


def save_state(state: SicaState) -> None:
    """Save SICA state."""
    state.save(get_state_file(state.config_name))


def archive_state(state: SicaState) -> None:
    """Move state to run archive and delete active state."""
    run_dir = Path(state.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    final_state = run_dir / "final_state.json"
    final_state.write_text(json.dumps(state.to_dict(), indent=2))
    get_state_file(state.config_name).unlink(missing_ok=True)


def run_benchmark(
    command: str,
    timeout: int = 300,
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
        r"Tests:\s*(?:(\d+)\s+passed)?(?:,\s*(\d+)\s+failed)?",
        output, re.IGNORECASE
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
    benchmark_file.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": state.benchmark_cmd,
        **benchmark_result,
    }, indent=2))

    # Save stdout/stderr
    stdout = benchmark_result.get("stdout", "")
    stderr = benchmark_result.get("stderr", "")
    if stdout:
        (iter_dir / "stdout.txt").write_text(str(stdout))
    if stderr:
        (iter_dir / "stderr.txt").write_text(str(stderr))

    # Save git diff
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
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
                entries.append((
                    score,
                    i,
                    d.get("passed", 0),
                    d.get("failed", 0),
                    d.get("errors", 0),
                ))
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
    summary_match = re.search(
        r"=+ short test summary info =+\n(.*?)(?:=+|$)",
        output, re.DOTALL
    )
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
) -> str:
    """Generate compact improvement prompt with archive analysis."""
    failures = extract_failures(benchmark_result)
    top_n = 10
    archive_summary = get_archive_summary(state, top_n=top_n)
    original = state.prompt
    run_dir = state.run_dir

    passed = benchmark_result.get("passed", 0)
    failed = benchmark_result.get("failed", 0)

    parts = [
        f"## Results: {passed}✓ {failed}✗",
        "",
        f"## Top {top_n} Iterations",
        archive_summary,
        "",
        "## Failures",
        failures,
        "",
        "## Journal",
        f"MUST update {run_dir}/journal.md:",
        "- BEFORE: what you'll try and why",
        "- AFTER: what happened, what you learned",
        "",
        "## Archive",
        f"If stuck, read {run_dir}/iteration_N/changes.diff to restore a better approach.",
    ]

    if original:
        parts.extend(["", "## Task", original])

    return "\n".join(parts)


def main() -> None:
    # DEBUG: Write to temp file to trace full execution
    debug_file = Path("/tmp/sica-stop-hook-debug.txt")
    debug_lines = [f"Time: {datetime.now(timezone.utc).isoformat()}"]

    def dbg(msg: str) -> None:
        debug_lines.append(msg)
        debug_file.write_text("\n".join(debug_lines))

    try:
        dbg(f"CWD: {os.getcwd()}")
        active = find_active_config()
        dbg(f"Active config: {active}")
        state_path = get_state_file(active) if active else None
        dbg(f"State path: {state_path}")
        state_exists = state_path.exists() if state_path else False
        dbg(f"State exists: {state_exists}")
    except Exception as e:
        dbg(f"Init error: {e}")

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        dbg(f"Hook input keys: {list(hook_input.keys())}")
    except json.JSONDecodeError:
        hook_input = {}
        dbg("No hook input")

    transcript_path = hook_input.get("transcript_path", "")
    dbg(f"Transcript: {transcript_path}")

    # Check for active SICA loop
    state = read_state()
    if not state:
        dbg("No state - exiting")
        sys.exit(0)

    dbg(f"State loaded: iter={state.iteration}, max={state.max_iterations}")

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
        log(f"SICA: Max iterations ({max_iter}) reached.")
        archive_state(state)
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
    archive_iteration(state, benchmark_result, transcript_path)

    # Track recent scores for convergence
    state.recent_scores.append(round(float(score), 4))
    state.recent_scores = state.recent_scores[-10:]

    # Check convergence
    if len(state.recent_scores) == 10 and len(set(state.recent_scores)) == 1:
        log(f"SICA: Converged at score {state.recent_scores[0]:.2f}")
        archive_state(state)
        sys.exit(0)

    # Check target reached
    if float(score) >= state.target_score:
        log(f"SICA: Target score ({state.target_score}) reached!")
        archive_state(state)
        sys.exit(0)

    # Promise detected but tests failing
    if promise_detected:
        log("SICA: Promise detected but tests still failing! Continuing...")

    # Update state
    state.iteration += 1
    state.last_score = float(score)
    save_state(state)

    # Generate improvement prompt
    improvement_prompt = generate_improvement_prompt(state, benchmark_result)

    # Build system message
    run_dir = state.run_dir
    promise = state.completion_promise
    system_msg = (
        f"SICA iter {state.iteration} | "
        f"Score: {score:.2f}/{state.target_score} | "
        f"RULES: ONE fix→exit immediately. NO manual tests. NO test changes. "
        f"Read/update {run_dir}/journal.md. Done: <promise>{promise}</promise>"
    )

    # Block exit and continue
    print(json.dumps({
        "decision": "block",
        "reason": improvement_prompt,
        "systemMessage": system_msg,
    }))


if __name__ == "__main__":
    main()
