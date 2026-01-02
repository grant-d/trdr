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


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def read_state_file() -> dict | None:
    """Read the SICA state file if it exists."""
    state_file = Path(".sica/current_run.json")
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_state(state: dict) -> None:
    """Save the SICA state file."""
    state_file = Path(".sica/current_run.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def _archive_state(state: dict) -> None:
    """Move state file to run archive for post-mortem review."""
    state_file = Path(".sica/current_run.json")
    if state_file.exists() and state.get("run_dir"):
        run_dir = Path(state["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        final_state = run_dir / "final_state.json"
        state_file.rename(final_state)
    else:
        state_file.unlink(missing_ok=True)


def run_benchmark(command: str, timeout: int = 300) -> dict:
    """Run the benchmark command and parse results.

    Returns dict with:
        - exit_code: int
        - stdout: str
        - stderr: str
        - duration: float
        - passed: int (extracted from output)
        - failed: int (extracted from output)
        - errors: int (extracted from output)
        - score: float (0.0-1.0)
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

    # Parse test results from common formats
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

    # pytest summary line: "X failed, Y passed" or "Y passed, X failed" or "Y passed"
    # Look for individual counts anywhere in the output
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

    # Jest: "Tests: X passed, Y failed, Z total"
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


def archive_iteration(state: dict, benchmark_result: dict, transcript_path: str) -> Path:
    """Save iteration results to archive."""
    run_dir = Path(state["run_dir"])
    iteration = state["iteration"]
    iter_dir = run_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark results
    benchmark_file = iter_dir / "benchmark.json"
    benchmark_file.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": state["benchmark_cmd"],
        **benchmark_result,
    }, indent=2))

    # Save stdout/stderr separately for readability
    if benchmark_result["stdout"]:
        (iter_dir / "stdout.txt").write_text(benchmark_result["stdout"])
    if benchmark_result["stderr"]:
        (iter_dir / "stderr.txt").write_text(benchmark_result["stderr"])

    # Save git diff of changes
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if diff.stdout:
            (iter_dir / "changes.diff").write_text(diff.stdout)
    except Exception:
        pass

    # Extract last assistant message from transcript for summary
    try:
        if transcript_path and Path(transcript_path).exists():
            with open(transcript_path) as f:
                lines = f.readlines()
            # Find last assistant message
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


def get_archive_summary(state: dict, top_n: int = 10) -> str:
    """Generate compact CSV summary of top N iterations by score.

    Args:
        state: SICA state dict
        top_n: Number of best iterations to show
    """
    run_dir = Path(state["run_dir"])
    entries = []

    for i in range(state["iteration"]):
        iter_dir = run_dir / f"iteration_{i}"
        benchmark_file = iter_dir / "benchmark.json"

        if benchmark_file.exists():
            try:
                d = json.loads(benchmark_file.read_text())
                score = d.get('score', 0)
                entries.append((score, i, d.get('passed', 0), d.get('failed', 0), d.get('errors', 0)))
            except Exception:
                pass

    if not entries:
        return "No previous iterations."

    # Sort by score descending, take top N
    entries.sort(reverse=True)
    entries = entries[:top_n]
    rows = ["#,score,pass,fail,err"]
    for score, i, p, f, e in entries:
        rows.append(f"{i},{score:.2f},{p},{f},{e}")

    return "\n".join(rows)


def extract_failures(benchmark_result: dict) -> str:
    """Extract failure details from benchmark output."""
    output = benchmark_result["stdout"] + "\n" + benchmark_result["stderr"]

    # Look for common failure patterns
    failures = []

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

    # Error tracebacks (last line of each)
    for match in re.finditer(r"^E\s+(.+)$", output, re.MULTILINE):
        failures.append(f"- {match.group(1)[:200]}")

    if failures:
        return "\n".join(failures[:20])  # Limit to 20 failures

    # Fallback: last 50 lines of output if exit code non-zero
    if benchmark_result["exit_code"] != 0:
        lines = output.strip().split("\n")[-50:]
        return "\n".join(lines)

    return "No specific failures extracted."


def generate_improvement_prompt(state: dict, benchmark_result: dict) -> str:
    """Generate compact improvement prompt with archive analysis.

    Persistent rules are in systemMessage to avoid repetition.
    """
    failures = extract_failures(benchmark_result)
    top_n = 10
    archive_summary = get_archive_summary(state, top_n=top_n)
    original = state.get('original_prompt', '')
    run_dir = state.get('run_dir', '.sica')

    # Include archive analysis (core SICA feature)
    parts = [
        f"## Results: {benchmark_result['passed']}✓ {benchmark_result['failed']}✗",
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


def main():
    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    transcript_path = hook_input.get("transcript_path", "")

    # Check if SICA loop is active
    state = read_state_file()
    if not state:
        # No active loop - allow exit
        sys.exit(0)

    # Validate state
    if not isinstance(state.get("iteration"), int):
        log("SICA: State file corrupted (invalid iteration)")
        Path(".sica/current_run.json").unlink(missing_ok=True)
        sys.exit(0)

    # max_iterations: 0 = benchmark only, 1+ = improvement attempts after initial
    max_iter = max(0, state["max_iterations"])

    # Check max iterations (iteration is 0-indexed, so >= means we've done max_iter attempts)
    if state["iteration"] >= max_iter:
        log(f"SICA: Max iterations ({max_iter}) reached.")
        _archive_state(state)
        sys.exit(0)

    # Check transcript for promise or cancel signal
    promise_detected = False
    cancel_detected = False
    if transcript_path and Path(transcript_path).exists():
        try:
            with open(transcript_path) as f:
                content = f.read()
            promise = state.get("completion_promise", "")
            if promise and f"<promise>{promise}</promise>" in content:
                promise_detected = True
            # Check for cancel signal (user or Claude can trigger)
            if "<sica:cancel>" in content or "<sica:stop>" in content:
                cancel_detected = True
        except Exception:
            pass

    # Handle cancel signal
    if cancel_detected:
        log("SICA: Cancel signal detected. Stopping loop.")
        _archive_state(state)
        sys.exit(0)

    # ALWAYS run benchmark first (to verify promise and archive results)
    log(f"SICA: Running benchmark (iteration {state['iteration']})...")
    benchmark_result = run_benchmark(
        state["benchmark_cmd"],
        timeout=state.get("benchmark_timeout", 300)
    )

    log(f"SICA: Score={benchmark_result['score']:.2f} "
        f"(passed={benchmark_result['passed']}, failed={benchmark_result['failed']}, "
        f"errors={benchmark_result['errors']})")

    # Archive results
    archive_iteration(state, benchmark_result, transcript_path)

    # Track recent scores for convergence detection
    recent_scores = state.get("recent_scores", [])
    recent_scores.append(round(benchmark_result["score"], 4))
    recent_scores = recent_scores[-10:]  # keep last 10
    state["recent_scores"] = recent_scores

    # Check convergence (10 identical scores = converged, no further improvement possible)
    if len(recent_scores) == 10 and len(set(recent_scores)) == 1:
        log(f"SICA: Converged at score {recent_scores[0]:.2f} (10 identical iterations)")
        _archive_state(state)
        sys.exit(0)

    # Check if target reached
    if benchmark_result["score"] >= state["target_score"]:
        log(f"SICA: Target score ({state['target_score']}) reached!")
        _archive_state(state)
        sys.exit(0)

    # Promise detected but tests still failing
    if promise_detected:
        log(f"SICA: Promise detected but tests still failing! Continuing...")

    # Update state for next iteration
    state["iteration"] += 1
    state["last_score"] = benchmark_result["score"]
    save_state(state)

    # Generate improvement prompt
    improvement_prompt = generate_improvement_prompt(state, benchmark_result)

    # Build system message with persistent rules (avoids repeating in reason)
    run_dir = state.get('run_dir', '.sica')
    promise = state['completion_promise']
    system_msg = (
        f"SICA iter {state['iteration']} | "
        f"Score: {benchmark_result['score']:.2f}/{state['target_score']} | "
        f"RULES: ONE fix→exit immediately. NO manual tests. NO test changes. "
        f"Read/update {run_dir}/journal.md. Done: <promise>{promise}</promise>"
    )

    # Output JSON to block exit and continue
    print(json.dumps({
        "decision": "block",
        "reason": improvement_prompt,
        "systemMessage": system_msg,
    }))


if __name__ == "__main__":
    main()
