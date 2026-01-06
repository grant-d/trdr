"""Pareto frontier selection interface.

Provides CLI tools for viewing and selecting from Pareto-optimal solutions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .multi_objective import MooResult


def display_pareto_front(
    result: "MooResult",
    max_rows: int = 20,
    output_file: str | None = None,
) -> None:
    """Display Pareto frontier as formatted table and dump params to file.

    Args:
        result: MooResult with Pareto solutions
        max_rows: Maximum rows to display (truncates if exceeded)
        output_file: Path to write full params (default: pareto_params.txt)
    """
    if result.n_solutions == 0:
        print("No Pareto-optimal solutions found.")
        return

    # Compute scores for all solutions
    scores = _compute_scores(result)

    # Build header (no Params column, add Score)
    headers = ["#"] + result.objective_names + ["Score"]
    col_widths = [4] + [12] * len(result.objective_names) + [8]

    # Print header
    header_line = ""
    for i, h in enumerate(headers):
        header_line += f"{h:>{col_widths[i]}}  "
    print("\n" + "=" * len(header_line))
    print("PARETO FRONTIER")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    n_display = min(result.n_solutions, max_rows)
    for i in range(n_display):
        obj_dict = result.get_objectives_dict(i)

        row = f"{i:>{col_widths[0]}}  "
        for j, obj_name in enumerate(result.objective_names):
            val = obj_dict[obj_name]
            if obj_name in ("max_drawdown", "win_rate", "cagr"):
                row += f"{val:>{col_widths[j+1]}.1%}  "
            elif obj_name == "total_trades":
                row += f"{int(val):>{col_widths[j+1]}}  "
            elif obj_name == "alpha":
                row += f"{val:>{col_widths[j+1]}.2f}x  "
            else:
                row += f"{val:>{col_widths[j+1]}.2f}  "

        # Add score
        row += f"{scores[i]:>{col_widths[-1]}.3f}  "
        print(row)

    if result.n_solutions > max_rows:
        print(f"... ({result.n_solutions - max_rows} more solutions)")

    print("-" * len(header_line))
    print(f"Total: {result.n_solutions} Pareto-optimal solutions")
    print("=" * len(header_line))

    # Dump params to file
    _dump_params_to_file(result, scores, output_file or "pareto_params.txt")


def _compute_scores(result: "MooResult") -> list[float]:
    """Compute composite score for each solution.

    Score = weighted sum of normalized objectives.
    """
    scores = []
    for i in range(result.n_solutions):
        obj_dict = result.get_objectives_dict(i)
        score = 0.0

        # Weights for each objective (positive = maximize, negative = minimize)
        weights = {
            "cagr": 2.0,
            "calmar": 1.5,
            "sortino": 1.0,
            "profit_factor": 1.0,
            "win_rate": 0.5,
            "total_trades": 0.1,
            "max_drawdown": -2.0,
            "alpha": 1.5,
            "sharpe": 1.0,
        }

        for obj_name, val in obj_dict.items():
            weight = weights.get(obj_name, 0.0)
            # Normalize: percentages to decimals, cap extreme values
            if obj_name in ("max_drawdown", "win_rate", "cagr"):
                val = val  # Already decimal
            elif obj_name == "total_trades":
                val = val / 100.0  # Normalize
            score += weight * val

        scores.append(score)

    return scores


def _dump_params_to_file(
    result: "MooResult",
    scores: list[float],
    filepath: str,
) -> None:
    """Write all params to text file."""
    with open(filepath, "w") as f:
        f.write("PARETO FRONTIER PARAMETERS\n")
        f.write("=" * 60 + "\n\n")

        for i in range(result.n_solutions):
            param_dict = result.get_params_dict(i)
            obj_dict = result.get_objectives_dict(i)

            f.write(f"Solution {i} (Score: {scores[i]:.3f})\n")
            f.write("-" * 40 + "\n")

            f.write("Parameters:\n")
            for k, v in param_dict.items():
                if isinstance(v, int):
                    f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {k}: {v:.4f}\n")

            f.write("\nObjectives:\n")
            for k, v in obj_dict.items():
                if k in ("max_drawdown", "win_rate", "cagr"):
                    f.write(f"  {k}: {v:.2%}\n")
                elif k == "total_trades":
                    f.write(f"  {k}: {int(v)}\n")
                elif k == "alpha":
                    f.write(f"  {k}: {v:.2f}x\n")
                else:
                    f.write(f"  {k}: {v:.2f}\n")

            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write(f"Total: {result.n_solutions} solutions\n")

    print(f"\nParams written to: {filepath}")


def select_from_pareto(
    result: "MooResult",
    interactive: bool = True,
) -> dict[str, float] | None:
    """Select parameters from Pareto frontier.

    Args:
        result: MooResult with Pareto solutions
        interactive: If True, prompt user for selection. If False, return best Sharpe.

    Returns:
        Parameter dictionary for selected solution, or None if no solutions
    """
    if result.n_solutions == 0:
        return None

    if not interactive:
        # Auto-select: best Sharpe ratio
        return _select_best_sharpe(result)

    # Display options
    display_pareto_front(result)

    # Get user selection
    while True:
        try:
            choice_str = input(f"\nSelect solution (0-{result.n_solutions - 1}): ").strip()

            # Handle special commands
            if choice_str.lower() in ("q", "quit", "exit"):
                return None
            if choice_str.lower() == "best":
                return _select_best_sharpe(result)

            choice = int(choice_str)
            if 0 <= choice < result.n_solutions:
                selected = result.get_params_dict(choice)
                print(f"\nSelected: {selected}")
                return selected
            else:
                print(f"Invalid choice. Enter 0-{result.n_solutions - 1}")
        except ValueError:
            print("Enter a number, 'best' for best Sharpe, or 'q' to quit")


def _select_best_sharpe(result: "MooResult") -> dict[str, float]:
    """Select solution with best Sharpe ratio."""
    best_idx = 0
    best_sharpe = float("-inf")

    for i in range(result.n_solutions):
        obj_dict = result.get_objectives_dict(i)
        if "sharpe" in obj_dict and obj_dict["sharpe"] > best_sharpe:
            best_sharpe = obj_dict["sharpe"]
            best_idx = i

    return result.get_params_dict(best_idx)


def filter_pareto_front(
    result: "MooResult",
    constraints: dict[str, tuple[float | None, float | None]],
) -> list[int]:
    """Filter Pareto solutions by objective constraints.

    Args:
        result: MooResult with Pareto solutions
        constraints: Dict of objective -> (min, max) bounds

    Returns:
        List of solution indices that satisfy constraints

    Example:
        # Solutions with max_drawdown <= 15% and sharpe >= 1.0
        filter_pareto_front(result, {
            "max_drawdown": (None, 0.15),
            "sharpe": (1.0, None),
        })
    """
    valid_indices = []

    for i in range(result.n_solutions):
        obj_dict = result.get_objectives_dict(i)
        valid = True

        for obj_name, (min_val, max_val) in constraints.items():
            if obj_name not in obj_dict:
                continue
            val = obj_dict[obj_name]
            if min_val is not None and val < min_val:
                valid = False
                break
            if max_val is not None and val > max_val:
                valid = False
                break

        if valid:
            valid_indices.append(i)

    return valid_indices


def rank_pareto_solutions(
    result: "MooResult",
    weights: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Rank Pareto solutions by weighted sum of objectives.

    Args:
        result: MooResult with Pareto solutions
        weights: Dict of objective -> weight. Positive = maximize, negative = minimize.
                 Defaults to equal weights favoring Sharpe.

    Returns:
        List of (index, score) sorted by descending score
    """
    if weights is None:
        weights = {"sharpe": 1.0, "profit_factor": 0.5, "max_drawdown": -1.0}

    scores = []
    for i in range(result.n_solutions):
        obj_dict = result.get_objectives_dict(i)
        score = 0.0
        for obj_name, weight in weights.items():
            if obj_name in obj_dict:
                score += weight * obj_dict[obj_name]
        scores.append((i, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
