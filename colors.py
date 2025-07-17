"""
Color scheme for terminal output using pychalk.
"""

from typing import Callable
from chalk import bold, cyan, red, green, yellow, blue, magenta, white

ColorFunc = Callable[[str], str]

header: ColorFunc = lambda text: bold(cyan(text))
title: ColorFunc = lambda text: bold(white(text))

# Status indicators
success: ColorFunc = lambda text: bold(green(text))
warning: ColorFunc = lambda text: bold(yellow(text))
error: ColorFunc = lambda text: bold(red(text))
info: ColorFunc = lambda text: bold(blue(text))

# Data and values
value: ColorFunc = lambda text: bold(white(text))
highlight: ColorFunc = lambda text: bold(magenta(text))
metric: ColorFunc = lambda text: bold(white(text))

# Sections
section: ColorFunc = lambda text: bold(blue(text))
subsection: ColorFunc = lambda text: bold(cyan(text))

# Performance indicators
positive: ColorFunc = lambda text: bold(green(text))
negative: ColorFunc = lambda text: bold(red(text))
neutral: ColorFunc = lambda text: bold(yellow(text))


# Separator lines
def separator(char: str = "=", length: int = 80) -> str:
    """Create a colored separator line."""
    return str(white(char * length))


def format_number(num: float, decimals: int = 4, positive_color: bool = True) -> str:
    """Format a number with color based on its value."""
    formatted = f"{num:.{decimals}f}"
    if positive_color:
        if num > 0:
            return positive(formatted)
        elif num < 0:
            return negative(formatted)
        else:
            return neutral(formatted)
    else:
        return value(formatted)


def format_percentage(num: float, decimals: int = 2) -> str:
    """Format a percentage with color."""
    formatted = f"{num:.{decimals}f}%"
    if num > 0:
        return positive(formatted)
    elif num < 0:
        return negative(formatted)
    else:
        return neutral(formatted)
