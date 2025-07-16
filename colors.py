"""
Color scheme for terminal output using pychalk.
"""

from chalk import bold, cyan, red, green, yellow, blue, magenta, white

# Create chalk instances with different styles


# Style functions (chain chalk styles)
header = lambda text: bold(cyan(text))
title = lambda text: bold(white(text))

# Status indicators
success = lambda text: bold(green(text))
warning = lambda text: bold(yellow(text))
error = lambda text: bold(red(text))
info = lambda text: bold(blue(text))

# Data and values
value = lambda text: bold(white(text))
highlight = lambda text: bold(magenta(text))
metric = lambda text: bold(white(text))

# Sections
section = lambda text: bold(blue(text))
subsection = lambda text: bold(cyan(text))

# Performance indicators
positive = lambda text: bold(green(text))
negative = lambda text: bold(red(text))
neutral = lambda text: bold(yellow(text))

# Separator lines
def separator(char="=", length=80):
    """Create a colored separator line."""
    return white(char * length)

def format_number(num, decimals=4, positive_color=True):
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

def format_percentage(num, decimals=2):
    """Format a percentage with color."""
    formatted = f"{num:.{decimals}f}%"
    if num > 0:
        return positive(formatted)
    elif num < 0:
        return negative(formatted)
    else:
        return neutral(formatted)
