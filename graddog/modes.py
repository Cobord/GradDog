"""
Modes of differentiation
"""

from __future__ import annotations
from enum import Enum


class Mode(str, Enum):
    """
    modes of differentiation
    """

    FORWARD = "forward"
    REVERSE = "reverse"
