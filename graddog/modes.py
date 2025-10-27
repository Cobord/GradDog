"""
Modes of differentiation
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, Union


class Mode(str,Enum):
    FORWARD = 'forward'
    REVERSE = 'reverse'