"""Text Anonymizer: A library for anonymizing and de-anonymizing text."""

__version__ = "0.1.0"

from .core import anonymize, deanonymize
from .main import main

__all__ = ["anonymize", "deanonymize", "main"]
