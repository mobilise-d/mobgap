"""Compatibility imports for the secondary outcome calculator base class."""

from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator as BaseSDMOCalculator
from mobgap.secondary_outcomes.base import base_secondary_outcome_docfiller as base_sdmo_docfiller

__all__ = ["BaseSDMOCalculator", "base_sdmo_docfiller"]
