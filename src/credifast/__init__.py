"""CrediFast credit-risk decision-support prototype."""

from .domain import ApplicantProfile
from .service import evaluate_application

__all__ = ["ApplicantProfile", "evaluate_application"]
__version__ = "0.1.0"
