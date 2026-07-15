"""
LedgerSentry: real-time financial-transaction fraud detection with a tunable
reject-to-review knob.

Fraud-DETECTION software and research only. No trading, no moving money, no
personalized financial advice, anywhere in this package.
"""

__version__ = "0.1.0"

from .model import FRAUD, LEGIT, REVIEW, FraudDetector

__all__ = ["FraudDetector", "FRAUD", "LEGIT", "REVIEW"]
