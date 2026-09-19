"""Legacy empty namespace.

The old project had a monolithic ``scgrip.main`` placeholder with eager heavy
imports.  The current entry points are intentionally explicit:
``build/prepare_reference_bank.py``, ``build/prepare_external_cohort.py``,
``build/train_phase1.py``, and ``build/evaluate_phase1.py``.
"""

__all__: list[str] = []
