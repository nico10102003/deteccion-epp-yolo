"""Clases de incumplimiento de EPP permitidas en la aplicación."""

from __future__ import annotations

ALLOWED_CLASSES = frozenset({"no_helmet", "no_gloves"})


def normalize_class_name(class_name: str) -> str:
    """Normaliza el nombre de una clase para comparaciones."""
    return class_name.strip().lower().replace("-", "_")


def is_allowed_class(class_name: str) -> bool:
    """Indica si la clase es no_helmet o no_gloves."""
    return normalize_class_name(class_name) in ALLOWED_CLASSES
