"""Funciones para dibujar incumplimientos de EPP sobre imágenes."""

from __future__ import annotations

import cv2
import numpy as np

from src.classes import is_allowed_class, normalize_class_name
from src.models.predict_model import Detection

RED_COLOR = (220, 50, 50)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Dibuja únicamente detecciones de no_helmet y no_gloves.

    Las detecciones de incumplimiento se muestran en rojo.
    Cualquier otra clase recibida se ignora.
    """

    annotated = image.copy()

    for detection in detections:
        if not is_allowed_class(detection.class_name):
            continue

        normalized = normalize_class_name(detection.class_name)

        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            RED_COLOR,
            2,
        )

        label = f"{normalized} {detection.confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            RED_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotated
