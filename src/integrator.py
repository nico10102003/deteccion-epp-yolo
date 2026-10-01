"""Integración de preprocesamiento, inferencia y reglas de negocio de EPP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.classes import is_allowed_class, normalize_class_name
from src.models.predict_model import InferenceResult, predict


@dataclass(frozen=True)
class DetectionResponse:
    """Respuesta completa del pipeline de detección."""

    inference_result: InferenceResult
    num_helmets: int
    num_no_helmets: int
    num_gloves: int
    num_no_gloves: int
    num_vests: int = 0
    num_goggles: int = 0
    num_masks: int = 0
    num_safety_shoes: int = 0


def run_detection(
    model,
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    tracker=None,
) -> DetectionResponse:
    """Ejecuta el pipeline completo de detección.

    Solo se conservan las clases:
    - no_helmet
    - no_gloves
    """

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen recibida.")

    inference_result = predict(
        model=model,
        image=image,
        conf_threshold=conf_threshold,
    )

    detections = [
        detection
        for detection in inference_result.detections
        if is_allowed_class(detection.class_name)
    ]

    filtered_result = InferenceResult(
        detections=detections,
        inference_time_ms=inference_result.inference_time_ms,
    )

    num_no_helmets = sum(
        normalize_class_name(d.class_name) == "no_helmet" for d in detections
    )
    num_no_gloves = sum(
        normalize_class_name(d.class_name) == "no_gloves" for d in detections
    )

    if tracker is not None:
        tracker.log_inference(
            inference_time_ms=filtered_result.inference_time_ms,
            num_detections=len(detections),
            conf_threshold=conf_threshold,
            num_violations=len(detections),
        )

    return DetectionResponse(
        inference_result=filtered_result,
        num_helmets=0,
        num_no_helmets=num_no_helmets,
        num_gloves=0,
        num_no_gloves=num_no_gloves,
    )


# Compatibilidad con código y pruebas existentes.
EPPDetectionResponse = DetectionResponse
