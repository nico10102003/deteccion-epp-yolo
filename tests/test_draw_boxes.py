"""Pruebas unitarias de src/visualizations/draw_boxes.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.predict_model import Detection
from src.visualizations.draw_boxes import _color_for_class, draw_detections


class TestColorForClass:
    @pytest.mark.parametrize(
        "class_name",
        ["no-helmet", "no-gloves", "NO-HELMET", "no-Gloves"],
    )
    def test_violation_classes_are_red(self, class_name):
        # Arrange
        expected_color = (220, 50, 50)

        # Act
        color = _color_for_class(class_name)

        # Assert
        assert color == expected_color

    @pytest.mark.parametrize("class_name", ["helmet", "gloves", "HELMET", "Gloves"])
    def test_compliant_classes_are_green(self, class_name):
        # Arrange
        expected_color = (46, 204, 113)

        # Act
        color = _color_for_class(class_name)

        # Assert
        assert color == expected_color

    def test_person_class_is_gray(self):
        # Arrange
        class_name = "person"
        expected_color = (160, 160, 160)

        # Act
        color = _color_for_class(class_name)

        # Assert
        assert color == expected_color

    def test_unknown_class_defaults_to_green(self):
        # Arrange
        class_name = "something-else"
        expected_color = (46, 204, 113)

        # Act
        color = _color_for_class(class_name)

        # Assert
        assert color == expected_color


class TestDrawDetections:
    def test_does_not_mutate_original_image(self, sample_image_array, sample_detections):
        # Arrange
        original_copy = sample_image_array.copy()

        # Act
        draw_detections(sample_image_array, sample_detections)

        # Assert
        assert np.array_equal(sample_image_array, original_copy)

    def test_output_has_same_shape(self, sample_image_array, sample_detections):
        # Arrange
        expected_shape = sample_image_array.shape

        # Act
        annotated = draw_detections(sample_image_array, sample_detections)

        # Assert
        assert annotated.shape == expected_shape

    def test_empty_detections_returns_identical_image(self, sample_image_array):
        # Arrange
        detections = []

        # Act
        annotated = draw_detections(sample_image_array, detections)

        # Assert
        assert np.array_equal(annotated, sample_image_array)

    def test_drawing_changes_pixels_when_detections_present(self, sample_image_array, sample_detections):
        # Arrange
        original = sample_image_array

        # Act
        annotated = draw_detections(original, sample_detections)

        # Assert
        assert not np.array_equal(annotated, original)

    @pytest.mark.parametrize("num_detections", [1, 2, 5, 10])
    def test_handles_multiple_detections(self, sample_image_array, num_detections):
        # Arrange
        detections = [
            Detection(class_name="helmet", confidence=0.8, x1=i, y1=i, x2=i + 10, y2=i + 10)
            for i in range(num_detections)
        ]

        # Act
        annotated = draw_detections(sample_image_array, detections)

        # Assert
        assert annotated.shape == sample_image_array.shape
