"""Pruebas unitarias de src/integrator.py (Controlador)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.integrator import EPPDetectionResponse, run_detection


class TestRunDetection:
    def test_returns_epp_detection_response(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        model = mock_yolo_model

        # Act
        response = run_detection(model, small_image_bytes)

        # Assert
        assert isinstance(response, EPPDetectionResponse)

    def test_counts_helmet_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_helmets = 1

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_helmets == expected_helmets

    def test_counts_gloves_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_gloves = 0

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_gloves == expected_gloves

    def test_counts_vests_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_vests = 0

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_vests == expected_vests

    def test_counts_goggles_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_goggles = 0

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_goggles == expected_goggles

    def test_counts_safety_shoes_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_safety_shoes = 0

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_safety_shoes == expected_safety_shoes

    def test_counts_total_detections(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        expected_detections = 2

        # Act
        response = run_detection(mock_yolo_model, small_image_bytes)

        # Assert
        assert response.num_detections == expected_detections

    def test_propagates_conf_threshold_to_model(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        conf_threshold = 0.6

        # Act
        run_detection(
            mock_yolo_model,
            small_image_bytes,
            conf_threshold=conf_threshold,
        )

        # Assert
        _, kwargs = mock_yolo_model.predict.call_args
        assert kwargs["conf"] == conf_threshold

    def test_calls_tracker_when_provided(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        tracker = MagicMock()

        # Act
        run_detection(
            mock_yolo_model,
            small_image_bytes,
            tracker=tracker,
        )

        # Assert
        tracker.log_inference_run.assert_called_once()

    def test_does_not_call_tracker_when_none(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        tracker = None

        # Act
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
            tracker=tracker,
        )

        # Assert
        assert response is not None

    def test_tracker_receives_expected_kwargs(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        # Arrange
        tracker = MagicMock()
        conf_threshold = 0.4

        # Act
        run_detection(
            mock_yolo_model,
            small_image_bytes,
            conf_threshold=conf_threshold,
            tracker=tracker,
        )

        # Assert
        _, kwargs = tracker.log_inference_run.call_args
        assert kwargs["conf_threshold"] == conf_threshold
        assert kwargs["num_detections"] == 2
        assert kwargs["num_violations"] == 0

    @pytest.mark.parametrize(
        "conf",
        [0.1, 0.3, 0.5, 0.7],
    )
    def test_multiple_confidence_values_do_not_error(
        self,
        mock_yolo_model,
        small_image_bytes,
        conf,
    ):
        # Arrange
        model = mock_yolo_model

        # Act
        response = run_detection(
            model,
            small_image_bytes,
            conf_threshold=conf,
        )

        # Assert
        assert response.inference_result is not None
