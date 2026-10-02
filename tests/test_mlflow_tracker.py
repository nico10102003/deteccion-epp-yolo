"""Pruebas unitarias de src/tracking/mlflow_tracker.py (mlflow mockeado)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def mocked_mlflow():
    with patch("src.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_mlflow


class TestMLflowTrackerInit:
    def test_sets_tracking_uri(self, mocked_mlflow):
        # Arrange
        tracking_uri = "http://example:5000"

        # Act
        MLflowTracker(tracking_uri=tracking_uri)

        # Assert
        mocked_mlflow.set_tracking_uri.assert_called_once_with(tracking_uri)

    def test_sets_experiment(self, mocked_mlflow):
        # Arrange
        experiment_name = "mi-experimento"

        # Act
        MLflowTracker(experiment_name=experiment_name)

        # Assert
        mocked_mlflow.set_experiment.assert_called_once_with(experiment_name)


class TestLogInferenceRun:
    def test_starts_a_run(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()

        # Act
        tracker.log_inference_run(
            conf_threshold=0.25, num_detections=3, inference_time_ms=15.0, num_violations=1
        )

        # Assert
        mocked_mlflow.start_run.assert_called_once()

    def test_logs_conf_threshold_param(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()
        conf_threshold = 0.4

        # Act
        tracker.log_inference_run(
            conf_threshold=conf_threshold, num_detections=2, inference_time_ms=10.0, num_violations=0
        )

        # Assert
        mocked_mlflow.log_param.assert_any_call("conf_threshold", conf_threshold)

    def test_logs_num_detections_metric(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()
        num_detections = 7

        # Act
        tracker.log_inference_run(
            conf_threshold=0.25, num_detections=num_detections, inference_time_ms=10.0, num_violations=0
        )

        # Assert
        mocked_mlflow.log_metric.assert_any_call("num_detections", num_detections)

    def test_logs_environment_tag(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()
        environment = "prod"

        # Act
        tracker.log_inference_run(
            conf_threshold=0.25,
            num_detections=1,
            inference_time_ms=10.0,
            num_violations=0,
            environment=environment,
        )

        # Assert
        mocked_mlflow.set_tag.assert_any_call("environment", environment)

    @pytest.mark.parametrize("num_violations", [0, 1, 3, 10])
    def test_various_violation_counts_logged(self, mocked_mlflow, num_violations):
        # Arrange
        tracker = MLflowTracker()

        # Act
        tracker.log_inference_run(
            conf_threshold=0.25,
            num_detections=num_violations,
            inference_time_ms=5.0,
            num_violations=num_violations,
        )

        # Assert
        mocked_mlflow.log_metric.assert_any_call("num_violations", num_violations)


class TestLogEvaluation:
    def test_logs_all_evaluation_metrics(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()

        # Act
        tracker.log_evaluation(
            precision=0.9, recall=0.85, f1=0.87, map50=0.88, dataset_version="v1"
        )

        # Assert
        mocked_mlflow.log_metric.assert_any_call("precision", 0.9)
        mocked_mlflow.log_metric.assert_any_call("recall", 0.85)
        mocked_mlflow.log_metric.assert_any_call("f1_score", 0.87)
        mocked_mlflow.log_metric.assert_any_call("map50", 0.88)

    def test_tags_run_type_as_evaluation(self, mocked_mlflow):
        # Arrange
        tracker = MLflowTracker()

        # Act
        tracker.log_evaluation(precision=0.9, recall=0.9, f1=0.9, map50=0.9, dataset_version="v2")

        # Assert
        mocked_mlflow.set_tag.assert_any_call("run_type", "evaluation")
