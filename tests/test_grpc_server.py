"""Pruebas unitarias del servicio gRPC (Detect / HealthCheck).

Requiere que los stubs `inference_pb2` / `inference_pb2_grpc` hayan sido
generados con `scripts/generate_grpc.sh`. Si no existen, estas pruebas
se omiten automáticamente (no rompen el resto de la suite).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "src.grpc_service.inference_pb2",
    reason="Genera los stubs con scripts/generate_grpc.sh antes de correr estas pruebas.",
)

from src.grpc_service import inference_pb2  # noqa: E402
from src.grpc_service.server import EPPInferenceServicer, serve  # noqa: E402


@pytest.fixture
def servicer(mock_yolo_model):
    with patch("src.grpc_service.server.MLflowTracker"):
        srv = EPPInferenceServicer(use_mlflow=False)
        srv._model = mock_yolo_model
        yield srv


@pytest.fixture
def grpc_context():
    return MagicMock()


@pytest.fixture
def servicer_without_model():
    with patch("src.grpc_service.server.MLflowTracker"):
        return EPPInferenceServicer(model_path="/nonexistent.pt", use_mlflow=False)


class TestDetect:
    def test_returns_detections(self, servicer, grpc_context, small_image_bytes):
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="test-1"
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert len(response.detections) == 2

    def test_preserves_request_id(self, servicer, grpc_context, small_image_bytes):
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="abc-123"
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert response.request_id == "abc-123"

    def test_sets_internal_error_on_bad_image(self, servicer, grpc_context):
        # Arrange
        request = inference_pb2.DetectRequest(image_data=b"", conf_threshold=0.25, request_id="x")

        # Act
        servicer.Detect(request, grpc_context)

        # Assert
        grpc_context.set_code.assert_called_once()

    def test_default_conf_threshold_applied_when_zero(self, servicer, grpc_context, small_image_bytes):
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.0, request_id="y"
        )

        # Act
        servicer.Detect(request, grpc_context)

        # Assert
        _, kwargs = servicer.model.predict.call_args
        assert kwargs["conf"] == 0.25

    def test_model_load_error_sets_unavailable(
        self, servicer_without_model, grpc_context, small_image_bytes
    ):
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="z"
        )

        # Act
        servicer_without_model.Detect(request, grpc_context)

        # Assert
        grpc_context.set_code.assert_called_once()

    def test_returns_model_version(self, servicer, grpc_context, small_image_bytes):
        """Debe devolver la versión del modelo."""
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes,
            conf_threshold=0.25,
            request_id="version-test",
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert response.model_version == "yolo11n-epp-v0.1"

    def test_returns_inference_time(self, servicer, grpc_context, small_image_bytes):
        """Debe devolver el tiempo de inferencia."""
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes,
            conf_threshold=0.25,
            request_id="time-test",
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert response.inference_time_ms >= 0

    def test_converts_bounding_box(self, servicer, grpc_context, small_image_bytes):
        """Debe convertir correctamente las coordenadas del bounding box."""
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes,
            conf_threshold=0.25,
            request_id="box-test",
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        box = response.detections[0].box
        assert box.x1 == pytest.approx(10.0)
        assert box.y1 == pytest.approx(10.0)
        assert box.x2 == pytest.approx(50.0)
        assert box.y2 == pytest.approx(50.0)

    def test_returns_detection_class_name(self, servicer, grpc_context, small_image_bytes):
        """Debe conservar el nombre de la clase detectada."""
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes,
            conf_threshold=0.25,
            request_id="class-test",
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert response.detections[0].class_name == "helmet"

    def test_returns_detection_confidence(self, servicer, grpc_context, small_image_bytes):
        """Debe conservar la confianza de la detección."""
        # Arrange
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes,
            conf_threshold=0.25,
            request_id="confidence-test",
        )

        # Act
        response = servicer.Detect(request, grpc_context)

        # Assert
        assert response.detections[0].confidence == pytest.approx(0.91)


class TestHealthCheck:
    def test_reports_model_loaded_true(self, servicer, grpc_context):
        # Arrange
        request = inference_pb2.HealthRequest()

        # Act
        response = servicer.HealthCheck(request, grpc_context)

        # Assert
        assert response.model_loaded is True

    def test_reports_model_loaded_false_when_missing(self, servicer_without_model, grpc_context):
        # Arrange
        request = inference_pb2.HealthRequest()

        # Act
        response = servicer_without_model.HealthCheck(request, grpc_context)

        # Assert
        assert response.model_loaded is False

    def test_healthcheck_returns_model_name(self, servicer, grpc_context):
        """Debe devolver el nombre del modelo en HealthCheck."""
        # Arrange
        request = inference_pb2.HealthRequest()

        # Act
        response = servicer.HealthCheck(request, grpc_context)

        # Assert
        assert response.model_name == "yolo11n-epp-mock"


class TestServe:
    @pytest.fixture
    def grpc_server_mocks(self):
        """Mockea grpc.server, el registro y el servicer; devuelve (fábrica grpc.server, servidor falso)."""
        with patch("src.grpc_service.server.grpc.server") as mock_grpc_server, patch(
            "src.grpc_service.server.inference_pb2_grpc.add_EPPInferenceServiceServicer_to_server"
        ), patch("src.grpc_service.server.EPPInferenceServicer"):
            server = MagicMock()
            mock_grpc_server.return_value = server
            yield mock_grpc_server, server

    def test_creates_grpc_server(self, grpc_server_mocks):
        """Debe crear correctamente el servidor gRPC."""
        # Arrange
        grpc_server_factory, _ = grpc_server_mocks

        # Act
        serve()

        # Assert
        grpc_server_factory.assert_called_once()

    def test_configures_grpc_port(self, grpc_server_mocks):
        """Debe configurar el puerto del servidor gRPC."""
        # Arrange
        _, mock_server = grpc_server_mocks
        expected_address = "[::]:50051"

        # Act
        serve()

        # Assert
        mock_server.add_insecure_port.assert_called_once_with(expected_address)

    def test_starts_and_waits_for_server(self, grpc_server_mocks):
        """Debe iniciar el servidor y esperar su terminación."""
        # Arrange
        _, server = grpc_server_mocks

        # Act
        serve()

        # Assert
        server.start.assert_called_once()
        server.wait_for_termination.assert_called_once()
