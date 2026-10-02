"""Pruebas unitarias para el cliente gRPC de detección de EPP."""

from unittest.mock import MagicMock, patch

import pytest

from src.grpc_service.client import (
    ClientDetection,
    ClientDetectResult,
    detect,
)


@pytest.fixture
def mock_response():
    """Crea una respuesta gRPC simulada con una detección."""
    response = MagicMock()
    response.inference_time_ms = 15.5
    response.model_version = "yolo11n-epp-v0.1"

    detection = MagicMock()
    detection.class_name = "helmet"
    detection.confidence = 0.95
    detection.box.x1 = 10.0
    detection.box.y1 = 20.0
    detection.box.x2 = 100.0
    detection.box.y2 = 200.0

    response.detections = [detection]
    return response


@pytest.fixture
def grpc_mocks(mock_response):
    """Mockea el canal gRPC y el stub; devuelve (mock_channel, stub)."""
    with patch("src.grpc_service.client.grpc.insecure_channel") as mock_channel, patch(
        "src.grpc_service.client.inference_pb2_grpc.EPPInferenceServiceStub"
    ) as mock_stub_cls:
        stub = mock_stub_cls.return_value
        stub.Detect.return_value = mock_response
        yield mock_channel, stub


@pytest.mark.usefixtures("grpc_mocks")
def test_detect_returns_client_detect_result():
    """Debe devolver un ClientDetectResult."""
    # Arrange
    image_bytes = b"image"

    # Act
    result = detect(image_bytes)

    # Assert
    assert isinstance(result, ClientDetectResult)


def test_detect_uses_default_address(grpc_mocks):
    """Debe usar localhost:50051 por defecto."""
    # Arrange
    mock_channel, _ = grpc_mocks

    # Act
    detect(b"image")

    # Assert
    mock_channel.assert_called_once_with("localhost:50051")


def test_detect_uses_custom_address(grpc_mocks):
    """Debe aceptar una dirección personalizada."""
    # Arrange
    mock_channel, _ = grpc_mocks
    address = "192.168.1.10:6000"

    # Act
    detect(b"image", address=address)

    # Assert
    mock_channel.assert_called_once_with(address)


def test_detect_sends_image_bytes(grpc_mocks):
    """Debe enviar correctamente los bytes de la imagen."""
    # Arrange
    _, stub = grpc_mocks
    image_bytes = b"fake-image-data"

    # Act
    detect(image_bytes)

    # Assert
    request = stub.Detect.call_args.args[0]
    assert request.image_data == image_bytes


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_detect_sends_conf_threshold(grpc_mocks, threshold):
    """Debe enviar el umbral de confianza recibido."""
    # Arrange
    _, stub = grpc_mocks

    # Act
    detect(b"image", conf_threshold=threshold)

    # Assert
    request = stub.Detect.call_args.args[0]
    assert request.conf_threshold == pytest.approx(threshold)


def test_detect_generates_request_id(grpc_mocks):
    """Debe generar un identificador único para la solicitud."""
    # Arrange
    _, stub = grpc_mocks

    # Act
    detect(b"image")

    # Assert
    request = stub.Detect.call_args.args[0]
    assert request.request_id
    assert len(request.request_id) == 36


def test_detect_uses_ten_second_timeout(grpc_mocks):
    """Debe usar un timeout de 10 segundos."""
    # Arrange
    _, stub = grpc_mocks

    # Act
    detect(b"image")

    # Assert
    assert stub.Detect.call_args.kwargs["timeout"] == 10


@pytest.mark.usefixtures("grpc_mocks")
def test_detect_parses_detection():
    """Debe convertir una detección gRPC a ClientDetection."""
    # Arrange
    image_bytes = b"image"

    # Act
    result = detect(image_bytes)

    # Assert
    detection = result.detections[0]
    assert isinstance(detection, ClientDetection)
    assert detection.class_name == "helmet"
    assert detection.confidence == pytest.approx(0.95)
    assert detection.box == pytest.approx((10.0, 20.0, 100.0, 200.0))


@pytest.mark.usefixtures("grpc_mocks")
def test_detect_returns_inference_metadata():
    """Debe conservar tiempo de inferencia y versión del modelo."""
    # Arrange
    image_bytes = b"image"

    # Act
    result = detect(image_bytes)

    # Assert
    assert result.inference_time_ms == pytest.approx(15.5)
    assert result.model_version == "yolo11n-epp-v0.1"


@pytest.mark.usefixtures("grpc_mocks")
def test_detect_parses_multiple_detections(mock_response):
    """Debe convertir correctamente múltiples detecciones."""
    # Arrange
    second = MagicMock()
    second.class_name = "Gloves"
    second.confidence = 0.88
    second.box.x1 = 30.0
    second.box.y1 = 40.0
    second.box.x2 = 130.0
    second.box.y2 = 140.0
    mock_response.detections.append(second)

    # Act
    result = detect(b"image")

    # Assert
    assert len(result.detections) == 2
    assert result.detections[1].class_name == "Gloves"
    assert result.detections[1].box == pytest.approx(
        (30.0, 40.0, 130.0, 140.0)
    )
