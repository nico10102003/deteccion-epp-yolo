"""Pruebas unitarias de src/data/preprocess_img.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess_img import (
    InvalidImageError,
    decode_image_bytes,
    preprocess,
    resize_if_needed,
)


class TestDecodeImageBytes:
    def test_decodes_valid_png(self, small_image_bytes):
        # Arrange
        image_bytes = small_image_bytes

        # Act
        image = decode_image_bytes(image_bytes)

        # Assert
        assert isinstance(image, np.ndarray)
        assert image.shape == (64, 64, 3)

    def test_raises_on_empty_bytes(self):
        # Arrange
        image_bytes = b""

        # Act / Assert
        with pytest.raises(InvalidImageError):
            decode_image_bytes(image_bytes)

    def test_raises_on_garbage_bytes(self):
        # Arrange
        image_bytes = b"not-an-image-at-all"

        # Act / Assert
        with pytest.raises(InvalidImageError):
            decode_image_bytes(image_bytes)

    @pytest.mark.parametrize("size", [(1, 1), (16, 32), (32, 16), (256, 256)])
    def test_decodes_various_sizes(self, image_bytes_factory, size):
        # Arrange
        w, h = size
        image_bytes = image_bytes_factory(w, h)

        # Act
        image = decode_image_bytes(image_bytes)

        # Assert
        assert image.shape[:2] == (h, w)

    def test_output_is_rgb_three_channels(self, small_image_bytes):
        # Arrange
        image_bytes = small_image_bytes

        # Act
        image = decode_image_bytes(image_bytes)

        # Assert
        assert image.shape[2] == 3


class TestResizeIfNeeded:
    def test_no_resize_when_within_limit(self, sample_image_array):
        # Arrange
        max_side = 1280

        # Act
        result = resize_if_needed(sample_image_array, max_side=max_side)

        # Assert
        assert result.shape == sample_image_array.shape

    def test_resizes_when_exceeding_limit(self):
        # Arrange
        big = np.zeros((3000, 1000, 3), dtype=np.uint8)
        max_side = 1000

        # Act
        result = resize_if_needed(big, max_side=max_side)

        # Assert
        assert max(result.shape[:2]) <= max_side

    def test_preserves_aspect_ratio(self):
        # Arrange
        big = np.zeros((2000, 1000, 3), dtype=np.uint8)
        original_ratio = 2000 / 1000

        # Act
        result = resize_if_needed(big, max_side=1000)

        # Assert
        new_ratio = result.shape[0] / result.shape[1]
        assert abs(original_ratio - new_ratio) < 0.05

    @pytest.mark.parametrize("max_side", [100, 500, 1280, 2000])
    def test_various_max_side_values(self, max_side):
        # Arrange
        big = np.zeros((1600, 1600, 3), dtype=np.uint8)

        # Act
        result = resize_if_needed(big, max_side=max_side)

        # Assert
        assert max(result.shape[:2]) <= max(max_side, 1600)


class TestPreprocessPipeline:
    def test_small_image_passes_through_unchanged_size(self, small_image_bytes):
        # Arrange
        image_bytes = small_image_bytes

        # Act
        image = preprocess(image_bytes)

        # Assert
        assert image.shape == (64, 64, 3)

    def test_large_image_gets_resized(self, large_image_bytes):
        # Arrange
        image_bytes = large_image_bytes

        # Act
        image = preprocess(image_bytes)

        # Assert
        assert max(image.shape[:2]) <= 1280

    def test_raises_invalid_image_error_end_to_end(self):
        # Arrange
        image_bytes = b""

        # Act / Assert
        with pytest.raises(InvalidImageError):
            preprocess(image_bytes)

    def test_custom_max_side_respected(self, large_image_bytes):
        # Arrange
        max_side = 500

        # Act
        image = preprocess(large_image_bytes, max_side=max_side)

        # Assert
        assert max(image.shape[:2]) <= max_side
