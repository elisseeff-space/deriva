"""Tests for common.types module."""

from __future__ import annotations

from deriva.common.types import ProgressUpdate


class TestProgressUpdate:
    """Tests for ProgressUpdate dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        update = ProgressUpdate()

        assert update.phase == ""
        assert update.step == ""
        assert update.status == "processing"
        assert update.current == 0
        assert update.total == 0
        assert update.message == ""
        assert update.stats == {}

    def test_str_complete_no_step(self):
        """Should format complete status without step."""
        update = ProgressUpdate(
            phase="extraction",
            status="complete",
            message="10 files processed",
        )

        result = str(update)
        assert "extraction complete" in result
        assert "10 files processed" in result

    def test_str_with_step(self):
        """Should format with step information."""
        update = ProgressUpdate(
            phase="derivation",
            step="ApplicationComponent",
            status="processing",
            current=3,
            total=10,
        )

        result = str(update)
        assert "derivation" in result
        assert "3/10" in result
        assert "ApplicationComponent" in result
        assert "processing" in result

    def test_str_without_step(self):
        """Should format without step."""
        update = ProgressUpdate(
            phase="validation",
            status="starting",
        )

        result = str(update)
        assert "validation" in result
        assert "starting" in result

    def test_with_stats(self):
        """Should accept stats dictionary."""
        update = ProgressUpdate(
            phase="test",
            stats={"elements_created": 5, "time_ms": 100},
        )

        assert update.stats["elements_created"] == 5
        assert update.stats["time_ms"] == 100

    def test_error_status(self):
        """Should handle error status."""
        update = ProgressUpdate(
            phase="extraction",
            step="File",
            status="error",
            current=5,
            total=10,
            message="Failed to process",
        )

        result = str(update)
        assert "error" in result
        assert "extraction" in result

    def test_all_fields_set(self):
        """Should handle all fields being set."""
        update = ProgressUpdate(
            phase="derivation",
            step="BusinessProcess",
            status="complete",
            current=10,
            total=10,
            message="All done",
            stats={"created": 10, "updated": 5},
        )

        assert update.phase == "derivation"
        assert update.step == "BusinessProcess"
        assert update.status == "complete"
        assert update.current == 10
        assert update.total == 10
        assert update.message == "All done"
        assert update.stats == {"created": 10, "updated": 5}
