"""Tests for common.types module."""

from __future__ import annotations

from deriva.common.types import ErrorContext, ProgressUpdate, create_error


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


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_str_message_only(self):
        """Should format with just message."""
        ctx = ErrorContext(message="Something went wrong")
        assert str(ctx) == "Something went wrong"

    def test_str_with_repo(self):
        """Should include repo in string."""
        ctx = ErrorContext(message="Error", repo_name="my-repo")
        result = str(ctx)
        assert "Error" in result
        assert "repo=my-repo" in result

    def test_str_with_all_fields(self):
        """Should include all fields in string."""
        ctx = ErrorContext(
            message="Failed",
            repo_name="repo",
            step_name="TypeDef",
            phase_name="extraction",
            file_path="/src/test.py",
            batch_number=3,
            exception_type="ValueError",
        )
        result = str(ctx)
        assert "Failed" in result
        assert "repo=repo" in result
        assert "step=TypeDef" in result
        assert "phase=extraction" in result
        assert "file=/src/test.py" in result
        assert "batch=3" in result
        assert "exception=ValueError" in result

    def test_to_dict_minimal(self):
        """Should create dict with just message."""
        ctx = ErrorContext(message="Error occurred")
        d = ctx.to_dict()
        assert d["message"] == "Error occurred"
        assert d["recoverable"] is True
        assert "repo_name" not in d

    def test_to_dict_with_all_fields(self):
        """Should include all set fields in dict."""
        ctx = ErrorContext(
            message="Failed",
            repo_name="repo",
            step_name="Step",
            phase_name="Phase",
            file_path="/path",
            batch_number=5,
            exception_type="TypeError",
            recoverable=False,
        )
        d = ctx.to_dict()
        assert d["message"] == "Failed"
        assert d["repo_name"] == "repo"
        assert d["step_name"] == "Step"
        assert d["phase_name"] == "Phase"
        assert d["file_path"] == "/path"
        assert d["batch_number"] == 5
        assert d["exception_type"] == "TypeError"
        assert d["recoverable"] is False


class TestCreateError:
    """Tests for create_error function."""

    def test_simple_message(self):
        """Should create simple error string."""
        result = create_error("Something failed")
        assert "Something failed" in result

    def test_with_context(self):
        """Should include context in error string."""
        result = create_error(
            "Failed to process",
            repo_name="test-repo",
            step_name="Extraction",
        )
        assert "Failed to process" in result
        assert "repo=test-repo" in result
        assert "step=Extraction" in result

    def test_with_exception(self):
        """Should extract exception type."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = create_error("Caught error", exception=e)
        assert "Caught error" in result
        assert "exception=ValueError" in result

    def test_with_batch(self):
        """Should include batch number."""
        result = create_error("Batch error", batch_number=42)
        assert "batch=42" in result
