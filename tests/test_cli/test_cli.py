"""Tests for cli.cli module (typer-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from deriva.cli.cli import app, main
from deriva.cli.commands.benchmark import _get_run_stats_from_ocel
from deriva.cli.commands.run import (
    _print_derivation_result,
    _print_extraction_result,
    _print_pipeline_result,
)

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_main_help(self):
        """Should show main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Deriva CLI" in result.stdout

    def test_run_help(self):
        """Should show run command help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Pipeline stage" in result.stdout

    def test_config_help(self):
        """Should show config command help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "enable" in result.stdout

    def test_repo_help(self):
        """Should show repo command help."""
        result = runner.invoke(app, ["repo", "--help"])
        assert result.exit_code == 0
        assert "clone" in result.stdout
        assert "list" in result.stdout

    def test_benchmark_help(self):
        """Should show benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "analyze" in result.stdout


class TestPrintExtractionResult:
    """Tests for _print_extraction_result helper."""

    def test_prints_stats(self, capsys):
        """Should print extraction statistics."""
        result = {
            "stats": {
                "repos_processed": 2,
                "nodes_created": 100,
                "edges_created": 50,
                "steps_completed": 5,
                "steps_skipped": 1,
            }
        }
        _print_extraction_result(result)
        output = capsys.readouterr().out

        assert "EXTRACTION RESULTS" in output
        assert "Repos processed:  2" in output
        assert "Nodes created:    100" in output

    def test_prints_errors(self, capsys):
        """Should print errors when present."""
        result = {
            "stats": {},
            "errors": ["Error 1", "Error 2"],
        }
        _print_extraction_result(result)
        output = capsys.readouterr().out

        assert "Errors (2)" in output
        assert "Error 1" in output

    def test_prints_warnings(self, capsys):
        """Should print warnings when present."""
        result = {
            "stats": {},
            "warnings": ["Warning 1", "Warning 2", "Warning 3"],
        }
        _print_extraction_result(result)
        output = capsys.readouterr().out

        assert "Warnings (3)" in output
        assert "Warning 1" in output

    def test_truncates_many_warnings(self, capsys):
        """Should truncate when more than 5 warnings."""
        result = {
            "stats": {},
            "warnings": [f"Warning {i}" for i in range(10)],
        }
        _print_extraction_result(result)
        output = capsys.readouterr().out

        assert "... and 5 more" in output

    def test_truncates_many_errors(self, capsys):
        """Should truncate when more than 5 errors."""
        result = {
            "stats": {},
            "errors": [f"Error {i}" for i in range(10)],
        }
        _print_extraction_result(result)
        output = capsys.readouterr().out

        assert "... and 5 more" in output


class TestPrintDerivationResult:
    """Tests for _print_derivation_result helper."""

    def test_prints_stats(self, capsys):
        """Should print derivation statistics."""
        result = {
            "stats": {
                "elements_created": 10,
                "relationships_created": 20,
                "elements_validated": 30,
                "issues_found": 2,
                "steps_completed": 5,
            }
        }
        _print_derivation_result(result)
        output = capsys.readouterr().out

        assert "DERIVATION RESULTS" in output
        assert "Elements created:      10" in output
        assert "Relationships created: 20" in output

    def test_prints_issues(self, capsys):
        """Should print issues when present."""
        result = {
            "stats": {},
            "issues": [
                {"severity": "warning", "message": "Missing relationship"},
            ],
        }
        _print_derivation_result(result)
        output = capsys.readouterr().out

        assert "Issues (1)" in output
        assert "[WARNING]" in output

    def test_prints_errors(self, capsys):
        """Should print errors when present."""
        result = {
            "stats": {},
            "errors": ["Error 1", "Error 2"],
        }
        _print_derivation_result(result)
        output = capsys.readouterr().out

        assert "Errors (2)" in output

    def test_truncates_many_issues(self, capsys):
        """Should truncate when more than 10 issues."""
        result = {
            "stats": {},
            "issues": [{"severity": "warning", "message": f"Issue {i}"} for i in range(15)],
        }
        _print_derivation_result(result)
        output = capsys.readouterr().out

        assert "... and 5 more" in output


class TestPrintPipelineResult:
    """Tests for _print_pipeline_result helper."""

    def test_prints_pipeline_summary(self, capsys):
        """Should print pipeline summary."""
        result = {
            "results": {
                "extraction": {"stats": {"nodes_created": 50}},
                "derivation": {"stats": {"elements_created": 10, "issues_found": 1}},
            }
        }
        _print_pipeline_result(result)
        output = capsys.readouterr().out

        assert "PIPELINE COMPLETE" in output
        assert "Extraction:" in output
        assert "Nodes created: 50" in output
        assert "Derivation:" in output
        assert "Elements created: 10" in output

    def test_prints_classification_results(self, capsys):
        """Should print classification results when present."""
        result = {
            "results": {
                "classification": {"stats": {"files_classified": 100, "files_undefined": 5}},
            }
        }
        _print_pipeline_result(result)
        output = capsys.readouterr().out

        assert "Classification:" in output
        assert "Files classified: 100" in output

    def test_prints_total_errors(self, capsys):
        """Should print total errors when present."""
        result = {
            "results": {},
            "errors": ["err1", "err2", "err3"],
        }
        _print_pipeline_result(result)
        output = capsys.readouterr().out

        assert "Total errors: 3" in output


class TestMain:
    """Tests for main entry point."""

    def test_main_returns_zero_on_success(self):
        """Should return 0 on success."""
        with patch("deriva.cli.cli.app") as mock_app:
            mock_app.return_value = None
            result = main()
        assert result == 0

    def test_main_returns_exit_code(self):
        """Should return exit code on SystemExit."""
        with patch("deriva.cli.cli.app") as mock_app:
            mock_app.side_effect = SystemExit(1)
            result = main()
        assert result == 1


class TestConfigListCommand:
    """Tests for config list command."""

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_lists_extraction_configs(self, mock_session_class):
        """Should list extraction configurations."""
        mock_session = MagicMock()
        mock_session.list_steps.return_value = [
            {"name": "BusinessConcept", "enabled": True, "sequence": 1},
            {"name": "TypeDefinition", "enabled": False, "sequence": 2},
        ]
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "list", "extraction"])

        assert result.exit_code == 0
        assert "EXTRACTION CONFIGURATIONS" in result.stdout
        assert "BusinessConcept" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_shows_message_when_no_configs(self, mock_session_class):
        """Should show message when no configurations found."""
        mock_session = MagicMock()
        mock_session.list_steps.return_value = []
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "list", "derivation"])

        assert result.exit_code == 0
        assert "No derivation configurations found" in result.stdout

    def test_invalid_step_type(self):
        """Should reject invalid step type."""
        result = runner.invoke(app, ["config", "list", "invalid"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestConfigEnableDisableCommand:
    """Tests for config enable and disable commands."""

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_enable_step_success(self, mock_session_class):
        """Should enable step successfully."""
        mock_session = MagicMock()
        mock_session.enable_step.return_value = True
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "enable", "extraction", "BusinessConcept"])

        assert result.exit_code == 0
        assert "Enabled" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_enable_step_not_found(self, mock_session_class):
        """Should return error when step not found."""
        mock_session = MagicMock()
        mock_session.enable_step.return_value = False
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "enable", "extraction", "NonExistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_disable_step_success(self, mock_session_class):
        """Should disable step successfully."""
        mock_session = MagicMock()
        mock_session.disable_step.return_value = True
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "disable", "derivation", "ApplicationComponent"])

        assert result.exit_code == 0
        assert "Disabled" in result.stdout


class TestClearCommand:
    """Tests for clear command."""

    @patch("deriva.cli.cli.PipelineSession")
    def test_clear_graph_success(self, mock_session_class):
        """Should clear graph successfully."""
        mock_session = MagicMock()
        mock_session.clear_graph.return_value = {"success": True, "message": "Graph cleared"}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["clear", "graph"])

        assert result.exit_code == 0
        assert "Graph cleared" in result.stdout

    @patch("deriva.cli.cli.PipelineSession")
    def test_clear_model_success(self, mock_session_class):
        """Should clear model successfully."""
        mock_session = MagicMock()
        mock_session.clear_model.return_value = {"success": True, "message": "Model cleared"}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["clear", "model"])

        assert result.exit_code == 0

    def test_clear_invalid_target(self):
        """Should reject invalid target."""
        result = runner.invoke(app, ["clear", "invalid"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestStatusCommand:
    """Tests for status command."""

    @patch("deriva.cli.cli.PipelineSession")
    def test_shows_status(self, mock_session_class):
        """Should show pipeline status."""
        mock_session = MagicMock()
        mock_session.list_steps.return_value = [{"enabled": True}, {"enabled": False}]
        mock_session.get_file_types.return_value = ["py", "js", "ts"]
        mock_session.get_graph_stats.return_value = {"total_nodes": 100}
        mock_session.get_archimate_stats.return_value = {"total_elements": 50}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "DERIVA STATUS" in result.stdout
        assert "1/2 steps enabled" in result.stdout

    @patch("deriva.cli.cli.PipelineSession")
    def test_handles_graph_connection_error(self, mock_session_class):
        """Should handle graph connection error gracefully."""
        mock_session = MagicMock()
        mock_session.list_steps.return_value = []
        mock_session.get_file_types.return_value = []
        mock_session.get_graph_stats.side_effect = Exception("Not connected")
        mock_session.get_archimate_stats.side_effect = Exception("Not connected")
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "(not connected)" in result.stdout


class TestExportCommand:
    """Tests for export command."""

    @patch("deriva.cli.cli.PipelineSession")
    def test_export_success(self, mock_session_class):
        """Should export model successfully."""
        mock_session = MagicMock()
        mock_session.export_model.return_value = {
            "success": True,
            "elements_exported": 50,
            "relationships_exported": 30,
            "output_path": "/path/to/model.xml",
        }
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["export", "-o", "out.xml", "-n", "MyModel"])

        assert result.exit_code == 0
        assert "Elements exported: 50" in result.stdout

    @patch("deriva.cli.cli.PipelineSession")
    def test_export_failure(self, mock_session_class):
        """Should return error on export failure."""
        mock_session = MagicMock()
        mock_session.export_model.return_value = {"success": False, "error": "No elements"}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["export"])

        assert result.exit_code == 1


class TestRepoCommands:
    """Tests for repo commands."""

    @patch("deriva.cli.commands.repo.PipelineSession")
    def test_repo_clone_success(self, mock_session_class):
        """Should clone repository successfully."""
        mock_session = MagicMock()
        mock_session.clone_repository.return_value = {
            "success": True,
            "name": "my_repo",
            "path": "/workspace/repos/my_repo",
            "url": "https://github.com/user/repo",
        }
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["repo", "clone", "https://github.com/user/repo"])

        assert result.exit_code == 0
        assert "cloned successfully" in result.stdout

    @patch("deriva.cli.commands.repo.PipelineSession")
    def test_repo_list(self, mock_session_class):
        """Should list repositories."""
        mock_session = MagicMock()
        mock_session.workspace_dir = "/workspace"
        mock_session.get_repositories.return_value = [
            {"name": "repo1"},
            {"name": "repo2"},
        ]
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["repo", "list"])

        assert result.exit_code == 0
        assert "REPOSITORIES" in result.stdout
        assert "repo1" in result.stdout
        assert "Total: 2 repositories" in result.stdout

    @patch("deriva.cli.commands.repo.PipelineSession")
    def test_repo_delete_success(self, mock_session_class):
        """Should delete repository successfully."""
        mock_session = MagicMock()
        mock_session.delete_repository.return_value = {"success": True}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["repo", "delete", "my_repo"])

        assert result.exit_code == 0
        assert "deleted successfully" in result.stdout

    @patch("deriva.cli.commands.repo.PipelineSession")
    def test_repo_info(self, mock_session_class):
        """Should show repository info."""
        mock_session = MagicMock()
        mock_session.get_repository_info.return_value = {
            "name": "my_repo",
            "path": "/workspace/repos/my_repo",
            "url": "https://github.com/user/repo",
            "branch": "main",
            "last_commit": "abc123",
            "is_dirty": False,
            "size_mb": 15.5,
            "cloned_at": "2024-01-01",
        }
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["repo", "info", "my_repo"])

        assert result.exit_code == 0
        assert "REPOSITORY: my_repo" in result.stdout


class TestRunCommand:
    """Tests for run command."""

    @patch("deriva.cli.cli.create_progress_reporter")
    @patch("deriva.cli.cli.PipelineSession")
    def test_run_extraction(self, mock_session_class, mock_progress):
        """Should run extraction stage."""
        mock_session = MagicMock()
        mock_session.llm_info = {"provider": "openai", "model": "gpt-4"}
        mock_session.run_extraction.return_value = {
            "success": True,
            "stats": {"nodes_created": 100, "edges_created": 50},
        }
        mock_session_class.return_value.__enter__.return_value = mock_session

        mock_reporter = MagicMock()
        mock_progress.return_value = mock_reporter
        mock_reporter.__enter__ = MagicMock(return_value=mock_reporter)
        mock_reporter.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["run", "extraction"])

        assert result.exit_code == 0
        assert "EXTRACTION" in result.stdout

    @patch("deriva.cli.cli.PipelineSession")
    def test_run_derivation_without_llm(self, mock_session_class):
        """Should return error when running derivation without LLM."""
        mock_session = MagicMock()
        mock_session.llm_info = None
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["run", "derivation"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_run_invalid_stage(self):
        """Should reject invalid stage."""
        result = runner.invoke(app, ["run", "invalid"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_run_invalid_phase_for_extraction(self):
        """Should reject invalid phase for extraction."""
        result = runner.invoke(app, ["run", "extraction", "--phase", "generate"])
        assert result.exit_code == 1
        assert "not valid for extraction" in result.output

    def test_run_invalid_phase_for_derivation(self):
        """Should reject invalid phase for derivation."""
        result = runner.invoke(app, ["run", "derivation", "--phase", "classify"])
        assert result.exit_code == 1
        assert "not valid for derivation" in result.output


class TestBenchmarkCommands:
    """Tests for benchmark commands."""

    @patch("deriva.cli.commands.benchmark.create_benchmark_progress_reporter")
    @patch("deriva.cli.commands.benchmark.PipelineSession")
    def test_benchmark_run_success(self, mock_session_class, mock_progress):
        """Should run benchmark successfully."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.session_id = "bench_123"
        mock_result.runs_completed = 3
        mock_result.runs_failed = 0
        mock_result.duration_seconds = 120.5
        mock_result.ocel_path = "workspace/benchmarks/bench_123/ocel.json"
        mock_result.success = True
        mock_result.errors = []
        mock_session.run_benchmark.return_value = mock_result
        mock_session_class.return_value.__enter__.return_value = mock_session

        mock_reporter = MagicMock()
        mock_progress.return_value = mock_reporter
        mock_reporter.__enter__ = MagicMock(return_value=mock_reporter)
        mock_reporter.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["benchmark", "run", "--repos", "repo1", "--models", "gpt4"])

        assert result.exit_code == 0
        assert "BENCHMARK COMPLETE" in result.stdout

    @patch("deriva.cli.commands.benchmark.PipelineSession")
    def test_benchmark_list(self, mock_session_class):
        """Should list benchmark sessions."""
        mock_session = MagicMock()
        mock_session.list_benchmarks.return_value = [
            {
                "session_id": "bench_001",
                "status": "completed",
                "started_at": "2024-01-01T10:00:00",
                "description": "Test run",
            },
        ]
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["benchmark", "list"])

        assert result.exit_code == 0
        assert "bench_001" in result.stdout

    @patch("deriva.cli.commands.benchmark.PipelineSession")
    def test_benchmark_models_empty(self, mock_session_class):
        """Should show message when no models configured."""
        mock_session = MagicMock()
        mock_session.list_benchmark_models.return_value = {}
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["benchmark", "models"])

        assert result.exit_code == 0
        assert "No benchmark model" in result.stdout


class TestGetRunStatsFromOcel:
    """Tests for _get_run_stats_from_ocel helper."""

    def test_extracts_stats_from_complete_run_events(self):
        """Should extract node/edge counts from CompleteRun events."""
        mock_analyzer = MagicMock()

        event1 = MagicMock()
        event1.activity = "CompleteRun"
        event1.objects = {"Model": ["gpt4"]}
        event1.attributes = {"stats": {"extraction": {"nodes_created": 100, "edges_created": 50}}}

        event2 = MagicMock()
        event2.activity = "CompleteRun"
        event2.objects = {"Model": ["gpt4"]}
        event2.attributes = {"stats": {"extraction": {"nodes_created": 110, "edges_created": 55}}}

        event3 = MagicMock()
        event3.activity = "StartRun"  # Not CompleteRun - should be ignored
        event3.objects = {"Model": ["gpt4"]}

        mock_analyzer.ocel_log.events = [event1, event2, event3]

        result = _get_run_stats_from_ocel(mock_analyzer)

        assert "gpt4" in result
        assert len(result["gpt4"]) == 2
        assert result["gpt4"][0] == (100, 50)
        assert result["gpt4"][1] == (110, 55)

    def test_returns_empty_for_no_complete_run_events(self):
        """Should return empty dict when no CompleteRun events."""
        mock_analyzer = MagicMock()

        event = MagicMock()
        event.activity = "StartRun"
        event.objects = {}

        mock_analyzer.ocel_log.events = [event]

        result = _get_run_stats_from_ocel(mock_analyzer)

        assert result == {}

    def test_skips_events_without_model(self):
        """Should skip events without Model object."""
        mock_analyzer = MagicMock()

        event = MagicMock()
        event.activity = "CompleteRun"
        event.objects = {"Model": [None]}

        mock_analyzer.ocel_log.events = [event]

        result = _get_run_stats_from_ocel(mock_analyzer)

        assert result == {}


class TestFiletypeCommands:
    """Tests for filetype commands."""

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_filetype_list_success(self, mock_session_class):
        """Should list file types."""
        mock_session = MagicMock()
        mock_session.get_file_types.return_value = [
            {"extension": ".py", "file_type": "code", "subtype": "python"},
            {"extension": ".js", "file_type": "code", "subtype": "javascript"},
        ]
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "filetype", "list"])

        assert result.exit_code == 0
        assert "FILE TYPE REGISTRY" in result.stdout
        assert ".py" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_filetype_add_success(self, mock_session_class):
        """Should add file type successfully."""
        mock_session = MagicMock()
        mock_session.add_file_type.return_value = True
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "filetype", "add", ".rs", "code", "rust"])

        assert result.exit_code == 0
        assert "Added file type" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_filetype_delete_success(self, mock_session_class):
        """Should delete file type successfully."""
        mock_session = MagicMock()
        mock_session.delete_file_type.return_value = True
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "filetype", "delete", ".rs"])

        assert result.exit_code == 0
        assert "Deleted file type" in result.stdout

    @patch("deriva.cli.commands.config.PipelineSession")
    def test_filetype_stats(self, mock_session_class):
        """Should show file type statistics."""
        mock_session = MagicMock()
        mock_session.get_file_type_stats.return_value = {
            "code": 50,
            "config": 10,
        }
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = runner.invoke(app, ["config", "filetype", "stats"])

        assert result.exit_code == 0
        assert "FILE TYPE STATISTICS" in result.stdout
