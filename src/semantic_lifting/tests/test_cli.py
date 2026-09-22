"""CLI smoke tests for semantic_lifting."""

from __future__ import annotations

from click.testing import CliRunner

from wato_semantic_lifting.cli import main


def test_help_exits_zero():
    """--help must exit 0 and print usage."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_run_help_exits_zero():
    """run --help must exit 0 and show bag option."""
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--bag" in result.output
