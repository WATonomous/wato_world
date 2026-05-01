from click.testing import CliRunner

from wato_open_vocab_discovery.cli import main


def test_help_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
