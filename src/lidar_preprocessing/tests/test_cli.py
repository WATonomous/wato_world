import inspect

from click.testing import CliRunner

from wato_lidar_preprocessing.cli import main


def test_help_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


def test_viz_options_match_callback_parameters():
    command = main.commands["viz"]
    option_names = {param.name for param in command.params}
    callback_names = set(inspect.signature(command.callback).parameters)

    assert option_names == callback_names


def test_viz_defaults_to_one_html_backend_option():
    command = main.commands["viz"]
    backend_options = [param for param in command.params if param.name == "backend"]

    assert len(backend_options) == 1
    assert backend_options[0].default == "html"
