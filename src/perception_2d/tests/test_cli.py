from click.testing import CliRunner

from wato_perception_2d.cli import main


def test_help_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


def test_refine_flag_reaches_effective_pipeline_config(tmp_path, monkeypatch):
    from wato_perception_2d import cli

    cfg_file = tmp_path / "perception.yaml"
    cfg_file.write_text("segmentation:\n  refinement:\n    enabled: false\n")
    received = []
    monkeypatch.setattr(cli, "run_pipeline", lambda cfg, **kw: received.append(cfg))
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--bag",
            "test",
            "--config",
            str(cfg_file),
            "--refine-masks",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert received[0].segmentation.refinement.enabled
