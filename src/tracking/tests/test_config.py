"""tracking's configuration: the shipped YAML and its bounds."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from wato_tracking.config import ConfirmationConfig, load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "tracking.yaml"


def test_the_shipped_config_loads_with_confirmation_on_at_ms3d_counts():
    c = load_config(str(CONFIG)).confirmation
    assert c.enabled
    assert (
        c.min_confident_vehicle,
        c.min_confident_parked_vehicle,
        c.min_confident_other,
    ) == (7, 3, 1)
    assert c == ConfirmationConfig()


@pytest.mark.parametrize(
    "field",
    ["min_confident_vehicle", "min_confident_parked_vehicle", "min_confident_other"],
)
def test_negative_counts_are_rejected(field):
    with pytest.raises(ValidationError):
        ConfirmationConfig(**{field: -1})
