from pathlib import Path

import pytest
import ymmsl
from ymmsl.v0_2 import Configuration, Reference, resolve

from imas_streams.muscle3_datasource import DATA_SOURCE

ymmsl_config = """
ymmsl_version: v0.2

description: Test muscle3 datasource actor

imports:
- from imas_streams.data_source import implementation imas_streams_source

models:
    test:
        description: Simple test model
        components:
            source:
                implementation: imas_streams_source
                description: Data source
                ports:
                    o_i: ids_out
                    s: trigger

resources:
    source:
        threads: 1
"""


@pytest.mark.xfail(
    tuple(map(int, ymmsl.__version__.split(".")[:3])) < (0, 15, 1),
    reason="Test needs YMMSL Entry Points plugins",
)
def test_load_ymmsl_config():
    config = ymmsl.load_as(Configuration, ymmsl_config)
    resolve(Reference([]), config)
    config.check_consistent()


def test_load_ymmsl_config_from_ymmsl_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Prepare yMMSL path
    ymmsl_file = tmp_path / "imas_streams" / "data_source.ymmsl"
    ymmsl_file.parent.mkdir(parents=True, exist_ok=True)
    ymmsl_file.write_text(DATA_SOURCE)
    monkeypatch.setenv("YMMSL_PATH", str(tmp_path))

    config = ymmsl.load_as(Configuration, ymmsl_config)
    resolve(Reference([]), config)
    config.check_consistent()
