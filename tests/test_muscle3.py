from pathlib import Path
from unittest.mock import Mock, patch

import imas
import pytest
import ymmsl
from ymmsl.v0_2 import Configuration, Operator, Reference, resolve

from imas_streams import StreamingIDSConsumer, StreamingIDSProducer
from imas_streams.muscle3_config import DATA_SOURCE
from imas_streams.muscle3_datasource import DynamicDataSource

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
    tuple(map(int, ymmsl.__version__.partition("-")[0].split(".")[:3])) < (0, 15, 1),
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


def test_dynamic_port_configuration():
    with patch("imas_streams.muscle3_datasource.Instance") as mock:
        list_ports = mock.return_value.list_ports
        list_ports.return_value = {}
        with pytest.raises(RuntimeError, match="needs at least one"):
            DynamicDataSource()
        list_ports.return_value = {Operator.F_INIT: ["test"]}
        with pytest.raises(RuntimeError, match="does not support F_INIT ports"):
            DynamicDataSource()
        list_ports.return_value = {Operator.O_F: ["test"]}
        with pytest.raises(RuntimeError, match="does not support O_F ports"):
            DynamicDataSource()
        list_ports.return_value = {Operator.O_I: ["out"], Operator.S: ["in"]}
        DynamicDataSource()


def test_dynamic_port_topics(caplog: pytest.LogCaptureFixture):
    caplog.set_level("INFO")
    with patch("imas_streams.muscle3_datasource.Instance") as mock:
        list_ports = mock.return_value.list_ports
        list_ports.return_value = {Operator.O_I: ["out"], Operator.S: ["in"]}

        source = DynamicDataSource()
        with pytest.raises(RuntimeError, match="Invalid line"):
            source._parse_topics("invalid line")
        with pytest.raises(RuntimeError, match="Invalid line"):
            source._parse_topics("valid: x\ninvalid")
        with pytest.raises(RuntimeError, match="topic is missing for input"):
            source._parse_topics("a:x\nout:ok")
        with pytest.raises(RuntimeError, match="topic is missing for output"):
            source._parse_topics("a:x\nin:ok")

        parsed = source._parse_topics("in:topic1\nout:topic2")
        assert parsed == {"in": "topic1", "out": "topic2"}

        caplog.clear()
        parsed = source._parse_topics("""
                in  :   topic1
            out :           this.is.a.longer:name.for.the.topic  
            ignored: this.should.be.logged
        """)
        assert parsed == {"in": "topic1", "out": "this.is.a.longer:name.for.the.topic"}
        assert len(caplog.record_tuples) == 1
        assert "Ignoring kafka topic" in caplog.record_tuples[0][2]


class MockKafkaConsumer:
    def __init__(self, times: list[float]) -> None:
        self.times = times
        self.ids = imas.IDSFactory().equilibrium()
        self.ids.ids_properties.homogeneous_time = 1
        self.ids.time = [0.0]
        self.producer = StreamingIDSProducer(self.ids)
        self.consumer = StreamingIDSConsumer(self.producer.metadata, return_copy=False)

    def stream(self):
        for time in self.times:
            self.ids.time = [float(time)]
            message = self.producer.create_message(self.ids)
            yield self.consumer.process_message(message)


def test_dynamic_ids_synchronization():
    self = Mock()
    self.consumers = {
        "main": MockKafkaConsumer([0, 1, 2, 3, 4, 5]),
        "lockstep": MockKafkaConsumer([0, 1, 2, 3, 4, 5]),
        "slower": MockKafkaConsumer([0, 4, 8]),
        "faster": MockKafkaConsumer([0, 0.8, 1.6, 2.4, 3.2, 4, 4.8, 5.6]),
    }

    def extract_times(data: dict[str, tuple[float, bytes]]) -> dict[str, float]:
        return {k: v[0] for k, v in data.items()}

    generated = [x.copy() for x in DynamicDataSource.generate_serialized_idss(self)]
    assert len(generated) == 6
    assert extract_times(generated[0]) == dict(main=0, lockstep=0, slower=0, faster=0)
    assert extract_times(generated[1]) == dict(main=1, lockstep=1, slower=0, faster=0.8)
    assert extract_times(generated[2]) == dict(main=2, lockstep=2, slower=0, faster=1.6)
    assert extract_times(generated[3]) == dict(main=3, lockstep=3, slower=0, faster=2.4)
    assert extract_times(generated[4]) == dict(main=4, lockstep=4, slower=4, faster=4)
    assert extract_times(generated[5]) == dict(main=5, lockstep=5, slower=4, faster=4.8)
