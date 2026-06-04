import contextlib
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import confluent_kafka.admin
import imas
import numpy as np
import pytest
import ymmsl
from libmuscle import Message
from ymmsl.v0_2 import Configuration, Operator, Reference, resolve

from imas_streams import StreamingIDSConsumer, StreamingIDSProducer
from imas_streams.kafka import KafkaProducer, KafkaSettings
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
    """Mock object for KafkaConsumer to test IDS synchronization"""

    def __init__(self, times: list[float]) -> None:
        """Mock Kafka stream with empty equilibrium IDSs at the provided time points."""
        self.times = times
        self.ids = imas.IDSFactory().equilibrium()
        self.ids.ids_properties.homogeneous_time = 1
        self.ids.time = [0.0]
        self.producer = StreamingIDSProducer(self.ids)
        self.consumer = StreamingIDSConsumer(self.producer.metadata, return_copy=False)

    def stream(self, timeout=None):
        for time in self.times:
            self.ids.time = [float(time)]
            message = self.producer.create_message(self.ids)
            yield self.consumer.process_message(message)


def extract_times(data: dict[str, tuple[float, bytes]]) -> dict[str, float]:
    """Extract time values from the yielded data of
    DynamicDataSource.generate_serialized_idss."""
    return {k: v[0] for k, v in data.items()}


def test_dynamic_ids_synchronization():
    self = Mock()
    self.consumers = {
        "main": MockKafkaConsumer([0, 1, 2, 3, 4, 5]),
        "lockstep": MockKafkaConsumer([0, 1, 2, 3, 4, 5]),
        "slower": MockKafkaConsumer([0, 4, 8]),
        "faster": MockKafkaConsumer([0, 0.8, 1.6, 2.4, 3.2, 4, 4.8, 5.6]),
    }

    generated = [
        extract_times(x) for x in DynamicDataSource.generate_serialized_idss(self)
    ]
    assert generated == [
        dict(main=0, lockstep=0, slower=0, faster=0),
        dict(main=1, lockstep=1, slower=0, faster=0.8),
        dict(main=2, lockstep=2, slower=0, faster=1.6),
        dict(main=3, lockstep=3, slower=0, faster=2.4),
        dict(main=4, lockstep=4, slower=4, faster=4),
        dict(main=5, lockstep=5, slower=4, faster=4.8),
    ]


def test_dynamic_ids_synchronization_with_offset():
    self = Mock()
    self.consumers = {
        "main": MockKafkaConsumer([0, 1, 2, 3]),
        "delayed": MockKafkaConsumer([1.5, 2, 2.5]),
        "early": MockKafkaConsumer([-1, 1, 3]),
    }

    generated = [
        extract_times(x) for x in DynamicDataSource.generate_serialized_idss(self)
    ]
    assert generated == [
        dict(main=2, delayed=2, early=1),
        dict(main=3, delayed=2.5, early=3),
    ]


def test_dynamic_ids_synchronization_with_offset2():
    self = Mock()
    self.consumers = {
        # Now delayed will be the 'main' stream for determining time output!
        "delayed": MockKafkaConsumer([1.5, 2, 2.5]),
        "main": MockKafkaConsumer([0, 1, 2, 3]),
        "early": MockKafkaConsumer([-1, 1, 3]),
    }

    generated = [
        extract_times(x) for x in DynamicDataSource.generate_serialized_idss(self)
    ]
    assert generated == [
        dict(main=1, delayed=1.5, early=1),
        dict(main=2, delayed=2, early=1),
        dict(main=2, delayed=2.5, early=1),
    ]


def test_dynamic_data_source_actor(muscle3_tester, kafka_host):
    # Ensure topics are cleared before start
    with confluent_kafka.admin.AdminClient({"bootstrap.servers": kafka_host}) as client:
        fs = client.delete_topics(
            ["test.magnetics", "test.pf_active", "test.equilibrium"]
        )
        for _topic, future in fs.items():
            # Raises an exception when the topic did not exists or could not be deleted
            with contextlib.suppress(confluent_kafka.KafkaException):
                future.result()

    # Populate magnetics and pf_active topics
    times = {"magnetics": np.linspace(0, 10, 11), "pf_active": np.linspace(-1, 12, 27)}
    for ids_name in ["magnetics", "pf_active"]:
        ids = imas.IDSFactory().new(ids_name)
        ids.ids_properties.homogeneous_time = 1
        ids.time = times[ids_name][:1]
        prod = StreamingIDSProducer(ids)
        kprod = KafkaProducer(
            KafkaSettings(host=kafka_host, topic_name=f"test.{ids_name}"),
            prod.metadata,
        )
        for t in times[ids_name]:
            ids.time = [t]
            kprod.produce(bytes(prod.create_message(ids)))
        del kprod  # Run cleanup logic

    # Start muscle3 actor
    tester = muscle3_tester.start_implementation(
        f"""
        ymmsl_version: v0.2
        programs:
          imas_streams:
            ports:
              o_i: magnetics_out pf_active_out
              s: equilibrium_in
            executable: {sys.executable}
            args: -m imas_streams dynamic-kafka-to-muscle3
        settings:
          kafka_host: {kafka_host}
          kafka_timeout: 10.0
          kafka_topics: |
            magnetics_out: test.magnetics
            pf_active_out: test.pf_active
            equilibrium_in: test.equilibrium
        """,
        "imas_streams",
        default_timeout=20,
    )

    mag, pfa, eq = map(imas.IDSFactory().new, ["magnetics", "pf_active", "equilibrium"])
    eq.ids_properties.homogeneous_time = 1
    for expected_time in np.linspace(0, 10, 11):
        mag.deserialize(tester.receive("magnetics_out").data)
        assert np.array_equal(mag.time, [expected_time])
        pfa.deserialize(tester.receive("pf_active_out").data)
        assert np.array_equal(pfa.time, [expected_time])
        eq.time = [expected_time]
        tester.send("equilibrium_in", Message(expected_time, data=eq.serialize()))

    # Check that the messages were sent to Kafka
    consumer = confluent_kafka.Consumer(
        {
            "bootstrap.servers": kafka_host,
            "group.id": "pytest",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe(["test.equilibrium"])
    for expected_time in np.linspace(0, 10, 11):
        msg = consumer.poll(10)
        assert msg is not None, f"Missing message for t={expected_time}"
        data = msg.value()
        assert data is not None
        eq.deserialize(data)
        assert np.array_equal(eq.time, [expected_time])
