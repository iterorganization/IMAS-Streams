from unittest.mock import patch

import imas
from click.testing import CliRunner
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from imas.util import idsdiffgen

from imas_streams import BatchedIDSConsumer
from imas_streams.cli import main
from imas_streams.kafka import KafkaSettings


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


@patch("imas_streams.kafka.KafkaConsumer")
def test_kafka_to_imasentry(mock_kafkaconsumer, tmp_path):
    # Make some testdata to store
    ids = imas.IDSFactory().core_profiles()
    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [0.0, 0.1, 0.2]
    ids.global_quantities.ip = [0.2, 0.3, 0.4]
    mock_kafkaconsumer.return_value.stream.return_value = [None, None, ids]

    # Run CLI
    runner = CliRunner()
    uri = f"imas:hdf5?path={tmp_path}"
    result = runner.invoke(
        main,
        ["kafka-to-imasentry", "kafka_host:port", "topic_name", uri, "--batch-size=10"],
    )

    assert result.exit_code == 0
    mock_kafkaconsumer.assert_called_once_with(
        KafkaSettings(host="kafka_host:port", topic_name="topic_name"),
        BatchedIDSConsumer,
        batch_size=10,
        return_copy=False,
    )
    # Check that the IDS was stored correctly
    with imas.DBEntry(uri, "r") as entry:
        ids2 = entry.get("core_profiles")
    assert list(idsdiffgen(ids, ids2)) == []
