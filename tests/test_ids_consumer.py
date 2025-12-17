import os

import imas
import numpy as np
import pytest

from imas_streams import StreamingIDSConsumer, StreamingIMASMetadata
from imas_streams.metadata import DynamicData

DD_VERSION = os.getenv("IMAS_VERSION", "4.0.0")


@pytest.fixture
def magnetics_metadata():
    ids = imas.IDSFactory(DD_VERSION).new("magnetics")

    ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [0.0]

    ids.flux_loop.resize(5)
    for i, loop in enumerate(ids.flux_loop):
        loop.name = f"flux_loop_{i}"
        loop.position.resize(1)
        loop.position[0].r = i / 2
        loop.position[0].z = i / 2

    return StreamingIMASMetadata(
        data_dictionary_version=DD_VERSION,
        ids_name="magnetics",
        static_data=ids,
        dynamic_data=[
            DynamicData(path="time", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[0]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[1]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[2]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[3]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[4]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[0]/voltage/data", shape=(1,), data_type="f64"),
        ],
    )


def test_ids_consumer(magnetics_metadata):
    consumer = StreamingIDSConsumer(magnetics_metadata)

    data = np.arange(7, dtype="<f8")
    ids = consumer.process_message(data.tobytes())

    assert np.array_equal(ids.time, [0.0])
    assert np.array_equal(ids.flux_loop[0].flux.data, [1.0])
    assert np.array_equal(ids.flux_loop[1].flux.data, [2.0])
    assert np.array_equal(ids.flux_loop[2].flux.data, [3.0])
    assert np.array_equal(ids.flux_loop[3].flux.data, [4.0])
    assert np.array_equal(ids.flux_loop[4].flux.data, [5.0])
    assert np.array_equal(ids.flux_loop[0].voltage.data, [6.0])

    # Test error when processing invalid size messages
    with pytest.raises(ValueError):
        consumer.process_message(b"")
    with pytest.raises(ValueError):
        consumer.process_message(b"1234")


def test_streaming_reader_copy(magnetics_metadata):
    reader = StreamingIDSConsumer(magnetics_metadata)
    test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8")

    ids = reader.process_message(test_data.tobytes())
    # Verify that all fields have the correct value
    for i, dyndata in enumerate(magnetics_metadata.dynamic_data):
        assert np.array_equal(ids[dyndata.path], [test_data[i]])

    # Make some changes to the IDS
    assert ids.time[0] == 0.0
    ids.time[0] = 1.0
    assert ids.time[0] == 1.0
    assert len(ids.flux_loop) == 5
    ids.flux_loop.resize(0)
    assert len(ids.flux_loop) == 0

    # And check that we get a fresh IDS after another call to process_message
    ids2 = reader.process_message(test_data.tobytes())
    assert ids is not ids2
    assert ids2.time[0] == 0.0
    assert len(ids2.flux_loop) == 5


def test_streaming_reader_nocopy(magnetics_metadata):
    reader = StreamingIDSConsumer(magnetics_metadata)
    test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8")

    ids = reader.process_message(test_data.tobytes(), return_copy=False)
    # Check that we're not allowed to alter array values:
    with pytest.raises(ValueError, match="read-only"):
        ids.time[0] = 1.0

    # Make some changes to the IDS
    assert ids.time[0] == 0.0
    ids.time = [1.0]
    assert ids.time[0] == 1.0
    assert len(ids.flux_loop) == 5
    ids.flux_loop.resize(0)
    assert len(ids.flux_loop) == 0

    # And check that our changes broke everything
    ids2 = reader.process_message(test_data.tobytes(), return_copy=False)
    assert ids is ids2
    assert ids2.time[0] == 1.0
    assert len(ids2.flux_loop) == 0

    # Even when requesting a copy later
    ids2 = reader.process_message(test_data.tobytes(), return_copy=True)
    assert ids is not ids2
    assert ids2.time[0] == 1.0
    assert len(ids2.flux_loop) == 0
