import os

import imas
import numpy as np
import pytest
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

from imas_streams import BatchedIDSConsumer, StreamingIDSConsumer, StreamingIMASMetadata
from imas_streams.metadata import DynamicData

DD_VERSION = os.getenv("IMAS_VERSION", "4.0.0")


@pytest.fixture
def magnetics_metadata():
    ids = imas.IDSFactory(DD_VERSION).new("magnetics")

    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
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
    reader = StreamingIDSConsumer(magnetics_metadata, return_copy=False)
    test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8")

    ids = reader.process_message(test_data.tobytes())
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
    ids2 = reader.process_message(test_data.tobytes())
    assert ids is ids2
    assert ids2.time[0] == 1.0
    assert len(ids2.flux_loop) == 0


@pytest.mark.parametrize("batch_size", [1, 2, 5, 7, 10, 13, 20])
def test_batched_reader(magnetics_metadata, batch_size):
    reader = BatchedIDSConsumer(magnetics_metadata, batch_size)

    def check_data(ids, expected_time):
        assert np.array_equal(ids.time, expected_time)
        assert np.array_equal(ids.flux_loop[0].flux.data, expected_time + 1)
        assert np.array_equal(ids.flux_loop[1].flux.data, expected_time + 2)
        assert np.array_equal(ids.flux_loop[2].flux.data, expected_time + 3)
        assert np.array_equal(ids.flux_loop[3].flux.data, expected_time + 4)
        assert np.array_equal(ids.flux_loop[4].flux.data, expected_time + 5)
        assert np.array_equal(ids.flux_loop[0].voltage.data, expected_time + 6)

    # Pretend sending 20 messages
    for i in range(20):
        test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8") + i
        ids = reader.process_message(test_data.tobytes())
        # Only expect a result after batch_size items are processed
        if ids is None:
            assert (i + 1) % batch_size != 0
            continue
        expected_time = np.arange(batch_size, dtype=float) + (i + 1 - batch_size)
        check_data(ids, expected_time)

    # Check that any remainders are handled
    msg_remaining = 20 % batch_size
    ids = reader.finalize()
    if ids is None:
        assert msg_remaining == 0
    else:
        assert len(ids.time) == msg_remaining != 0
        expected_time = np.arange(msg_remaining, dtype=float) + 20 - msg_remaining
        check_data(ids, expected_time)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 7, 10, 13, 20])
def test_batched_reader_cp(batch_size):
    ids = imas.IDSFactory(DD_VERSION).new("core_profiles")

    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [0.0]
    ids.profiles_1d.resize(1)
    ids.profiles_1d[0].grid.rho_tor_norm = np.linspace(0, 1, 6)
    ids.profiles_1d[0].ion.resize(1)
    ids.profiles_1d[0].ion[0].z_ion = 1

    metadata = StreamingIMASMetadata(
        data_dictionary_version=DD_VERSION,
        ids_name="magnetics",
        static_data=ids,
        dynamic_data=[
            DynamicData(path="time", shape=(1,), data_type="f64"),
            DynamicData(path="profiles_1d[0]/zeff", shape=(6,), data_type="f64"),
            DynamicData(
                path="profiles_1d[0]/ion[0]/density", shape=(6,), data_type="f64"
            ),
            DynamicData(path="global_quantities/ip", shape=(1,), data_type="f64"),
        ],
    )
    reader = BatchedIDSConsumer(metadata, batch_size)

    def check_data(ids, expected_time):
        assert np.array_equal(ids.time, expected_time)
        assert np.array_equal(ids.global_quantities.ip, expected_time + 13)
        # Check dynamic AoS
        assert len(ids.profiles_1d) == len(expected_time)
        for j, p1d in enumerate(ids.profiles_1d):
            assert np.array_equal(p1d.zeff, np.arange(6) + expected_time[j] + 1)
            assert np.array_equal(
                p1d.ion[0].density, np.arange(6) + expected_time[j] + 7
            )
            assert p1d.ion[0].z_ion == 1

    # Pretend sending 20 messages
    for i in range(20):
        test_data = np.arange(14, dtype="<f8") + i
        ids = reader.process_message(test_data.tobytes())
        # Only expect a result after batch_size items are processed
        if ids is None:
            assert (i + 1) % batch_size != 0
            continue
        expected_time = np.arange(batch_size, dtype=float) + (i + 1 - batch_size)
        check_data(ids, expected_time)

    # Check that any remainders are handled
    msg_remaining = 20 % batch_size
    ids = reader.finalize()
    if ids is None:
        assert msg_remaining == 0
    else:
        assert len(ids.time) == msg_remaining != 0
        expected_time = np.arange(msg_remaining, dtype=float) + 20 - msg_remaining
        check_data(ids, expected_time)
