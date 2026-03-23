import os

import imas
import numpy as np
import pytest

from imas_streams import StreamingIMASMetadata
from imas_streams.metadata import DynamicData
from imas_streams.xarray_consumers import StreamingXArrayConsumer

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


def test_xarray_consumer(magnetics_metadata):
    consumer = StreamingXArrayConsumer(magnetics_metadata)

    data = np.arange(7, dtype="<f8")
    dataset = consumer.process_message(data.tobytes())
    assert dataset is not None

    assert np.array_equal(dataset.time, [0.0])
    assert np.array_equal(
        dataset["flux_loop.flux.data"],
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
    )
    assert np.array_equal(
        dataset["flux_loop.voltage.data"],
        [[6.0]] + [[np.nan]] * 4,
        equal_nan=True,
    )

    # Test error when processing invalid size messages
    with pytest.raises(ValueError):
        consumer.process_message(b"")
    with pytest.raises(ValueError):
        consumer.process_message(b"1234")


def test_xarray_consumer_shuffled_aos(magnetics_metadata):
    # Keep time as first item, but the rest is randomized
    # Also, there's no voltage data for flux loops 0, 2 and 4
    magnetics_metadata.dynamic_data = [
        DynamicData(path="time", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[1]/voltage/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[4]/flux/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[0]/flux/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[1]/flux/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[3]/voltage/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[3]/flux/data", shape=(1,), data_type="f64"),
        DynamicData(path="flux_loop[2]/flux/data", shape=(1,), data_type="f64"),
    ]
    consumer = StreamingXArrayConsumer(magnetics_metadata)

    data = np.arange(8, dtype="<f8")
    dataset = consumer.process_message(data.tobytes())
    assert dataset is not None

    assert np.array_equal(dataset.time, [0.0])
    assert np.array_equal(
        dataset["flux_loop.flux.data"],
        [[3.0], [4.0], [7.0], [6.0], [2.0]],
    )
    assert np.array_equal(
        dataset["flux_loop.voltage.data"],
        [[np.nan], [1.0], [np.nan], [5.0], [np.nan]],
        equal_nan=True,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 5, 7, 10, 13, 20])
def test_xarray_batched(magnetics_metadata, batch_size):
    reader = StreamingXArrayConsumer(magnetics_metadata, batch_size=batch_size)

    def check_data(dataset, expected_time):
        assert np.array_equal(dataset.time, expected_time)
        assert np.array_equal(
            dataset["flux_loop.flux.data"],
            [[1.0], [2.0], [3.0], [4.0], [5.0]] + expected_time[None, :],
        )
        assert np.array_equal(
            dataset["flux_loop.voltage.data"],
            [[6.0]] + [[np.nan]] * 4 + expected_time[None, :],
            equal_nan=True,
        )

    # Pretend sending 20 messages
    for i in range(20):
        test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8") + i
        dataset = reader.process_message(test_data.tobytes())
        # Only expect a result after batch_size items are processed
        if dataset is None:
            assert (i + 1) % batch_size != 0
            continue
        expected_time = np.arange(batch_size, dtype=float) + (i + 1 - batch_size)
        check_data(dataset, expected_time)

    # Check that any remainders are handled
    msg_remaining = 20 % batch_size
    dataset = reader.finalize()
    if dataset is None:
        assert msg_remaining == 0
    else:
        assert len(dataset.time) == msg_remaining != 0
        expected_time = np.arange(msg_remaining, dtype=float) + 20 - msg_remaining
        check_data(dataset, expected_time)
