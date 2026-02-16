import contextlib
import logging
import os

import imas
import pytest
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

pytest.importorskip("confluent_kafka")

import confluent_kafka.admin

from imas_streams import StreamingIDSConsumer, StreamingIDSProducer
from imas_streams.kafka import KafkaConsumer, KafkaProducer, KafkaSettings

logger = logging.getLogger(__name__)


@pytest.fixture
def kafka_host():
    value = os.getenv("KAFKA_HOST")
    if not value:
        pytest.fail("Cannot connect to Kafka server: KAFKA_HOST not set.")
    return value


@pytest.fixture
def test_magnetics():
    ids = imas.IDSFactory().magnetics()
    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [1.0]
    ids.flux_loop.resize(1)
    ids.flux_loop[0].name = "test"
    ids.flux_loop[0].flux.data = [0.0]
    return ids


@pytest.fixture(autouse=True)
def delete_test_topic(kafka_host):
    # Ensure that there is no existing 'test' topic in the Kafka server
    client = confluent_kafka.admin.AdminClient({"bootstrap.servers": kafka_host})
    fs = client.delete_topics(["test"])
    for _topic, future in fs.items():
        # Raises an exception when the topic did not exists or could not be deleted
        with contextlib.suppress(confluent_kafka.KafkaException):
            future.result()


def test_kafka_producer_consumer(kafka_host, test_magnetics):
    ids_producer = StreamingIDSProducer(test_magnetics)
    settings = KafkaSettings(host=kafka_host, topic_name="test")
    kafka_producer = KafkaProducer(settings, ids_producer.metadata)

    for i in range(5):
        test_magnetics.time[0] = i
        test_magnetics.flux_loop[0].flux.data[0] = 1 - i / 10

        message = ids_producer.create_message(test_magnetics)
        kafka_producer.produce(bytes(message))

    kafka_consumer = KafkaConsumer(settings, StreamingIDSConsumer)
    for i, ids in enumerate(kafka_consumer.stream(timeout=0.1)):
        assert ids.time[0] == i
        assert ids.flux_loop[0].name == "test"
        assert ids.flux_loop[0].flux.data[0] == 1 - i / 10
    assert i == 4  # We should have received 5 messages

    # Check that we can do this again, and pass extra argument to StreamingIDSConsumer
    kafka_consumer = KafkaConsumer(settings, StreamingIDSConsumer, return_copy=False)
    for i, ids in enumerate(kafka_consumer.stream(timeout=0.1)):
        assert ids.time[0] == i
        assert ids.flux_loop[0].name == "test"
        assert ids.flux_loop[0].flux.data[0] == 1 - i / 10
    assert i == 4  # We should have received 5 messages


def test_kafka_producer_topic_exists(kafka_host, test_magnetics):
    ids_producer = StreamingIDSProducer(test_magnetics)
    settings = KafkaSettings(host=kafka_host, topic_name="test")
    KafkaProducer(settings, ids_producer.metadata)  # This will create the topic

    with pytest.raises(confluent_kafka.KafkaException, match="TOPIC_ALREADY_EXISTS"):
        # Expect an exception, the topic 'test' already exists:
        KafkaProducer(settings, ids_producer.metadata)
