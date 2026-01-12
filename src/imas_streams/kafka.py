import base64
import logging
import random
import time
import uuid
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict

from imas_streams.abc import StreamConsumer
from imas_streams.metadata import StreamingIMASMetadata

logger = logging.getLogger(__name__)

# Check if optional kafka dependency is available
try:
    import confluent_kafka
    from confluent_kafka.admin import AdminClient, NewTopic
except ImportError:
    logger.error("Optional dependency 'confluent_kafka' is not installed.")
    raise


DEFAULT_KAFKA_CONSUMER_TIMEOUT = 60  # seconds
_STREAMING_HEADER_KEY = "streaming-imas-metadata"


class KafkaSettings(BaseModel):
    """Dynamic data specifier."""

    host: str
    """The kafka instance ip address (bootstrap servers)."""
    topic_name: str
    """Name of the topic to stream data to."""

    model_config = ConfigDict(extra="forbid")


def _create_kafka_topic(settings: KafkaSettings):
    """Create a new kafka topic."""
    conf = {"bootstrap.servers": settings.host}
    admin_client = AdminClient(conf)

    # Create a NewTopic object to define the topic's configuration.
    new_topic = NewTopic(
        topic=settings.topic_name,
        num_partitions=1,  # More partitions are not useful
        replication_factor=1,
    )

    # The create_topics function is asynchronous. It returns a dictionary
    # of futures, where each key is the topic name and the value is a future
    # that completes when the topic creation is done.
    fs = admin_client.create_topics([new_topic])

    # Wait for the topic creation operation to complete.
    # The result() call will raise an exception if the topic creation failed.
    for _topic, future in fs.items():
        # This will raise an exception when the topic exists or could not be created
        future.result()


class KafkaProducer:
    """Producer of streaming IMAS data to a Kafka topic."""

    def __init__(
        self, settings: KafkaSettings, metadata: StreamingIMASMetadata
    ) -> None:
        self._settings = settings
        self._metadata = metadata
        conf = {"bootstrap.servers": settings.host}
        self._producer = confluent_kafka.Producer(conf)

        # Use a fixed message key to ensure ordering of the messages by Kafka
        random_id = random.SystemRandom().randbytes(6)
        self._message_key = f"IMAS-Kafka-{base64.b64encode(random_id).decode()}"

        # Create the topic and send the metadata as first message
        _create_kafka_topic(settings)
        self._producer.produce(
            topic=self._settings.topic_name,
            value=self._metadata.model_dump_json().encode(),
            key=self._message_key,
            headers={_STREAMING_HEADER_KEY: metadata.metadata_version},
        )
        self._expected_message_size = self._metadata.nbytes

    def produce(self, message: bytes) -> None:
        """Produce a time frame to the configured Kafka topic."""
        if len(message) != self._expected_message_size:
            raise ValueError(
                f"Unexpected message size of {len(message)} bytes. "
                "Was expecting {self._expected_message_size} bytes."
            )

        self._producer.produce(
            topic=self._settings.topic_name,
            value=message,
            key=self._message_key,
        )


class KafkaConsumer:
    """Consumer of streaming IMAS data from a Kafka topic."""

    def __init__(
        self,
        settings: KafkaSettings,
        stream_consumer_cls: type[StreamConsumer],
        **stream_consumer_kwargs,
    ) -> None:  # TODO: type
        self._settings = settings
        conf = {
            "bootstrap.servers": settings.host,
            "auto.offset.reset": "earliest",
            "group.id": str(uuid.uuid4()),
        }
        self._consumer = confluent_kafka.Consumer(conf)

        # Subscribe to the topic, retry until successful
        self._metadata = self._subscribe()
        self._stream_consumer = stream_consumer_cls(
            self._metadata, **stream_consumer_kwargs
        )

    def _subscribe(
        self, timeout=DEFAULT_KAFKA_CONSUMER_TIMEOUT
    ) -> StreamingIMASMetadata:
        # Wait until the requested topic is available
        start_time = time.monotonic()
        topic_name = self._settings.topic_name
        while True:
            metadata = self._consumer.list_topics(topic=topic_name, timeout=1)
            topic_metadata = metadata.topics.get(topic_name)
            if topic_metadata is not None and topic_metadata.error is None:
                break
            if (start_time + timeout) <= time.monotonic():
                raise RuntimeError(
                    f"Timeout while waiting for kafka topic '{topic_name}'"
                )

        # The topic is available, subscribe and receive metadata
        self._consumer.subscribe([topic_name])
        msg = self._consumer.poll(timeout)
        if msg is None:
            raise RuntimeError("Timeout while waiting for streaming metadata.")
        if msg.error() is not None:
            raise msg.error()
        headers = dict(msg.headers())
        if _STREAMING_HEADER_KEY not in headers:
            raise RuntimeError(
                f"Topic '{topic_name}' does not contain IMAS streaming metadata."
            )
        return StreamingIMASMetadata.model_validate_json(msg.value())

    def stream(self, *, timeout=DEFAULT_KAFKA_CONSUMER_TIMEOUT) -> Iterator[Any]:
        """Keep receiving and yielding messages on the Kafka topic until no new message
        is produced within specified timeout.

        Args:
            timeout: Timeout in seconds. The stream is expected to be closed when no new
                messages arrive within this time period after the last received message.

        Yields:
            Binary messages.
        """
        try:
            while True:
                msg = self._consumer.poll(timeout)
                if msg is None:
                    logger.info(
                        "Nothing received on topic '%s' for %f seconds: stream closed.",
                        self._settings.topic_name,
                        timeout,
                    )
                    break
                if msg.error():
                    raise msg.error()

                yield self._stream_consumer.process_message(msg.value())
        finally:
            self._consumer.commit()
            self._consumer.close()
