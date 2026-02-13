import base64
import logging
import random
import time
import uuid
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict

from imas_streams.metadata import StreamingIMASMetadata
from imas_streams.protocols import StreamConsumer

logger = logging.getLogger(__name__)

# Check if optional kafka dependency is available
try:
    import confluent_kafka
    from confluent_kafka.admin import AdminClient, NewTopic
except ImportError:
    logger.error("Optional dependency 'confluent_kafka' is not installed.")
    raise


DEFAULT_KAFKA_CONSUMER_TIMEOUT = 60  # seconds
_INITIAL_BACKOFF_TIME = 0.02  # seconds
_MAXIMUM_BACKOFF_TIME = 1.0  # seconds
_STREAMING_HEADER_KEY = "streaming-imas-metadata"


class KafkaSettings(BaseModel):
    """Settings for the Kafka Consumer and Producer."""

    host: str
    """The kafka instance ip address (bootstrap servers)."""
    topic_name: str
    """Name of the topic to stream data to."""

    model_config = ConfigDict(extra="forbid")


def _create_kafka_topic(settings: KafkaSettings):
    """Create a new kafka topic.

    This will raise an exception when the topic already exists, or if the topic could
    not be created (potentially due to missing permissions).
    """
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
        """Create a Kafka Consumer.

        This will connect to the Kafka cluster and create a topic to send data to. The
        streaming IMAS metadata is sent as first message on the topic.

        N.B. An exception is raised when the topic already exists on the Kafka cluster.
        """
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
        self._producer.poll(0)
        self._expected_message_size = self._metadata.nbytes

    def __del__(self):
        """Cleanup Kafka Producer resources"""
        # Ensure all messages are sent
        self._producer.flush()

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
        self._producer.poll(0)


class KafkaConsumer:
    """Consumer of streaming IMAS data from a Kafka topic.

    Example:
        The following example creates a Kafka consumer that attempts to stream data from
        the ``test`` topic on a kafka instance running on ``localhost:9092``. If the
        ``test`` topic doesn't exist within 5 seconds, an exception is raised. This
        consumer will create IDSs (with :py:class:`~imas_streams.StreamingIDSConsumer`).
        The additional keyword argument (``return_copy=True``) is forwarded to the
        ``StreamingIDSConsumer``.

        The stream is assumed to end once 10 seconds have passed without new messages.

        .. code-block:: python

            consumer = KafkaConsumer(
                KafkaSettings(host="localhost:9092", topic="test"),
                StreamingIDSConsumer,
                timeout=5,
                return_copy=True,
            )
            for ids in consumer.stream(timeout=10):
                do_something_with_data(ids)
    """

    def __init__(
        self,
        settings: KafkaSettings,
        stream_consumer_cls: type[StreamConsumer],
        *,
        timeout=DEFAULT_KAFKA_CONSUMER_TIMEOUT,
        **stream_consumer_kwargs,
    ) -> None:
        """Create a new KafkaConsumer.

        N.B. This will block until the requested topic is created on the Kafka cluster,
        or fail after ``timeout`` seconds.

        Args:
            settings: Kafka host and topic to connect to.
            stream_consumer_cls: StreamConsumer type used for processing the received
                messages.
            timeout: Maximum time (in seconds) to wait for the topic.
        """
        self._settings = settings
        conf = {
            "bootstrap.servers": settings.host,
            "auto.offset.reset": "earliest",
            "group.id": str(uuid.uuid4()),
        }
        self._consumer = confluent_kafka.Consumer(conf)

        # Subscribe to the topic, retry until successful
        self._metadata = self._subscribe(timeout)
        self._stream_consumer = stream_consumer_cls(
            self._metadata, **stream_consumer_kwargs
        )

    def _subscribe(self, timeout) -> StreamingIMASMetadata:
        """Subscribe to the requested Kafka topic and receive streaming IMAS metadata.

        Args:
            timeout: Maximum time (in seconds) to wait for the topic to be created on
                the Kafka cluster.

        Returns:
            Streaming IMAS metadata for the subscribed topic.
        """
        # Wait until the requested topic is available
        start_time = time.monotonic()
        topic_name = self._settings.topic_name
        logger.info("Subscribing to topic '%s' ...", topic_name)
        backoff = _INITIAL_BACKOFF_TIME
        while True:
            metadata = self._consumer.list_topics(topic=topic_name, timeout=1)
            topic_metadata = metadata.topics.get(topic_name)
            if topic_metadata is not None and topic_metadata.error is None:
                break
            if (start_time + timeout) <= time.monotonic():
                raise RuntimeError(
                    f"Timeout reached while waiting for kafka topic '{topic_name}'"
                )
            # Don't busy-loop when topic does not exist yet
            logger.info(
                "Topic '%s' does not exist yet, waiting %gs before retry",
                topic_name,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAXIMUM_BACKOFF_TIME)

        # The topic is available, subscribe and receive metadata
        self._consumer.subscribe([topic_name])
        logger.info("Subscribed to topic '%s'", topic_name)
        msg = self._consumer.poll(timeout)
        if msg is None:
            raise RuntimeError("Timeout reached while waiting for streaming metadata.")
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
            Data produced by the StreamConsumer, e.g. an IDS for the
            StreamingIDSConsumer.

            For batching consumers (such as the BatchingIDSConsumer) the last yielded
            value may contain fewer time slices than the batch size.
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
                    # Yield any remaining data
                    result = self._stream_consumer.finalize()
                    if result is not None:
                        yield result
                    break
                if msg.error():
                    raise msg.error()

                yield self._stream_consumer.process_message(msg.value())
        finally:
            self._consumer.commit()
            self._consumer.close()
