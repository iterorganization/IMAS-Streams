import logging
from collections.abc import Iterator

import libmuscle
from confluent_kafka import Producer
from libmuscle import Instance, Message
from packaging.version import Version
from ymmsl import Operator

from imas_streams import BatchedIDSConsumer, StreamingIDSConsumer
from imas_streams.kafka import (
    DEFAULT_KAFKA_CONSUMER_TIMEOUT,
    KafkaConsumer,
    KafkaSettings,
    create_kafka_topic,
)

logger = logging.getLogger(__name__)


def data_source():
    """MUSCLE3 data source streaming data from a single IMAS Stream on a Kafka topic."""
    if Version(libmuscle.__version__) < Version("0.9"):
        raise RuntimeError("This actor requires libmuscle version 0.9.0 or later")

    logger.info("Creating libmuscle instance")
    instance = Instance({Operator.O_I: ["ids_out"], Operator.S: ["trigger"]})

    while instance.reuse_instance():
        logger.info("Reading settings")
        kafka_host = instance.get_setting("kafka_host", "str")
        kafka_topic = instance.get_setting("kafka_topic", "str")
        batch_size = instance.get_setting("batch_size", "int", default=1)
        most_recent_only = instance.get_setting(
            "most_recent_only", "bool", default=False
        )
        if most_recent_only and batch_size != 1:
            raise RuntimeError(
                "'Most recent only' mode is incompatible with a batch size larger "
                "than 1. Please update the MUSCLE3 settings to set 'most_recent_only: "
                "false', or 'batch_size: 1'."
            )
        if most_recent_only and not instance.is_connected("trigger"):
            raise RuntimeError(
                "'Most recent only' mode requires that the 'trigger' port is connected."
            )

        logger.info("Creating kafka consumer")
        consumer = KafkaConsumer(
            KafkaSettings(host=kafka_host, topic_name=kafka_topic),
            BatchedIDSConsumer,
            # FIXME: depends on https://github.com/iterorganization/IMAS-Streams/pull/11
            # most_recent_only=most_recent_only,
            batch_size=batch_size,
        )

        logger.info("Start reading from the IMAS data stream")
        for ids in consumer.stream():
            if ids is None:
                continue  # Batch is not yet complete

            serialized = ids.serialize()
            time = ids.time[0]
            msg = Message(time, data=serialized)
            instance.send("ids_out", msg)

            # Optional: wait for trigger message
            if instance.is_connected("triggger"):
                instance.receive("trigger")
        logger.info("IMAS data stream ended")

    logger.info("Reuse loop finished")


def dynamic_data_source():
    """MUSCLE3 data source supporting streaming from multiple Kafka topics and
    publishing data back to Kafka.
    """
    # Check which version of M3 supports dynamic O_I and S ports
    if Version(libmuscle.__version__) <= Version("0.9.1"):
        # N.B. Develop branch with version 0.9.2.dev1 also works
        raise RuntimeError("This actor requires libmuscle version 0.10.0 or later")
    DynamicDataSource().run()


class DynamicDataSource:
    def __init__(self) -> None:
        logger.info("Creating libmuscle instance")
        # Don't specify ports to allow dynamic input/output ports
        self.instance = Instance()

        # Check the dynamic port configuration
        ports = self.instance.list_ports()
        for operator in [Operator.F_INIT, Operator.O_F]:
            if ports.get(operator):
                raise RuntimeError(
                    f"imas_streams does not support {operator.name} ports, but the "
                    f"following ports were defined: {', '.join(ports[operator])}"
                )
        self.output_ports = [
            port
            for port in ports.get(Operator.O_I, [])
            if self.instance.is_connected(port)
        ]
        self.input_ports = [
            port
            for port in ports.get(Operator.S, [])
            if self.instance.is_connected(port)
        ]
        if not self.output_ports or not self.input_ports:
            raise RuntimeError(
                "imas_streams needs at least one O_I port and one S port."
            )

        self.consumers: dict[str, KafkaConsumer] = {}
        """Kafka consumer per output port."""
        self.producer: Producer
        """Kafka producer."""

    def run(self) -> None:
        """Run MUSCLE3 reuse loop."""
        while self.instance.reuse_instance():
            logger.info("Reading settings")
            kafka_host = self.instance.get_setting("kafka_host", "str")
            kafka_topics = self.instance.get_setting("kafka_topics", "str")
            self.kafka_timeout = self.instance.get_setting(
                "kafka_timeout", "float", default=DEFAULT_KAFKA_CONSUMER_TIMEOUT
            )

            logger.info("Setting up Kafka Producer")
            self.producer = Producer({"bootstrap.servers": kafka_host})
            topic_per_port = self._parse_topics(kafka_topics)
            logger.info("Setting up Kafka Consumers for each stream")
            for port, topic in topic_per_port.items():
                if port in self.output_ports:
                    self.consumers[port] = KafkaConsumer(
                        KafkaSettings(host=kafka_host, topic_name=topic),
                        StreamingIDSConsumer,
                        return_copy=False,
                        timeout=self.kafka_timeout,
                    )
                else:
                    create_kafka_topic(KafkaSettings(host=kafka_host, topic_name=topic))

            for msgs in self.generate_serialized_idss():
                for port, (t, data) in msgs.items():
                    self.instance.send(port, Message(t, data=data))

                for port in self.input_ports:
                    msg = self.instance.receive(port)
                    # FIXME: This publishes serialized IDSs instead of streaming IMAS
                    # data. Our test case (EFIT++) doesn't produce data that adheres to
                    # the the IMAS-Streams assumptions (see README.md) so we cannot do
                    # better at the moment, unfortunately...
                    self.producer.produce(
                        topic=topic_per_port[port],
                        value=msg.data,
                    )
                    self.producer.poll(0)

            # Cleanup
            self.consumers = {}
            self.producer.flush()

    def _parse_topics(self, kafka_topics: str) -> dict[str, str]:
        """Parse kafka topics and return a dict {port_name: topic_name}."""
        topic_per_port: dict[str, str] = {}
        for line in kafka_topics.splitlines():
            if not line.strip():
                continue
            port, _, topic = map(str.strip, line.partition(":"))
            if not topic or not port:
                raise RuntimeError(
                    f"Invalid line encountered in 'kafka_topics' setting: '{line}'"
                )

            if port in self.output_ports or port in self.input_ports:
                topic_per_port[port] = topic
            else:
                logger.info(
                    "Ignoring kafka topic '%s' for disconnected port '%s'", topic, port
                )

        # Exception handling: each port needs to have a topic configured:
        if len(topic_per_port) != len(self.output_ports) + len(self.input_ports):
            missing_output = [p for p in self.output_ports if p not in topic_per_port]
            missing_input = [p for p in self.input_ports if p not in topic_per_port]
            missing_msgs = []
            if missing_output:
                missing_msgs.append(f"output ports: {', '.join(missing_output)}")
            if missing_input:
                missing_msgs.append(f"input ports: {', '.join(missing_input)}")
            missing_msg = " and ".join(missing_msgs)
            raise RuntimeError(
                f"Kafka topic is missing for {missing_msg}. Please add a line to the "
                "'kafka_topics' setting for each port."
            )

        return topic_per_port

    def generate_serialized_idss(self) -> Iterator[dict[str, tuple[float, bytes]]]:
        """Generate synchronized, serialized IDSs for the subscribed streams."""
        # Receive once on each stream:
        streams = {
            port: consumer.stream(timeout=self.kafka_timeout)
            for port, consumer in self.consumers.items()
        }
        idss = {port: next(stream) for port, stream in streams.items()}

        latest_starttime = max(ids.time[0] for ids in idss.values())
        main_port = next(iter(streams))
        main_ids = idss[main_port]
        main_stream = streams[main_port]

        # Skip ahead the main stream
        if main_ids.time[0] < latest_starttime:
            logger.info("Skipping messages until start time of latest stream")
            while main_ids.time[0] < latest_starttime:
                next(main_stream)

        # Generate time-synchronized serialized IDSs
        curdata: dict[str, tuple[float, bytes]] = {
            port: (ids.time[0], ids.serialize()) for port, ids in idss.items()
        }
        while True:
            # Get the last message <= main_ids.time[0] for each stream
            for port, ids in idss.items():
                if ids is main_ids:
                    continue
                while ids.time[0] <= main_ids.time[0]:
                    # Note: we may serialize too much here. For example, when the main
                    # stream produces data at 10 Hz, and a secondary stream at 20 Hz, we
                    # need to throw away every other serialized IDS. If this becomes a
                    # bottleneck we could optimize in two ways:
                    # 1. Stash the data at the streaming IMAS level instead of a
                    #    serialized IDS. Apply the buffer and serialize the IDS only
                    #    when needed.
                    # 2. Improve serialization and  directly copy the bytes from the
                    #    Streaming IDS data frame into the serialized IDS (assuming we
                    #    use the flexbuffers serialization protocol).
                    curdata[port] = (ids.time[0], ids.serialize())
                    try:
                        next(streams[port])
                    except StopIteration:
                        break
            yield curdata

            # Fetch the next message of the main stream
            try:
                next(main_stream)
            except StopIteration:
                return
            curdata[main_port] = (main_ids.time[0], main_ids.serialize())
