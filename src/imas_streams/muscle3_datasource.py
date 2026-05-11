import logging

import libmuscle
from libmuscle import Instance, Message
from ymmsl import Operator

from imas_streams import BatchedIDSConsumer
from imas_streams.kafka import KafkaConsumer, KafkaSettings

logger = logging.getLogger(__name__)


def data_source():
    """MUSCLE3 data source streaming data from a single IMAS Stream on a Kafka topic."""
    if tuple(map(int, libmuscle.__version__.split(".")[:2])) < (0, 9):
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
    # FIXME: Check which version of M3 supports dynamic O_I and S ports
    # See PR: https://github.com/multiscale/muscle3/pull/350
    if tuple(map(int, libmuscle.__version__.split(".")[:2])) < (0, 10):
        raise RuntimeError("This actor requires libmuscle version 0.10.0 or later")

    logger.info("Creating libmuscle instance")
    instance = Instance()  # Don't specify ports to allow dynamic input/output ports

    # Check the dynamic port configuration
    ports = instance.list_ports()
    for operator in [Operator.F_INIT, Operator.O_F]:
        if ports.get(operator):
            raise RuntimeError(
                f"imas_streams does not support {operator.name} ports, but the "
                f"following ports were defined: {', '.join(ports[Operator.F_INIT])}"
            )
    output_ports = [port for port in ports[Operator.O_I] if instance.is_connected(port)]
    input_ports = [port for port in ports[Operator.S] if instance.is_connected(port)]
    if not output_ports or not input_ports:
        raise RuntimeError("imas_streams needs at least one O_I port and one S port.")

    while instance.reuse_instance():
        logger.info("Reading settings")
        kafka_host = instance.get_setting("kafka_host", "str")
        kafka_topics = instance.get_setting("kafka_topics", "str")

        topic_per_port = _parse_topics(kafka_topics, output_ports, input_ports)
        output_port_topics = {
            port: topic
            for port, topic in topic_per_port.items()
            if port in output_ports
        }

        # TODO:
        # 1 Create kafka clients for each topic we want to receive data for
        # 2 Create kafka producer for each topic we need to send data on
        # 3 Synchronize messages from different streams
        # 4 Read message in streams, instance.send() on the respective ports
        # 5 instance.receive() on input ports, create streaming IMAS data frame and send
        #   to Kafka
        # 6 Repeat 4-6 until stream has ended


def _parse_topics(
    kafka_topics: str, output_ports: list[str], input_ports: list[str]
) -> dict[str, str]:
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

        if port in output_ports or port in input_ports:
            topic_per_port[port] = topic
        else:
            logger.info(
                "Ignoring kafka topic '%s' for disconnected port '%s'", topic, port
            )

    if len(topic_per_port) != len(output_ports) + len(input_ports):
        missing_output = [port for port in output_ports if port not in topic_per_port]
        missing_input = [port for port in input_ports if port not in topic_per_port]
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
