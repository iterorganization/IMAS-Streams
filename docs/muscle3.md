# IMAS-Streams MUSCLE3 integration

IMAS-Streams contains two MUSCLE3 components:
1. `imas_streams_source` is a simple component that reads data from a single
   IMAS data Stream in a Kafka topic, and makes that data available as
   serialized IDSs to the MUSCLE3 simulation. Multiple time slices can be
   batched in a single IDS.
2. `imas_streams` is a more complex component that reads data from multiple IMAS
   data Streams on a Kafka cluster. It synchronizes the data when these streams
   do not use the same time bases. It also publishes data back to one or more
   Kafka topics.

We provide more details on these component in the following sections.


## `imas_streams_source` component

<!--
N.B. this is copied from muscle3_config.py, ideally we'll use the sphinx-ymmsl
plugin to keep a single source of truth.
-->

Data source reading Streaming IMAS data from a Kafka topic and making it
available in a MUSCLE3 simulation.

The `ids_out` port sends one message for every `batch_size` time slices
streamed over the configured kafka topic. The type of IDS depends on the
configured kafka topic: please take care that this matches the IDS that is
expected for components receiving the message.

You may use the `trigger` port to indicate that the previous message is
processed and a new message may be sent. If this port is not connected then
this component will send messages on the `ids_out` port as soon as they are
available.


### Example configuration

```yaml
ymmsl_version: v0.2
description: Example usage for imas_streams_source component

imports:
- from imas_streams.data_source import implementation imas_streams_source

models:
  example:
    decription: Simple example model
    components:
      source:
        implementation: imas_streams_source
        description: Data source
        ports:
          o_i: ids_out
          s: trigger
      physics_component:
        implementation: my_physics_program
        description: Physics simulation
        ports:
          f_init: equilibrium_in
          o_f: trigger
    conduits:
      source.ids_out: physics_component.equilibrium_in
      physics_component.trigger: source.trigger

settings:
  kafka_host: localhost:9092
  kafka_topic: test.equilibrium

programs:
  my_physics_program:
    executable: /path/to/my/physics/program
```


## `imas_streams` component

<!--
N.B. this is copied from muscle3_config.py, ideally we'll use the sphinx-ymmsl
plugin to keep a single source of truth.
-->

This is a data source that reads Streaming IMAS data from multiple Kafka
topics and makes the data available in a MUSCLE3 simulation.

To use this component in your simulation, you need to configure the
following:

1. Define output ports for each IMAS Stream for the O_I operator.
2. Specify the Kafka host to connect to in the `kafka_host` setting.
3. Specify the Kafka topic names for each output port in the `kafka_topics`
    setting as `<output_port>: <kafka_topic>`.

### Stream synchronization

This component will synchronize messages from different streams if they don't
have the same time base:

1. All streams must have data available before a message is sent to the MUSCLE3
   workflow.
2. Messages will be sent at the same frequency as the first stream (as
   configured in the `kafka_topics` setting).
3. Messages for the other stream use the following interpolation strategy:

   If there is a data point at exactly the same moment as the first stream,
   then that data is sent.
   Otherwise, the data at the latest time before that time is sent.

For example, if there are three streams with data at the following time points:

- Stream A: data at t = [0, 1, 2, 3]
- Stream B: data at t = [1, 3]
- Stream C: data at t = [0, 1.5, 3, 4.5]

If stream A is the first configured stream, then this component will send three
messages:

1. The first message at t=1: this is the first moment that stream B has data
   for. The data for stream A at t=0 is discarded. Since stream C doesn't have
   data at t=1, the data at t=0 is sent instead.
2. The second message will be sent at t=2, which repeats the data for Stream B
   at t=1, and uses the data at t=1.5 for Stream C.
3. The last message is sent at t=3, for which there is data on all three
   streams.

### Example configuration

```yaml
ymmsl_version: v0.2
description: Example usage for imas_streams component
imports:
- from imas_streams.data_source import implementation imas_streams
models:
  example:
    components:
      streams:
        description: IMAS Streams data source and sink
        implementation: imas_streams
        ports:
          o_i: magnetics_out pf_active_out
          s: equilibrium_in
      equilibrium:
        description: Equilibrium reconstruction code
        implementation: my_equilibrium_program
        ports:
          f_init: magnetics_in pf_active_in
          o_f: equilibrium_out
    conduits:
      streams.magnetics_out: equilibrium.magnetics_in
      streams.pf_active_out: equilibrium.pf_active_in
      equilibrium.equilibrium_out: streams.equilibrium_in
settings:
  streams.kafka_host: localhost:9092
  streams.kafka_topics: |
    magnetics_out: kafka.topic.for.magnetics
    pf_active_out: kafka.topic.for.pf_active
    equilibrium_in: kafka.topic.for.equilibrium
programs:
  my_equilibrium_program:
    executable: /path/to/my/equilibrium/program
```

In this example, IMAS Streams from two topics are made available:

1. The data on the topic `kafka.topic.for.magnetics` is sent on the output
    port `magnetics_out`.
2. The data on the topic `kafka.topic.for.pf_active` is sent on the output
    port `pf_active_out`.

The data received on the S port `equilibrium_in` is published to
`kafka.topic.for.equilibrium`.
