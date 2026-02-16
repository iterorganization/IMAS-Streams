# IMAS Streams

Producers and consumers of streaming IMAS data.

## Development status

This project is in active development and may introduce breaking changes at any
moment.

## Installation

IMAS-Streams is not yet available on PyPI, but you can install it directly from
this git repository:

```bash
pip install 'imas-streams @ git+https://github.com/iterorganization/IMAS-Streams.git'
# If you wish to use the kafka features:
pip install 'imas-streams[kafka] @ git+https://github.com/iterorganization/IMAS-Streams.git'
```

## Command line interface

IMAS-Streams comes with a command line interface: the program `imas-streams`
should be available after installation. Note that some commands may require the
optional `kafka` dependency, see [Installation](#Installation).

```bash
imas-streams --version  # display version of installed imas-streams library
imas-streams --help     # display help message and list available commands
```

## Application programming interface (API)

API documentation is not yet available. The code is self-documented with
docstrings. All classes directly available in the
[`imas_streams`](src/imas_streams/__init__.py) and
[`imas_streams.kafka`](src/imas_streams/kafka.py) modules are considered public
API.

## Design goal and use cases

The goal of this project is to define a way to efficiently stream
IMAS-compatible data between producers and consumers, and develop the
corresponding applications and libraries to work with these IMAS data streams.

### Context

Before this project, IMAS-compatible data was typically transferred in one of
two ways:

1.  As a completely filled IDS. This IDS contains both static data and dynamic
    (time-dependent) data for all time samples of a simulation or experiment.
2.  As an IDS with a single time slice. This IDS contains both the static data
    and the dynamic data for a single point in time.

This project attempts to create an efficient protocol for streaming IMAS data,
where the static data is sent once, and only dynamic data is sent with each time
point. The main use cases are for applications where the overhead for sending
static data with each time slice is significant, for example when streaming live
experimental data.

### Limitations

For efficiency of the streaming protocol and its implementation, this project
sets the following constraints on the data that is streamed. Note that these
constraints may be lifted with future developments:

1.  All dynamic quantities must be known at the start of the stream.
2.  The dimension sizes
    ([shape](https://numpy.org/doc/stable/reference/generated/numpy.shape.html))
    of all dynamic quantities must be known at the start of the stream.
3.  Only quantities that the Data Dictionary labels as
    ["Dynamic"](https://imas-data-dictionary.readthedocs.io/en/latest/coordinates.html#static-constant-and-dynamic-nodes)
    can be dynamic data in the stream.
4.  Only [floating point
    quantities](https://imas-data-dictionary.readthedocs.io/en/latest/data_types.html#floating-point-data-types)
    can be dynamic data in the stream.
5.  All dynamic data must use the same time base.

## Legal

Copyright 2025 ITER Organization. The code in this repository is licensed under
[LGPL-3.0](LICENSE.txt).
