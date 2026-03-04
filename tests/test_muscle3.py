import ymmsl
from ymmsl.v0_2 import Reference, resolve

ymmsl_config = """
ymmsl_version: v0.2

description: Test muscle3 datasource actor

imports:
- from imas_streams.data_source import implementation imas_streams_source

models:
    test:
        description: Simple test model
        components:
            source:
                implementation: imas_streams_source
                description: Data source
                ports:
                    o_i: ids_out
                    s: trigger

resources:
    source:
        threads: 1
"""


def test_load_ymmsl_config():
    config = ymmsl.load(ymmsl_config)
    resolve(Reference([]), config)
    config.check_consistent()
