#!/bin/bash

# generates a flamegraph for rank0
# the svg file, e.g. flamegraph.daint.svg, can be viewed using a web browser
# requires FlameGraph has been cloned:
# git clone https://github.com/brendangregg/FlameGraph.git

if [ $SLURM_PROCID -eq 0 ]; then
    perf record -g --call-graph fp -o perf.$CLUSTER_NAME.data ./pdsygst
    perf script -i perf.$CLUSTER_NAME.data > perf.$CLUSTER_NAME.script
    ./FlameGraph/stackcollapse-perf.pl perf.$CLUSTER_NAME.script > perf.$CLUSTER_NAME.folded
    ./FlameGraph/flamegraph.pl ./perf.$CLUSTER_NAME.folded > flamegraph.$CLUSTER_NAME.svg
else
    ./pdsygst
fi

