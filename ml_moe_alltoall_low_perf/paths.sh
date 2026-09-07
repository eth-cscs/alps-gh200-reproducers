#!/bin/bash
# =============================================================================
# common/paths.sh  —  ENVIRONMENT-SPECIFIC PATHS
#
# Edit this file once per user/cluster. Every knob uses `:=` so an experiment
# script can still override any path by assigning it before sourcing train.sh.
#
# Sourced by common/train.sh (and submit.sh) before any other defaults.
# =============================================================================

# -- Scratch / base directories --
: "${SCRATCH_DIR:=/ritom/scratch/cscs/$USER}"

# -- Project root (this repo) --
# SLURM copies the submitted script to its spool dir, so $0 is unreliable.
# Launch scripts use this to source train.sh with an absolute path.
: "${SCRIPTS_ROOT:=/capstor/scratch/cscs/gfu/frameworks/myscripts}"

# -- Codebase --
: "${MEGATRON_LM_DIR:=/capstor/scratch/cscs/gfu/frameworks/Megatron-LM}"

# -- Container / environment image --
: "${IMAGE_ENV:=./alps-pytorch2602.toml}"

# -- Output / logging --
: "${SLURM_LOG_DIR:=$SCRATCH_DIR/slurmlogs}"

# -- Triton kernel cache --
# Keep this on a persistent path: it is the cache that stops Triton from
# recompiling every kernel at every launch. train.sh puts one shared cache dir
# underneath, used by all ranks. (Inductor and cpp_extension caches are NOT here
# — train.sh keeps those on node-local /tmp, per job.)
#: "${JIT_CACHE_BASE:=/iopsstor/scratch/cscs/gfu/.cache}"

## -- UCCL (MoE flex/DeepEP dispatcher; built once from source, then reused) --
#: "${UCCL_SOURCE_DIR:=/capstor/scratch/cscs/gfu/frameworks/uccl-sai}"  # uccl checkout (override to your own clone)
#: "${UCCL_INSTALL_BASE:=$SCRATCH_DIR/uccl_python}"       # shared install lands in $UCCL_INSTALL_BASE/default
#: "${UCCL_CONTAINER_ENV_FILE:=$IMAGE_ENV}"               # build container (keep == training image for ABI match)
