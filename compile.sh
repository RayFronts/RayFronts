#!/bin/bash
# Builds the Rayfronts C++ extension
# E.g usage: CMAKE_INSTALL_PREFIX=$CONDA_PREFIX ./compile.sh
# This script is intended to be run from the root of the repository.

cmake -S rayfronts/csrc/ -B rayfronts/csrc/build
cmake --build rayfronts/csrc/build/
