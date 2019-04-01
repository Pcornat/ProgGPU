#!/usr/bin/env bash
cmake --build build/ --target $1 -- -j
mv cmake-build-release/$1 prog