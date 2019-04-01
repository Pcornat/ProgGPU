#!/usr/bin/env bash
cmake --build build/ --target $1 -- -j
mv build/$1 prog