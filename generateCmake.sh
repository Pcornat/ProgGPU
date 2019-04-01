#!/usr/bin/env bash

cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -G "Unix Makefiles" ../
cd ../