#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Move into build, run CMake with compile_commands.json generation, then build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake --build .

# Go back and run the executable
cd ..
./build/babblec
