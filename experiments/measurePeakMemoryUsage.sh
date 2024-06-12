#!/bin/bash

program="/home/twang/sources/projects/QuEST/build/demo 32"
profiler="nvidia-smi --query-gpu=memory.used --format=csv -i 1 -lms 5"
log_file="profileResult.log"

$profiler > $log_file &
profiler_pid=$!

$program

kill $profiler_pid
