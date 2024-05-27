#!/bin/bash

program=/home/twang/sources/projects/QuEST/build/demo
profiler="nvidia-smi --query-gpu=memory.used --format=csv -i 0 -lms 5"
log_file="profileResult.log"

$profiler > $log_file &
profiler_pid=$!

$program

kill $profiler_pid
