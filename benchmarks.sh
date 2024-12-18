#!/bin/bash

# compile CPU and GPU versions
nvcc slime_cpu.cu -o slime_cpu
nvcc slime_gpu_gl2.cu slime_kernels.cu -L libs -o slimeGL -lGL -lGLU -lglut

# clean output directories
rm -r ~/build/slime/out
mkdir -p ~/build/slime/out

# Define a list of tuples for (height, width) pairs
tuples="(75,100) (150,200) (300,400) (600,800) (900,1200)"

# Loop over each tuple
for tuple in $tuples; do
  # Remove parentheses
  clean_tuple=${tuple//[\(\)]/}
  
  # Split the tuple into parts
  IFS=',' read -r first second <<< "$clean_tuple"
  
  # Access individual elements and use for benchmark runs of CPU and GPU versions
  echo "--- Trial: Height=$first, Width=$second ---"
  srun --exclusive --ntasks 1 -G 1 -c 1 -- ./slime_cpu $first $second > ~/build/slime/out/cpu_$first.$second.out
  srun --exclusive --ntasks 1 -G 1 -c 1 --export ALL -- prime-run ./slimeGL 0 $first $second > ~/build/slime/out/gpu_$first.$second.out
done