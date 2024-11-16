#!/bin/bash

make -C ~/build/lab-5
rm -r ~/build/lab-5/out
mkdir ~/build/lab-5/out

for n in hereford_256 hereford_512 bansberia sombrero_galaxy hereford whirlpool
do
    echo "--- Benchmark: N = $n ---"
    for trial in 1 2 3
    do
        srun --exclusive --ntasks 1 --mem=16G -c 1 -G 1 -- ~/build/lab-5/lab5 /data/cs4981_gpu_programming/images/$n.ppm ~/source/lab-5/imgs/cpu_blur_$n.ppm ~/source/lab-5/imgs/gpu_blur_$n.ppm blur > ~/build/lab-5/out/$n.$trial.blur.out
        srun --exclusive --ntasks 1 --mem=16G -c 1 -G 1 -- ~/build/lab-5/lab5 /data/cs4981_gpu_programming/images/$n.ppm ~/source/lab-5/imgs/cpu_edge_$n.ppm ~/source/lab-5/imgs/gpu_edge_$n.ppm edge > ~/build/lab-5/out/$n.$trial.edge.out
        echo "Trial $trial Completed"
    done
done