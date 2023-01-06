#!/bin/bash
for (( c=1; c<=65; c++ ))
do
	export THREADS=$c
	echo $c >> OutR.txt
	for (( i=1; i<=15;i++ ))
	do
 		(time srun --partition=cpar --cpus-per-task=40 make runpar) 2>&1 | tee >> OutR.txt
	done
done
