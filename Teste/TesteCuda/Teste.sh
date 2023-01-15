#!/bin/bash
for (( c=1; c<=65; c++ ))
do
	export THREADS=$c
 	sbatch run.sh
done
