#!/bin/bash
for (( c=1; c<=12; c++ ))
do
	export THREADS=$c
	echo $c >> Out.txt
	for (( i=1; i<=10;i++ ))
	do
 		(time make runpar) 2>&1 | tee >> Out.txt
	done
done
