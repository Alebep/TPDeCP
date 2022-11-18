#!/bin/bash
for (( c=1; c<=40; c= c + 4 ))
do
	export THREADS=$c
	echo $c >> OutR.txt
	for (( i=1; i<=5;i++ ))
	do
 		(time make runpar) 2>&1 | tee >> OutR.txt
	done
done
