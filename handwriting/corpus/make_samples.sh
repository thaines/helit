#! /bin/bash

# Helper script to generate lots of samples - very slow, but I don;t care.

mkdir samples

for i in `seq -w 64`;
do
  for j in `seq -w 5`;
  do
    echo samples/subject${i}_pen${j}.txt
    ./make_sample.py > samples/subject${i}_pen${j}.txt
  done
done

