#!/bin/bash

while read p; do
  echo $p
  wget -P $2 $p 1>> stdout.txt 2>> stderr.out
  done <$1

