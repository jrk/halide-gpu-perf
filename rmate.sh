#!/bin/sh
files=$@

for f in $files; do
    rmate $f
done
