#!/bin/sh
fswatch --event Updated -o -l 3 test.cu | xargs -n1 -I{} make run
