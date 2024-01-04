#!/bin/bash

while true
do
    cd /workspace/DigitalCurling3-Server/build
    ./digitalcurling3_server &

    cd /dc3/CurlingLightZero/shell
    python3 team1.py
done
