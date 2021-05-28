#!/bin/sh

#+ Autor:	Ran#
#+ Creado:	02/04/2021 18:40:32
#+ Editado:	04/04/2021 21:49:36

REP=1

for dataset in 32 64 128; do
    for epoch in $(seq 5 5 100); do
        for batch in 8 16 32 64 128 256; do
            echo "> Repeción #"$REP
#           ./ratas.py -d $dataset -e $epoch -b $batch -iter $REP
            ./ratas.py -d $dataset -e $epoch -b $batch
            REP=$((REP+1))
        done
    done
done
