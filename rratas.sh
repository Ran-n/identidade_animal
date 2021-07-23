#!/bin/sh

#+ Autor:	Ran#
#+ Creado:	02/04/2021 18:40:32
#+ Editado:	14/07/2021 00:18:58

REP=1

for dataset in 32 64 128; do
#    for epoch in $(seq 5 5 100); do
     for epoch in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30; do
        for batch in 8 16 32 64 128 256; do
            echo ''
            echo '> Repeción #'$REP
            ./ratas.py -d $dataset -e $epoch -b $batch
            REP=$((REP+1))
        done
    done
done
