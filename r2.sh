#!/bin/sh

#+ Autor:	Ran#
#+ Creado:	02/04/2021 18:40:32
#+ Editado:	26/07/2021 11:36:32

REP=1

execucion() {
    echo ''
    echo '> Repetición #'$REP
    ./ratas.py -d $dataset -e $epoch -b $batch
    REP=$((REP+1))
}

for dataset in 128; do
     for epoch in 30 35 40 45 50; do
        for batch in 8 16 32 64 128 256; do
            execucion
            execucion
            execucion
        done
    done
done
