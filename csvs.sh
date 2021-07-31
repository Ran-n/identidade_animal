#! /bin/sh
#+ Autor:	Ran#
#+ Creado:	31/07/2021 10:12:27
#+ Editado:	31/07/2021 10:20:12

#[[ -n "${2+set}" ]] && ./mover_metricas.sh metricas

if [ -n "${2+set}" ]; then
    ./mover_metricas.sh metricas
fi
./ms2csv.py -d $1
