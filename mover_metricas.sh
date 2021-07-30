#! /bin/sh
#+ Autor:	Ran#
#+ Creado:	31/07/2021 00:15:56
#+ Editado:	31/07/2021 00:15:56

for ele in $(ls saidas/*/*/*.metricas); do cp $ele metricas; done
