#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	07/06/2021 18:45:18
#+ Editado:	29/07/2021 21:02:16
#-----------------------------------------------------------------
import statistics
import os
import pandas as pd
import numpy as np
from scipy import stats
import json
import sys

from uteis.imprimir import jprint
#-----------------------------------------------------------------

def mensaxe_axuda():
	print('--Axuda--------------------------------------------------------')
	print('-a/-h/?\t-> Esta mensaxe de axuda.')
	print()
	print('-r\t-> Imprimir táboa de ratas.')
	print('-rtl\t-> Imprimir táboa de transfer learning de ratas.')
	print('-c\t-> Imprimir táboa de cascudas.')
	print('-ctl\t-> Imprimir táboa de tansfer learning de cascudas.')
	print('---------------------------------------------------------------')

def main(csv_ratas, csv_cascudas):
	# collemos os argumentos de entrada
	args = sys.argv[1:]

	# lemos as entradas por liña de comandos
	# se pide axuda mostramola e saimos
	if any(['-a' in args, '-h' in args, '?' in args, len(args) == 0, all(['-r' not in args, '-rtl' not in args, '-c' not in args,'-ctl' not in args])]):
		mensaxe_axuda()
		sys.exit()

	taboa_ratas_tl = {
					'inceptionv3': 	{'32': None, '64': None, '128': None, '150': None}, 
					'resnet50': 	{'32': None, '64': None, '128': None, '224': None}, 
					'vgg16': 		{'32': None, '64': None, '128': None, '224': None}
					}

	taboa_cascudas_tl = {
						'inceptionv3':	{'32': None, '64': None, '128': None, '150': None, '256': None}, 
						'resnet50': 	{'32': None, '64': None, '128': None, '224': None, '256': None}, 
						'vgg16': 		{'32': None, '64': None, '128': None, '224': None, '256': None}
						}

	taboa_ratas = {'32': None, '64': None, '128': None}
	taboa_cascudas = {'32': None, '64': None, '128': None, '256': None}

	ratas_total = pd.read_csv(csv_ratas)
	cascudas_total = pd.read_csv(csv_cascudas)

	ratas = ratas_total[ratas_total['rede'] == 'pecusCNN']
	ratas_tl = ratas_total[ratas_total['rede'] != 'pecusCNN']
	cascudas = cascudas_total[cascudas_total['rede'] == 'pecusCNN']
	cascudas_tl = cascudas_total[cascudas_total['rede'] != 'pecusCNN']


	# fora valores extraños
	metricas_ratas = ['accuracy', 'precision', 'recall', 'f', 'fpr']
	metricas_cascudas = ['macro accuracy', 'macro precision', 'macro recall', 'macro f', 'macro fpr', 'micro accuracy', 'micro precision', 'micro recall', 'micro f', 'micro fpr']

	# media e desviación tipica poblacional das métricas
	for tipo in ['inceptionv3', 'resnet50', 'vgg16']:
		taboa_ratas_tl[tipo] = {'mu': dict(ratas_tl[ratas_tl['rede'] == tipo][metricas_ratas].mean()),
								'sigma': dict(ratas_tl[ratas_tl['rede'] == tipo][metricas_ratas].var(ddof=0))}

	for tipo in ['inceptionv3', 'resnet50', 'vgg16']:
		taboa_cascudas_tl[tipo] = {'mu': dict(cascudas_tl[cascudas_tl['rede'] == tipo][metricas_cascudas].mean()),
								  	'sigma': dict(cascudas_tl[cascudas_tl['rede'] == tipo][metricas_cascudas].var(ddof=0))}


	for tipo in ['32', '64', '128']:
		taboa_ratas[tipo] = {'mu': dict(ratas[ratas['train dataset'] == int(tipo)][metricas_ratas].mean()),
							'sigma': dict(ratas[ratas['train dataset'] == int(tipo)][metricas_ratas].var(ddof=0))}

	for tipo in ['32', '64', '128', '256']:
		taboa_cascudas[tipo] = {'mu': dict(cascudas[cascudas['train dataset'] == int(tipo)][metricas_cascudas].mean()),
								'sigma': dict(cascudas[cascudas['train dataset'] == int(tipo)][metricas_cascudas].var(ddof=0))}

	for ele in args:
		if ele == '-r':
			print('Táboa ratas pecusCNN:')
			jprint(taboa_ratas)
			print()

		if ele == '-rtl':	
			print('Táboa ratas transfer learning:')
			jprint(taboa_ratas_tl)
			print()

		if ele == '-c':
			print('Táboa cascudas pecusCNN:')
			jprint(taboa_cascudas)
			print()

		if ele == '-ctl':
			print('Táboa cascudas transfer learning:')
			jprint(taboa_cascudas_tl)
			print()

#-----------------------------------------------------------------

if __name__ == '__main__':
	main(csv_ratas='saida_ratas.csv', csv_cascudas='saida_cascudas.csv')

#-----------------------------------------------------------------