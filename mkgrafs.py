#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	09/06/2021 20:08:14
#+ Editado:	31/07/2021 13:20:22
#-----------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import os

#-----------------------------------------------------------------

def heatmap(nome, datos, tit, xtit=None, ytit=None, xticklabels=None, extensions=['.svg', '.png', '.pdf']):
	plt.title(tit)
	
	if xticklabels:
		sns.heatmap(data=datos, annot=True, xticklabels=xticklabels)
	else:
		sns.heatmap(data=datos, annot=True)

	if xtit:
		plt.xlabel(xtit)

	if ytit:
		plt.ylabel(ytit)

	plt.xticks(rotation='horizontal')

	for extension in extensions:
		plt.savefig(nome+extension)
	plt.close()

def lineplot(nome, datos, metrica, tipo, colorscheme, extensions=['.svg', '.png', '.pdf']):
	"""
	nome 		-> Nome da imaxe
	datos 		-> dataframe completo
	metrica 	-> accuracy, precision, recall, f, fpr, macro accuracy, micro accuracy
	tipo 		-> epoch, batch
	colorscheme -> cores a usar
	"""

	if tipo == 'epoch':
		if metrica == 'macro accuracy':
			plt.title('$\\it{}$ das cascudas segundo o número de épocas'.format('Macro-Accuracy'))
		elif metrica == 'micro accuracy':
			plt.title('$\\it{}$ das cascudas segundo o número de épocas'.format('Micro-Accuracy'))
		elif metrica == 'f' or metrica == 'fpr':
			plt.title('{} segundo o número de épocas'.format(metrica.upper()))
		else:
			plt.title('$\\it{}$ das ratas segundo o número de épocas'.format(metrica.capitalize()))
		plt.xlabel('$\\it{}$'.format('Epoch'))

	elif tipo == 'batch':
		tipo = 'batch size'
		if metrica == 'macro accuracy':
			plt.title('$\\it{}$ das cascudas segundo o tamaño dos lotes'.format('Macro-Accuracy'))
		elif metrica == 'micro accuracy':
			plt.title('$\\it{}$ das cascudas segundo o tamaño dos lotes'.format('Micro-Accuracy'))
		elif metrica == 'f' or metrica == 'fpr':
			plt.title('{} segundo o tamaño dos lotes'.format(metrica.upper()))
		else:
			plt.title('$\\it{}$ das ratas segundo o tamaño dos lotes'.format(metrica.capitalize()))
		plt.xlabel('$\\it{}$ $\\it{}$'.format('Batch', 'Size'))
	else:
		sys.exit("Error")

	td_cols = datos['train dataset'].unique()
	reps = 0
	for train_dataset in td_cols:
		datos[datos['train dataset'] == td_cols[reps]].groupby([tipo])[metrica].mean().plot(
				x='', y='', label='Dataset '+str(td_cols[reps])+'x'+str(td_cols[reps]), color= colorscheme[reps]
			)
		reps += 1

	plt.grid(linestyle='dotted')
	
	if metrica == 'macro accuracy':
		plt.ylabel('$\\it{}$'.format('Macro-Accuracy'))
	elif metrica == 'micro accuracy':
		plt.ylabel('$\\it{}$'.format('Micro-Accuracy'))
	elif metrica == 'f' or metrica == 'fpr':
		plt.ylabel('{}'.format(metrica.upper()))
	else:
		plt.ylabel('$\\it{}$'.format(metrica.capitalize()))
	

	if tipo == 'epoch':
		plt.xlabel('$\\it{}$'.format('Epoch'))
		plt.legend(loc='lower right')
		
		eixo_x = [1,3,5,7,9,12,15,20,25,30,35,40,45,50]
		if list(td_cols) == [32, 64, 128]:
			#plt.xticks([1,3,5,7,9,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
			plt.xticks(eixo_x)
			plt.xlim([0.1,eixo_x[-1]+0.2])
		elif list(td_cols) == [32, 64, 128, 256]:
			#plt.xticks([1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60])
			plt.xticks(eixo_x)
			plt.xlim([0.7,eixo_x[-1]+0.2])

	elif tipo == 'batch size':
		if list(td_cols) == [32, 64, 128]:
			plt.legend(loc='upper right')
			plt.xticks([8,16,32,64,128,256])
		elif list(td_cols) == [32, 64, 128, 256]:
			plt.legend(loc='lower left')
			plt.xticks([8,16,32,64,128,256])
		plt.xlim([7,256.5])

		plt.xlabel('$\\it{}$ $\\it{}$'.format('Batch', 'Size'))

	for extension in extensions:
		plt.savefig(nome+extension)
	plt.close()

#-----------------------------------------------------------------

def main(carpeta, paleta, csv_ratas, csv_cascudas):
	ratas_total = pd.read_csv(csv_ratas)
	cascudas_total = pd.read_csv(csv_cascudas)

	ratas = ratas_total[ratas_total['rede'] == 'pecusCNN']
	ratas_tl = ratas_total[ratas_total['rede'] != 'pecusCNN']
	cascudas = cascudas_total[cascudas_total['rede'] == 'pecusCNN']
	cascudas_tl = cascudas_total[cascudas_total['rede'] != 'pecusCNN']

	ratas_500 = ratas[ratas['epoch'] < 500]
	cascudas_100 = cascudas[cascudas['epoch'] < 100]

	ratas_30 = ratas[ratas['epoch'] < 31]

	metricas_ratas = ['accuracy', 'precision', 'recall', 'f', 'fpr']
	metricas_cascudas = ['macro accuracy', 'macro precision', 'macro recall', 'macro f', 'macro fpr', 
						'micro accuracy', 'micro precision', 'micro recall', 'micro f', 'micro fpr']
	metricas_cascudas_macro = ['macro accuracy', 'macro precision', 'macro recall', 'macro f', 'macro fpr']
	metricas_cascudas_micro = ['micro accuracy', 'micro precision', 'micro recall', 'micro f', 'micro fpr']

	# 'Mapa de calor das métricas segundo o dataset entrenamento das ratas'
	heatmap(carpeta+'00-heatmap_ratas_train', pd.DataFrame(ratas.groupby(['train dataset']).mean())[metricas_ratas],
			'Mapa de calor xeral das métricas do dataset das ratas', xtit='Métricas')

	# 'Mapa de calor das métricas segundo o dataset entrenamento e validación das ratas'
	heatmap(carpeta+'01-heatmap_ratas_train_valid', pd.DataFrame(ratas.groupby(['train dataset', 'valid dataset']).mean())[metricas_ratas],
			'Mapa de calor granulado das métricas do dataset das ratas', xtit='Métricas')

	# 'Mapa de calor das métricas segundo o dataset entrenamento das cascudas'
	heatmap(carpeta+'02-heatmap_cascudas_train', pd.DataFrame(cascudas.groupby(['train dataset']).mean())[metricas_cascudas_macro],
			'Mapa de calor xeral das métricas do dataset das cascudas', xtit='Métricas', xticklabels=metricas_ratas)

	# 'Mapa de calor das métricas segundo o dataset entrenamento e validación das cascudas'
	heatmap(carpeta+'03-heatmap_cascudas_train_valid', pd.DataFrame(cascudas.groupby(['train dataset', 'valid dataset']).mean())[metricas_cascudas_macro],
			'Mapa de calor granulado das métricas do dataset das cascudas', xtit='Métricas', xticklabels=metricas_ratas)

	lineplot(carpeta+'04-lineplot_acc_epochs-ratas', datos=ratas_500, metrica='accuracy', tipo='epoch', colorscheme=paleta)
	#lineplot(carpeta+'04-lineplot_acc_epochs-ratas2', datos=ratas_30, metrica='accuracy', tipo='epoch', colorscheme=paleta)
	lineplot(carpeta+'05-lineplot_acc_batch-ratas', datos=ratas, metrica='accuracy', tipo='batch', colorscheme=paleta)

	lineplot(carpeta+'06-lineplot_acc_epochs-cascudas', datos=cascudas_100, metrica='macro accuracy', tipo='epoch', colorscheme=paleta)
	lineplot(carpeta+'07-lineplot_acc_batch-cascudas', datos=cascudas, metrica='macro accuracy', tipo='batch', colorscheme=paleta)

#-----------------------------------------------------------------

if __name__ == '__main__':
	carpeta = 'graficas/'
	if not os.path.exists(carpeta): os.mkdir(carpeta)

	paleta_cb8 = ['#00429d', '#316292', '#3c8385', '#36a476', '#fab5a0', '#fe6843', '#dc0000', '#8f0000']
	paleta_cb4 = ['#00429d', '#3c8385', '#fe6843', '#8f0000']
	paleta_cb4_2 = ['#00429d', '#36a476', '#fab5a0', '#dc0000']

	csv_ratas = 'saida_ratas.csv'
	csv_cascudas = 'saida_cascudas.csv'

	main(carpeta, paleta_cb4_2, csv_ratas, csv_cascudas)

#-----------------------------------------------------------------