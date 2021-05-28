#!/usr/bin/python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	01/05/2021 20:35:20
#+ Editado:	16/05/2021 17:21:16
#-----------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy import stats
#-----------------------------------------------------------------
def pie_probas_datasets(nome, colorscheme):
    labels = ['Dataset 32x32', 'Dataset 64x64', 'Dataset 128x128']

    plt.title('Distribución das probas')

    serie = data.groupby(['train dataset'])['epoch'].count()
    serie.name = ''
    # cantidade de probas
    #print(serie.sum())

    serie.plot.pie(labels=labels, colors=colorscheme, autopct='%1.1f%%', shadow=True, legend=False)

    plt.savefig(nome+'.svg')
    plt.savefig(nome+'.png')
    plt.close()
#-----------------------------------------------------------------
def lineplot_valores_epoch_batch(nome, valor, tipo, colorscheme):
	"""
	nome > Nome da imaxe
	valor > accuracy, precision, recall, f, fpr
	tipo > epoch, batch
	colorscheme > cores a usar
	"""
	if tipo == 'epoch':
		if valor == 'f' or valor == 'fpr':
			plt.title('{} segundo o número de épocas'.format(valor.upper()))
		else:
			plt.title('$\\it{}$ segundo o número de épocas'.format(valor.capitalize()))
		plt.xlabel('Época')

	elif tipo == 'batch':
		tipo = 'batch size'
		if valor == 'f' or valor == 'fpr':
			plt.title('{} segundo o tamaño dos lotes'.format(valor.upper()))
		else:
			plt.title('$\\it{}$ segundo o tamaño dos lotes'.format(valor.capitalize()))
		plt.xlabel('Tamaño de lote')
	else:
		sys.exit("Error")

	data32.groupby([tipo])[valor].mean().plot(x='', y='', label='Dataset 32x32', color= colorscheme[0])
	data64.groupby([tipo])[valor].mean().plot(x='', y='', label='Dataset 64x64', color= colorscheme[1])
	data128.groupby([tipo])[valor].mean().plot(x='', y='', label='Dataset 128x128', color= colorscheme[2])

	if valor == 'f' or valor == 'fpr':
		plt.ylabel('{}'.format(valor.upper()))
	else:
		plt.ylabel('$\\it{}$'.format(valor.capitalize()))
	
	if tipo == 'epoch':
		plt.xlabel('Época')
		plt.legend(loc='upper right')
		if valor == 'f' or valor == 'fpr':
			plt.legend(loc='lower right')
	elif tipo == 'batch size':
		plt.xlabel('Tamaño de lote')
		plt.legend(loc='lower right')
		if valor == 'fpr':
			plt.legend(loc='upper right')
	else:
		sys.exit("Error")

	plt.savefig(nome+'.svg')
	plt.savefig(nome+'.png')
	plt.close()
#-----------------------------------------------------------------
def barplot_valores_epoch_batch(nome, valor, colorscheme, prentrenadas = True, group_minhas = False):
	
	if group_minhas:
		index = ['32, 64 & 128', 'inceptionv3', 'resnet50', 'vgg16']
		acc = data[valor].mean()
		acc_mode = data[valor].mode()[0]
		prentrenadas = True
	else:
		index = ['32', '64', '128', 'inceptionv3', 'resnet50', 'vgg16']
		acc32 = data32[valor].mean()
		acc32_mode = data32[valor].mode()[0]
		acc64 = data64[valor].mean()
		acc64_mode = data64[valor].mode()[0]
		acc128 = data128[valor].mean()
		acc128_mode = data128[valor].mode()[0]
		
	if prentrenadas:
		acc_inceptionv3 = data_inceptionv3[valor].mean()
		acc_inceptionv3_mode = data_inceptionv3[valor].mode()[0]
		acc_resnet50 = data_resnet50[valor].mean()
		acc_resnet50_mode = data_resnet50[valor].mode()[0]
		acc_vgg16 = data_vgg16[valor].mean()
		acc_vgg16_mode = data_vgg16[valor].mode()[0]

		if group_minhas:
			colorscheme = [colorblind_6_amalgama_3_primeiros, colorscheme[-3], colorscheme[-2], colorscheme[-1]]
			df = pd.DataFrame({'media':[acc, acc_inceptionv3, acc_resnet50, acc_vgg16],
				'moda':[acc_mode, acc_inceptionv3_mode, acc_resnet50_mode, acc_vgg16_mode]}, index = index)
			ax = df['media'].plot.bar(color=colorscheme)
			df['moda'].plot(color=cor_linha_moda)
		else:
			df = pd.DataFrame({'media':[acc32, acc64, acc128, acc_inceptionv3, acc_resnet50, acc_vgg16],
				'moda':[acc32_mode, acc64_mode, acc128_mode, acc_inceptionv3_mode, acc_resnet50_mode, acc_vgg16_mode]}, index = index)
			ax = df['media'].plot.bar(color=colorscheme)
			df['moda'].plot(color=cor_linha_moda)

	else:
		index = index[:3]
		df = pd.DataFrame({'media':[acc32, acc64, acc128],
			'moda':[acc32_mode, acc64_mode, acc128_mode]}, index = index)
		ax = df['media'].plot.bar(color=colorscheme)
		df['moda'].plot(color=cor_linha_moda)

	if valor == 'f' or valor == 'fpr':
		plt.title('{} media nas distintas redes'.format(valor.upper()))
		plt.ylabel('{}'.format(valor.upper()))
	else:
		plt.title('$\\it{}$ media nas distintas redes'.format(valor.capitalize()))
		plt.ylabel('$\\it{}$'.format(valor.capitalize()))
	
	plt.xlabel('')
	plt.xticks(rotation='horizontal')

	for bar in ax.patches:
		ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()-0.005), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')

	plt.savefig(nome+'.svg')
	plt.savefig(nome+'.png')
	plt.close()
#-----------------------------------------------------------------
def lineplot_valores_epoch_batch_32(nome, valor, tipo, colorscheme):
	"""
	nome > Nome da imaxe
	valor > accuracy, precision, recall, f, fpr
	tipo > epoch, batch
	colorscheme > cores a usar
	"""
	if tipo == 'epoch':
		if valor == 'f' or valor == 'fpr':
			plt.title('{} segundo o número de épocas'.format(valor.upper()))
		else:
			plt.title('$\\it{}$ segundo o número de épocas'.format(valor.capitalize()))
		plt.xlabel('Época')

	elif tipo == 'batch':
		tipo = 'batch size'
		if valor == 'f' or valor == 'fpr':
			plt.title('{} segundo o tamaño dos lotes'.format(valor.upper()))
		else:
			plt.title('$\\it{}$ segundo o tamaño dos lotes'.format(valor.capitalize()))
		plt.xlabel('Tamaño de lote')
	else:
		sys.exit("Error")

	data32_32.groupby([tipo])[valor].mean().plot(x='', y='', label='Imaxes validación 32x32', color= colorscheme[0])
	data32_64.groupby([tipo])[valor].mean().plot(x='', y='', label='Imaxes validación 64x64', color= colorscheme[1])
	data32_128.groupby([tipo])[valor].mean().plot(x='', y='', label='Imaxes validación 128x128', color= colorscheme[2])

	if valor == 'f' or valor == 'fpr':
		plt.ylabel('{}'.format(valor.upper()))
	else:
		plt.ylabel('$\\it{}$'.format(valor.capitalize()))
	
	if tipo == 'epoch':
		plt.xlabel('Época')
		plt.legend(title='$\\it{Dataset}$ de entrenamento 32x32', loc='upper right')
		if valor == 'fpr':
			plt.legend(title='$\\it{Dataset}$ de entrenamento 32x32', loc='lower right')
		elif valor == 'recall':
			plt.legend(title='$\\it{Dataset}$ de entrenamento 32x32', loc='lower left')
	elif tipo == 'batch size':
		plt.xlabel('Tamaño de lote')
		plt.legend(title='$\\it{Dataset}$ de entrenamento 32x32', loc='lower right')
		if valor == 'fpr':
			plt.legend(title='$\\it{Dataset}$ de entrenamento 32x32', loc='upper right')
	else:
		sys.exit("Error")

	plt.savefig(nome+'.svg')
	plt.savefig(nome+'.png')
	plt.close()
#-----------------------------------------------------------------
def heatmap(nome, datos, tit, xtit=None, ytit=None):
	plt.title(tit)
	sns.heatmap(data=datos, annot=True)
	
	if xtit:
		plt.xlabel(xtit)
	if ytit:
		plt.ylabel(ytit)

	plt.xticks(rotation='horizontal')

	plt.savefig(nome+".svg")
	plt.savefig(nome+".png")
	plt.close()
#-----------------------------------------------------------------
def scatterplot(nome, datos, tit, x, y, tit_lenda=None, cor=None, estilo=None, tamanho=None, palette='deep'):
    plt.title(tit)

    sns.scatterplot(data=datos, x=x, y=y, hue=cor, style=estilo, size=tamanho, palette=palette)

    if tit_lenda:
        plt.legend(title=tit_lenda)

    plt.savefig(nome+".svg")
    plt.savefig(nome+".png")
    plt.close()
#-----------------------------------------------------------------
def boxplot(nome, datos, tit, x=None, y=None, swarm=False, cor='.25', palette=None):
    plt.title(tit)

    sns.boxplot(x=x, y=y, data=datos, palette=palette)

    if swarm:
        sns.swarmplot(x=x, y=y, data=datos, color=cor)

    plt.savefig(nome+".svg")
    plt.savefig(nome+".png")
    plt.close()
#-----------------------------------------------------------------
#-----------------------------------------------------------------
def fora_raros(df, columnas):
    for columna in columnas:
        df[columna] = df[columna][(pd.DataFrame(np.abs(stats.zscore(df[columna])))<2).all(axis=1)]

    return df
#-----------------------------------------------------------------
inicial = ['royalblue', 'forestgreen', 'gold']

colorblind_9 = ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#ffffe0', '#ffbcaf', '#f4777f', '#cf3759', '#93003a']
colorblind_6 = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a']
colorblind_6_amalgama_3_primeiros = '#5a81b9'
colorblind_3 = ['#00429d', '#ffffe0', '#93003a']

lineplot = ['#0000ff', '#008000', '#93003a']
cor_linha_moda = '#959084'
lineplot6 = ['#0000ff', '#6919e8', '#8d2ed1', '#ffa59e', '#dd4c65', '#93003a']

data_sc = pd.read_csv('resumo.csv')
data2 = pd.read_csv('resumo_total.csv')

columnas = ['accuracy', 'precision', 'recall', 'f', 'fpr']
data = fora_raros(data_sc.copy(), columnas)

data32con500  = data[data['train dataset'] == 32]
data32con500_sc  = data_sc[data_sc['train dataset'] == 32]
data32  = data32con500[data32con500['epoch'] < 500]
data32_sc  = data32con500_sc[data32con500_sc['epoch'] < 500]
data64  = data[data['train dataset'] == 64]
data64_sc  = data_sc[data_sc['train dataset'] == 64]
data128 = data[data['train dataset'] == 128]
data128_sc = data_sc[data_sc['train dataset'] == 128]

data_inceptionv3 = data2[data2['train dataset'] == 'inceptionv3']
data_resnet50 = data2[data2['train dataset'] == 'resnet50']
data_vgg16 = data2[data2['train dataset'] == 'vgg16']

data32_32 = data32[data32['valid dataset'] == 32]
data32_32_sc = data32_sc[data32_sc['valid dataset'] == 32]
data32_64 = data32[data32['valid dataset'] == 64]
data32_64_sc = data32_sc[data32_sc['valid dataset'] == 64]
data32_128 = data32[data32['valid dataset'] == 128]
data32_128_sc = data32_sc[data32_sc['valid dataset'] == 128]

data64_32 = data64[data64['valid dataset'] == 32]
data64_32_sc = data64_sc[data64_sc['valid dataset'] == 32]
data64_64 = data64[data64['valid dataset'] == 64]
data64_64_sc = data64_sc[data64_sc['valid dataset'] == 64]
data64_128 = data64[data64['valid dataset'] == 128]
data64_128_sc = data64_sc[data64_sc['valid dataset'] == 128]

data128_32 = data128[data128['valid dataset'] == 32]
data128_32_sc = data128_sc[data128_sc['valid dataset'] == 32]
data128_64 = data128[data128['valid dataset'] == 64]
data128_64_sc = data128_sc[data128_sc['valid dataset'] == 64]
data128_128 = data128[data128['valid dataset'] == 128]
data128_128_sc = data128_sc[data128_sc['valid dataset'] == 128]

#-----------------------------------------------------------------
""
pie_probas_datasets('graficas/000-distribucion', colorblind_6)
""
lineplot_valores_epoch_batch('graficas/001-accuracy_epocas', 'accuracy', 'epoch', lineplot)
lineplot_valores_epoch_batch('graficas/002-accuracy_lotes', 'accuracy', 'batch', lineplot)
lineplot_valores_epoch_batch('graficas/003-precision_epocas', 'precision', 'epoch', lineplot)
lineplot_valores_epoch_batch('graficas/004-precision_lotes', 'precision', 'batch', lineplot)
lineplot_valores_epoch_batch('graficas/005-recall_epocas', 'recall', 'epoch', lineplot)
lineplot_valores_epoch_batch('graficas/006-recall_lotes', 'recall', 'batch', lineplot)
lineplot_valores_epoch_batch('graficas/007-f_epocas', 'f', 'epoch', lineplot)
lineplot_valores_epoch_batch('graficas/008-f_lotes', 'f', 'batch', lineplot)
lineplot_valores_epoch_batch('graficas/009-fpr_epocas', 'fpr', 'epoch', lineplot)
lineplot_valores_epoch_batch('graficas/010-fpr_lotes', 'fpr', 'batch', lineplot)
""
barplot_valores_epoch_batch('graficas/011-acc-barplot', 'accuracy', colorblind_6)
barplot_valores_epoch_batch('graficas/012-acc-barplot_nativas', 'accuracy', colorblind_6, prentrenadas = False)
barplot_valores_epoch_batch('graficas/013-acc-barplot_amalgama_minhas', 'accuracy', colorblind_6, group_minhas = True)
barplot_valores_epoch_batch('graficas/014-precision-barplot', 'precision', colorblind_6)
barplot_valores_epoch_batch('graficas/015-precision-barplot_nativas', 'precision', colorblind_6, prentrenadas = False)
barplot_valores_epoch_batch('graficas/016-precision-barplot_amalgama_minhas', 'precision', colorblind_6, group_minhas = True)
barplot_valores_epoch_batch('graficas/017-recall-barplot', 'recall', colorblind_6)
barplot_valores_epoch_batch('graficas/018-recall-barplot_nativas', 'recall', colorblind_6, prentrenadas = False)
barplot_valores_epoch_batch('graficas/019-recall-barplot_amalgama_minhas', 'recall', colorblind_6, group_minhas = True)
barplot_valores_epoch_batch('graficas/020-f-barplot', 'f', colorblind_6)
barplot_valores_epoch_batch('graficas/021-f-barplot_nativas', 'f', colorblind_6, prentrenadas = False)
barplot_valores_epoch_batch('graficas/022-f-barplot_amalgama_minhas', 'f', colorblind_6, group_minhas = True)
barplot_valores_epoch_batch('graficas/023-fpr-barplot', 'fpr', colorblind_6)
barplot_valores_epoch_batch('graficas/024-fpr-barplot_nativas', 'fpr', colorblind_6, prentrenadas = False)
barplot_valores_epoch_batch('graficas/025-fpr-barplot_amalgama_minhas', 'fpr', colorblind_6, group_minhas = True)
""
lineplot_valores_epoch_batch_32('graficas/026-accuracy32_epocas', 'accuracy', 'epoch', lineplot)
lineplot_valores_epoch_batch_32('graficas/027-accuracy32_lotes', 'accuracy', 'batch', lineplot)
lineplot_valores_epoch_batch_32('graficas/028-precision32_epocas', 'precision', 'epoch', lineplot)
lineplot_valores_epoch_batch_32('graficas/029-precision32_lotes', 'precision', 'batch', lineplot)
lineplot_valores_epoch_batch_32('graficas/030-recall32_epocas', 'recall', 'epoch', lineplot)
lineplot_valores_epoch_batch_32('graficas/031-recall32_lotes', 'recall', 'batch', lineplot)
lineplot_valores_epoch_batch_32('graficas/032-f32_epocas', 'f', 'epoch', lineplot)
lineplot_valores_epoch_batch_32('graficas/033-f32_lotes', 'f', 'batch', lineplot)
lineplot_valores_epoch_batch_32('graficas/034-fpr32_epocas', 'fpr', 'epoch', lineplot)
lineplot_valores_epoch_batch_32('graficas/035-fpr32_lotes', 'fpr', 'batch', lineplot)
""
df1 = pd.DataFrame(data.groupby(['train dataset']).mean())[['accuracy', 'precision', 'recall', 'f', 'fpr']]
df1_1 = pd.DataFrame(data.groupby(['train dataset']).mean())[['accuracy']]
df1_2 = pd.DataFrame(data.groupby(['train dataset']).mean())[['precision']]
df1_3 = pd.DataFrame(data.groupby(['train dataset']).mean())[['recall']]
df1_4 = pd.DataFrame(data.groupby(['train dataset']).mean())[['f']]
df1_5 = pd.DataFrame(data.groupby(['train dataset']).mean())[['fpr']]

df2 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['accuracy', 'precision', 'recall', 'f', 'fpr']]
df2_1 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['accuracy']]
df2_2 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['precision']]
df2_3 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['recall']]
df2_4 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['f']]
df2_5 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean())[['fpr']]

df3 = pd.DataFrame(data.groupby(['train dataset']).mean()[['tp', 'fn', 'tn', 'fp']])
df4 = pd.DataFrame(data.groupby(['train dataset', 'valid dataset']).mean()[['tp', 'fn', 'tn', 'fp']])

heatmap('graficas/036-heatmap_global_global', df1, 'Mapa de calor global', xtit='Métricas')
heatmap('graficas/037-heatmap_global_acc', df1_1, 'Mapa de calor global de $\\it{}$'.format('accuracy'), xtit='Métricas')
heatmap('graficas/038-heatmap_global_precision', df1_2, 'Mapa de calor global de $\\it{}$'.format('precision'), xtit='Métricas')
heatmap('graficas/039-heatmap_global_recall', df1_3, 'Mapa de calor global de $\\it{}$'.format('recall'), xtit='Métricas')
heatmap('graficas/040-heatmap_global_f', df1_4, 'Mapa de calor global de F', xtit='Métricas')
heatmap('graficas/041-heatmap_global_fpr', df1_5, 'Mapa de calor global de FPR', xtit='Métricas')

heatmap('graficas/042-heatmap_granulado_global', df2, 'Mapa de calor granulado', xtit='Métricas')
heatmap('graficas/043-heatmap_granulado_acc', df2_1, 'Mapa de calor granulado de $\\it{}$'.format('accuracy'), xtit='Métricas')
heatmap('graficas/044-heatmap_granulado_precision', df2_2, 'Mapa de calor granulado de $\\it{}$'.format('precision'), xtit='Métricas')
heatmap('graficas/045-heatmap_granulado_recall', df2_3, 'Mapa de calor granulado de $\\it{}$'.format('recall'), xtit='Métricas')
heatmap('graficas/046-heatmap_granulado_f', df2_4, 'Mapa de calor granulado de F', xtit='Métricas')
heatmap('graficas/047-heatmap_granulado_fpr', df2_5, 'Mapa de calor granulado de FPR', xtit='Métricas')

heatmap('graficas/048-heatmap_confusion_matrix_global', df3, 'Mapa de calor global elementos matriz de confusión', xtit='Métricas')
heatmap('graficas/049-heatmap_confuxion_matrix_granulado', df4, 'Mapa de calor granulado elementos matriz de confusión', xtit='Métricas')
""
scatterplot('graficas/050-scatter_total', data, 'Éxito segundo o $\\it{dataset}$ de entrenamento', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ entrenamento', cor='train dataset', palette=lineplot)
scatterplot('graficas/051-scatter_total', data_sc, 'Éxito segundo o $\\it{dataset}$ de entrenamento (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ entrenamento', cor='train dataset', palette=lineplot)

scatterplot('graficas/052-scatter_32', data32, 'Éxito no $\\it{dataset}$ entrenamento 32', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=lineplot)
scatterplot('graficas/053-scatter_32', data32_sc, 'Éxito no $\\it{dataset}$ entrenamento 32 (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=lineplot)
scatterplot('graficas/054-scatter_64', data64, 'Éxito no $\\it{dataset}$ entrenamento 64', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=lineplot)
scatterplot('graficas/055-scatter_64', data64_sc, 'Éxito no $\\it{dataset}$ entrenamento 64 (valores atípicos)', 'accuracy','f', tit_lenda='$\\it{dataset}$ validación',  cor='valid dataset', palette=lineplot)
scatterplot('graficas/056-scatter_128', data128, 'Éxito no $\\it{dataset}$ entrenamento 128', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=lineplot)
scatterplot('graficas/057-scatter_128', data128_sc, 'Éxito no $\\it{dataset}$ entrenamento 128 (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=lineplot)

scatterplot('graficas/058-scatter_32_32', data32_32, 'Éxito no $\\it{dataset}$ 32 imaxes 32x32', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=[colorblind_6[0]])
scatterplot('graficas/059-scatter_32_32', data32_32_sc, 'Éxito no $\\it{dataset}$ 32 imaxes 32x32 (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=[colorblind_6[0]])
scatterplot('graficas/060-scatter_32_64', data32_64, 'Éxito no $\\it{dataset}$ 32 imaxes 64x64', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=[colorblind_6[1]])
scatterplot('graficas/061-scatter_32_64', data32_64_sc, 'Éxito no $\\it{dataset}$ 32 imaxes 64x64 (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=[colorblind_6[1]])
scatterplot('graficas/062-scatter_32_128', data32_128, 'Éxito no $\\it{dataset}$ 32 imaxes 128x128', 'accuracy', 'f', tit_lenda='$\\it{dataset}$ validación', cor='valid dataset', palette=[colorblind_6[2]])
scatterplot('graficas/063-scatter_32_128', data32_128_sc, 'Éxito no $\\it{dataset}$ 32 imaxes 128x128 (valores atípicos)', 'accuracy', 'f', tit_lenda='$\\it{Dataset}$ validación', cor='valid dataset', palette=[colorblind_6[2]])
""
boxplot('graficas/064-boxplot-acc_segundo_dataset_entrenamento', data, '$\\it{Accuracy}$ por $\it{dataset}$ de entrenamento', x='train dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/065-boxplot-acc_segundo_dataset_validacion', data, '$\\it{Accuracy}$  por $\it{dataset}$ de validación', x='valid dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/066-boxplot-f_segundo_dataset_entrenamento', data, 'f por $\it{dataset}$ de entrenamento', x='train dataset', y='f', palette=colorblind_6)
boxplot('graficas/067-boxplot-f_segundo_dataset_validacion', data, 'f por $\it{dataset}$ de validación', x='valid dataset', y='f', palette=colorblind_6)
boxplot('graficas/068-boxplot-acc_segundo_dataset_entrenamento-sc', data_sc, '$\\it{Accuracy}$  por $\it{dataset}$ de entrenamento (valores atípicos)', x='train dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/069-boxplot-acc_segundo_dataset_validacion-sc', data_sc, '$\\it{Accuracy}$  por $\it{dataset}$ de validación (valores atípicos)', x='valid dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/070-boxplot-f_segundo_dataset_entrenamento-sc', data_sc, 'f por $\it{dataset}$ de entrenamento (valores atípicos)', x='train dataset', y='f', palette=colorblind_6)
boxplot('graficas/071-boxplot-f_segundo_dataset_validacion-sc', data_sc, 'f por $\it{dataset}$ de validación (valores atípicos)', x='valid dataset', y='f', palette=colorblind_6)

boxplot('graficas/072-boxplot-acc_dataset32_segundo_valid_dataset', data32, '$\\it{Accuracy}$ do $\it{dataset}$ 32', x='valid dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/073-boxplot-f_dataset32_segundo_valid_dataset', data32, 'f do $\it{dataset}$ 32', x='valid dataset', y='f', palette=colorblind_6)
boxplot('graficas/074-boxplot-acc_dataset32_segundo_valid_dataset-sc', data32_sc, '$\\it{Accuracy}$ do $\it{dataset}$ 32 (valores atípicos)', x='valid dataset', y='accuracy', palette=colorblind_6)
boxplot('graficas/075-boxplot-f_dataset32_segundo_valid_dataset-sc', data32_sc, 'f do $\it{dataset}$ 32 (valores atípicos)', x='valid dataset', y='f', palette=colorblind_6)


for ele in [fich for fich in os.listdir('graficas') if fich.endswith('.svg')]:
    os.rename(os.getcwd()+'/graficas/'+ele,os.getcwd()+'/graficas/svg/'+ele)
#-----------------------------------------------------------------
