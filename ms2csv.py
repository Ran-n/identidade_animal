#!/usr/bin/python3
#-----------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	01/05/2021 16:30:52
#+ Creado:	05/06/2021 18:51:59
#+ Editado:	29/07/2021 14:35:30
#-----------------------------------------------------------
import os
import sys
import csv
import json

from uteis import ficheiro#, imprimir
#-----------------------------------------------------------

def gardar_csv(nome, contido):
	with open(nome, 'w') as csvfile:
		for row in contido:
			csvfile.write(','.join(row))
			csvfile.write('\n')

def get_info_filename(nome_fich):
	saida = []
	nome_fich, identidade = nome_fich.split('___')
	for ele in nome_fich.split(';'):
		try:
			saida.append(ele.split('_')[0].split('-')[2])
		except:
			pass
		saida.append(ele.split('_')[1])
	saida.append(identidade.split('.')[0])
	return saida

def main():
	# collemos os argumentos de entrada
	args = sys.argv[1:]

	# lemos as entradas por liña de comandos
	# se pide axuda mostramola e saimos
	if any(['-a' in args, '-h' in args, '?' in args, len(args)<2]):
		print('--Axuda--------------------------------------------------------')
		print('-a/-h/?\t--> Esta mensaxe de axuda')
		print('-d\t--> Path do directorio cos ficheiros\t\t[.]')
		print('-o\t--> Nome do ficheiro de saída (sen .csv)\t[saida]')
		print('-O\t--> Igual ca "-o" pero dando o full path\t[saida]')
		print('---------------------------------------------------------------')
		sys.exit()

	# get do path inicial
	path_inicial = os.getcwd()
	# path da carpeta cos ficheiros
	if '-d' in args:
		os.chdir(args[args.index('-d')+1])

	# nome do ficheiro de saida
	if '-o' in args:
		fich_saida = path_inicial+'/'+args[args.index('-o')+1]
	elif '-O' in args:
		fich_saida = args[args.index('-O')+1]
	else:
		fich_saida = 'saida'

	# collemos os nomes de todolos ficheiros
	ficheiros_ratas = []
	ficheiros_cascudas = []
	for fich in os.listdir():
		if fich.startswith('dataset-ratas'):
			ficheiros_ratas.append(fich)
		elif fich.startswith('dataset-cascudas'):
			ficheiros_cascudas.append(fich)
	# e ordeámolos
	ficheiros_ratas.sort()
	ficheiros_cascudas.sort()

	cabeceira = {
		'ratas': [
				'rede', 'train dataset', 'epoch', 'batch size', 'semente', 'id', 'valid dataset',
				'tp', 'fn', 'tn', 'fp',
				'accuracy', 'precision', 'recall', 'f', 'fpr'
				],
		'cascudas':[
				'rede', 'train dataset', 'epoch', 'batch size', 'semente', 'id', 'valid dataset', 
				'f1c1', 'f1c2', 'f1c3', 'f2c1', 'f2c2', 'f2c3', 'f3c1', 'f3c2', 'f3c3',
				'macro accuracy', 'macro precision', 'macro recall', 'macro f', 'macro fpr',
				'micro accuracy', 'micro precision', 'micro recall', 'micro f', 'micro fpr', 'micro tp', 'micro fn', 'micro tn', 'micro fp',
				'gregor accuracy', 'gregor precision', 'gregor recall', 'gregor f', 'gregor fpr', 'gregor tp', 'gregor fn', 'gregor tn', 'gregor fp',
				'grete accuracy', 'grete precision', 'grete recall', 'grete f', 'grete fpr', 'grete tp', 'grete fn', 'grete tn', 'grete fp',
				'samsa accuracy', 'samsa precision', 'samsa recall', 'samsa f', 'samsa fpr', 'samsa tp', 'samsa fn', 'samsa tn', 'samsa fp'
				]
	}

	contido_csv_ratas = [cabeceira['ratas']]
	if len(ficheiros_ratas) > 0:
		for fich in ficheiros_ratas:
			linha_csv_ratas = get_info_filename(fich)
			info = ficheiro.cargarJson(fich)

			linha_csv_32_ratas = linha_csv_ratas.copy()
			linha_csv_32_ratas.append('32')
			linha_csv_32_ratas.append(info['Dataset 32']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_32_ratas.append(info['Dataset 32']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_32_ratas.append(info['Dataset 32']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_32_ratas.append(info['Dataset 32']['tp|fn|tn|fp'].split('|')[3])
			linha_csv_32_ratas.append(str(info['Dataset 32']['accuracy']))
			linha_csv_32_ratas.append(str(info['Dataset 32']['precision']))
			linha_csv_32_ratas.append(str(info['Dataset 32']['recall']))
			linha_csv_32_ratas.append(str(info['Dataset 32']['f']))
			linha_csv_32_ratas.append(str(info['Dataset 32']['fpr']))
			contido_csv_ratas.append(linha_csv_32_ratas)

			linha_csv_64_ratas = linha_csv_ratas.copy()
			linha_csv_64_ratas.append('64')
			linha_csv_64_ratas.append(info['Dataset 64']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_64_ratas.append(info['Dataset 64']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_64_ratas.append(info['Dataset 64']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_64_ratas.append(info['Dataset 64']['tp|fn|tn|fp'].split('|')[3])
			linha_csv_64_ratas.append(str(info['Dataset 64']['accuracy']))
			linha_csv_64_ratas.append(str(info['Dataset 64']['precision']))
			linha_csv_64_ratas.append(str(info['Dataset 64']['recall']))
			linha_csv_64_ratas.append(str(info['Dataset 64']['f']))
			linha_csv_64_ratas.append(str(info['Dataset 64']['fpr']))
			contido_csv_ratas.append(linha_csv_64_ratas)

			linha_csv_128_ratas = linha_csv_ratas.copy()
			linha_csv_128_ratas.append('128')
			linha_csv_128_ratas.append(info['Dataset 128']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_128_ratas.append(info['Dataset 128']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_128_ratas.append(info['Dataset 128']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_128_ratas.append(info['Dataset 128']['tp|fn|tn|fp'].split('|')[3])
			linha_csv_128_ratas.append(str(info['Dataset 128']['accuracy']))
			linha_csv_128_ratas.append(str(info['Dataset 128']['precision']))
			linha_csv_128_ratas.append(str(info['Dataset 128']['recall']))
			linha_csv_128_ratas.append(str(info['Dataset 128']['f']))
			linha_csv_128_ratas.append(str(info['Dataset 128']['fpr']))
			contido_csv_ratas.append(linha_csv_128_ratas)

			if linha_csv_ratas[0] == 'inceptionv3':
				linha_csv_150_ratas = linha_csv_ratas.copy()
				linha_csv_150_ratas.append('150')
				linha_csv_150_ratas.append(info['Dataset 150']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_150_ratas.append(info['Dataset 150']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_150_ratas.append(info['Dataset 150']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_150_ratas.append(info['Dataset 150']['tp|fn|tn|fp'].split('|')[3])
				linha_csv_150_ratas.append(str(info['Dataset 150']['accuracy']))
				linha_csv_150_ratas.append(str(info['Dataset 150']['precision']))
				linha_csv_150_ratas.append(str(info['Dataset 150']['recall']))
				linha_csv_150_ratas.append(str(info['Dataset 150']['f']))
				linha_csv_150_ratas.append(str(info['Dataset 150']['fpr']))
				contido_csv_ratas.append(linha_csv_150_ratas)

			elif linha_csv_ratas[0] == 'vgg16' or linha_csv_ratas[0] == 'resnet50':
				linha_csv_224_ratas = linha_csv_ratas.copy()
				linha_csv_224_ratas.append('224')
				linha_csv_224_ratas.append(info['Dataset 224']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_224_ratas.append(info['Dataset 224']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_224_ratas.append(info['Dataset 224']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_224_ratas.append(info['Dataset 224']['tp|fn|tn|fp'].split('|')[3])
				linha_csv_224_ratas.append(str(info['Dataset 224']['accuracy']))
				linha_csv_224_ratas.append(str(info['Dataset 224']['precision']))
				linha_csv_224_ratas.append(str(info['Dataset 224']['recall']))
				linha_csv_224_ratas.append(str(info['Dataset 224']['f']))
				linha_csv_224_ratas.append(str(info['Dataset 224']['fpr']))
				contido_csv_ratas.append(linha_csv_224_ratas)

		# gardamos o contido no ficheiro
		gardar_csv(path_inicial+'/'+fich_saida+'_ratas.csv', contido_csv_ratas)

	contido_csv_cascudas = [cabeceira['cascudas']]
	if len(ficheiros_cascudas) > 0:
		for fich in ficheiros_cascudas:
			linha_csv_cascudas = get_info_filename(fich)
			info = ficheiro.cargarJson(fich)

			linha_csv_32_cascudas = linha_csv_cascudas.copy()
			linha_csv_32_cascudas.append('32')
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
			
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['macro']['accuracy']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['macro']['precision']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['macro']['recall']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['macro']['f']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['macro']['fpr']))
			
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['micro']['accuracy']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['micro']['precision']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['micro']['recall']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['micro']['f']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['total']['micro']['fpr']))
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_32_cascudas.append(info['Dataset 32']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_32_cascudas.append(str(info['Dataset 32']['gregor']['accuracy']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['gregor']['precision']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['gregor']['recall']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['gregor']['f']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['gregor']['fpr']))
			linha_csv_32_cascudas.append(info['Dataset 32']['gregor']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_32_cascudas.append(info['Dataset 32']['gregor']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_32_cascudas.append(info['Dataset 32']['gregor']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_32_cascudas.append(info['Dataset 32']['gregor']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_32_cascudas.append(str(info['Dataset 32']['grete']['accuracy']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['grete']['precision']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['grete']['recall']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['grete']['f']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['grete']['fpr']))
			linha_csv_32_cascudas.append(info['Dataset 32']['grete']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_32_cascudas.append(info['Dataset 32']['grete']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_32_cascudas.append(info['Dataset 32']['grete']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_32_cascudas.append(info['Dataset 32']['grete']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_32_cascudas.append(str(info['Dataset 32']['samsa']['accuracy']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['samsa']['precision']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['samsa']['recall']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['samsa']['f']))
			linha_csv_32_cascudas.append(str(info['Dataset 32']['samsa']['fpr']))
			linha_csv_32_cascudas.append(info['Dataset 32']['samsa']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_32_cascudas.append(info['Dataset 32']['samsa']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_32_cascudas.append(info['Dataset 32']['samsa']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_32_cascudas.append(info['Dataset 32']['samsa']['tp|fn|tn|fp'].split('|')[3])

			contido_csv_cascudas.append(linha_csv_32_cascudas)



			linha_csv_64_cascudas = linha_csv_cascudas.copy()
			linha_csv_64_cascudas.append('64')
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
			
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['macro']['accuracy']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['macro']['precision']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['macro']['recall']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['macro']['f']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['macro']['fpr']))
			
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['micro']['accuracy']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['micro']['precision']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['micro']['recall']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['micro']['f']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['total']['micro']['fpr']))
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_64_cascudas.append(info['Dataset 64']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_64_cascudas.append(str(info['Dataset 64']['gregor']['accuracy']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['gregor']['precision']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['gregor']['recall']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['gregor']['f']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['gregor']['fpr']))
			linha_csv_64_cascudas.append(info['Dataset 64']['gregor']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_64_cascudas.append(info['Dataset 64']['gregor']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_64_cascudas.append(info['Dataset 64']['gregor']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_64_cascudas.append(info['Dataset 64']['gregor']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_64_cascudas.append(str(info['Dataset 64']['grete']['accuracy']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['grete']['precision']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['grete']['recall']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['grete']['f']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['grete']['fpr']))
			linha_csv_64_cascudas.append(info['Dataset 64']['grete']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_64_cascudas.append(info['Dataset 64']['grete']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_64_cascudas.append(info['Dataset 64']['grete']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_64_cascudas.append(info['Dataset 64']['grete']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_64_cascudas.append(str(info['Dataset 64']['samsa']['accuracy']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['samsa']['precision']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['samsa']['recall']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['samsa']['f']))
			linha_csv_64_cascudas.append(str(info['Dataset 64']['samsa']['fpr']))
			linha_csv_64_cascudas.append(info['Dataset 64']['samsa']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_64_cascudas.append(info['Dataset 64']['samsa']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_64_cascudas.append(info['Dataset 64']['samsa']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_64_cascudas.append(info['Dataset 64']['samsa']['tp|fn|tn|fp'].split('|')[3])

			contido_csv_cascudas.append(linha_csv_64_cascudas)
			


			linha_csv_128_cascudas = linha_csv_cascudas.copy()
			linha_csv_128_cascudas.append('128')
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
			
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['macro']['accuracy']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['macro']['precision']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['macro']['recall']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['macro']['f']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['macro']['fpr']))
			
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['micro']['accuracy']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['micro']['precision']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['micro']['recall']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['micro']['f']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['total']['micro']['fpr']))
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_128_cascudas.append(info['Dataset 128']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_128_cascudas.append(str(info['Dataset 128']['gregor']['accuracy']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['gregor']['precision']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['gregor']['recall']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['gregor']['f']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['gregor']['fpr']))
			linha_csv_128_cascudas.append(info['Dataset 128']['gregor']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_128_cascudas.append(info['Dataset 128']['gregor']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_128_cascudas.append(info['Dataset 128']['gregor']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_128_cascudas.append(info['Dataset 128']['gregor']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_128_cascudas.append(str(info['Dataset 128']['grete']['accuracy']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['grete']['precision']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['grete']['recall']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['grete']['f']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['grete']['fpr']))
			linha_csv_128_cascudas.append(info['Dataset 128']['grete']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_128_cascudas.append(info['Dataset 128']['grete']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_128_cascudas.append(info['Dataset 128']['grete']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_128_cascudas.append(info['Dataset 128']['grete']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_128_cascudas.append(str(info['Dataset 128']['samsa']['accuracy']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['samsa']['precision']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['samsa']['recall']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['samsa']['f']))
			linha_csv_128_cascudas.append(str(info['Dataset 128']['samsa']['fpr']))
			linha_csv_128_cascudas.append(info['Dataset 128']['samsa']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_128_cascudas.append(info['Dataset 128']['samsa']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_128_cascudas.append(info['Dataset 128']['samsa']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_128_cascudas.append(info['Dataset 128']['samsa']['tp|fn|tn|fp'].split('|')[3])

			contido_csv_cascudas.append(linha_csv_128_cascudas)


			if linha_csv_cascudas[0] == 'inceptionv3':
				linha_csv_150_cascudas = linha_csv_cascudas.copy()
				linha_csv_150_cascudas.append('150')
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
				
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['macro']['accuracy']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['macro']['precision']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['macro']['recall']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['macro']['f']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['macro']['fpr']))
				
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['micro']['accuracy']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['micro']['precision']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['micro']['recall']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['micro']['f']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['total']['micro']['fpr']))
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_150_cascudas.append(info['Dataset 150']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_150_cascudas.append(str(info['Dataset 150']['gregor']['accuracy']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['gregor']['precision']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['gregor']['recall']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['gregor']['f']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['gregor']['fpr']))
				linha_csv_150_cascudas.append(info['Dataset 150']['gregor']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_150_cascudas.append(info['Dataset 150']['gregor']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_150_cascudas.append(info['Dataset 150']['gregor']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_150_cascudas.append(info['Dataset 150']['gregor']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_150_cascudas.append(str(info['Dataset 150']['grete']['accuracy']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['grete']['precision']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['grete']['recall']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['grete']['f']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['grete']['fpr']))
				linha_csv_150_cascudas.append(info['Dataset 150']['grete']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_150_cascudas.append(info['Dataset 150']['grete']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_150_cascudas.append(info['Dataset 150']['grete']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_150_cascudas.append(info['Dataset 150']['grete']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_150_cascudas.append(str(info['Dataset 150']['samsa']['accuracy']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['samsa']['precision']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['samsa']['recall']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['samsa']['f']))
				linha_csv_150_cascudas.append(str(info['Dataset 150']['samsa']['fpr']))
				linha_csv_150_cascudas.append(info['Dataset 150']['samsa']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_150_cascudas.append(info['Dataset 150']['samsa']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_150_cascudas.append(info['Dataset 150']['samsa']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_150_cascudas.append(info['Dataset 150']['samsa']['tp|fn|tn|fp'].split('|')[3])
				
				contido_csv_cascudas.append(linha_csv_150_cascudas)




			elif linha_csv_cascudas[0] == 'vgg16' or linha_csv_cascudas[0] == 'resnet50':
				linha_csv_224_cascudas = linha_csv_cascudas.copy()
				linha_csv_224_cascudas.append('224')
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
				
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['macro']['accuracy']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['macro']['precision']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['macro']['recall']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['macro']['f']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['macro']['fpr']))
				
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['micro']['accuracy']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['micro']['precision']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['micro']['recall']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['micro']['f']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['total']['micro']['fpr']))
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_224_cascudas.append(info['Dataset 224']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_224_cascudas.append(str(info['Dataset 224']['gregor']['accuracy']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['gregor']['precision']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['gregor']['recall']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['gregor']['f']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['gregor']['fpr']))
				linha_csv_224_cascudas.append(info['Dataset 224']['gregor']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_224_cascudas.append(info['Dataset 224']['gregor']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_224_cascudas.append(info['Dataset 224']['gregor']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_224_cascudas.append(info['Dataset 224']['gregor']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_224_cascudas.append(str(info['Dataset 224']['grete']['accuracy']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['grete']['precision']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['grete']['recall']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['grete']['f']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['grete']['fpr']))
				linha_csv_224_cascudas.append(info['Dataset 224']['grete']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_224_cascudas.append(info['Dataset 224']['grete']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_224_cascudas.append(info['Dataset 224']['grete']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_224_cascudas.append(info['Dataset 224']['grete']['tp|fn|tn|fp'].split('|')[3])

				linha_csv_224_cascudas.append(str(info['Dataset 224']['samsa']['accuracy']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['samsa']['precision']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['samsa']['recall']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['samsa']['f']))
				linha_csv_224_cascudas.append(str(info['Dataset 224']['samsa']['fpr']))
				linha_csv_224_cascudas.append(info['Dataset 224']['samsa']['tp|fn|tn|fp'].split('|')[0])
				linha_csv_224_cascudas.append(info['Dataset 224']['samsa']['tp|fn|tn|fp'].split('|')[1])
				linha_csv_224_cascudas.append(info['Dataset 224']['samsa']['tp|fn|tn|fp'].split('|')[2])
				linha_csv_224_cascudas.append(info['Dataset 224']['samsa']['tp|fn|tn|fp'].split('|')[3])
				
				contido_csv_cascudas.append(linha_csv_224_cascudas)




			linha_csv_256_cascudas = linha_csv_cascudas.copy()
			linha_csv_256_cascudas.append('256')
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[0])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[1])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[2])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[3])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[4])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[5])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[6])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[7])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'].split('|')[8])
			
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['macro']['accuracy']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['macro']['precision']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['macro']['recall']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['macro']['f']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['macro']['fpr']))
			
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['micro']['accuracy']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['micro']['precision']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['micro']['recall']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['micro']['f']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['total']['micro']['fpr']))
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['micro']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['micro']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['micro']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_256_cascudas.append(info['Dataset 256']['total']['micro']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_256_cascudas.append(str(info['Dataset 256']['gregor']['accuracy']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['gregor']['precision']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['gregor']['recall']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['gregor']['f']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['gregor']['fpr']))
			linha_csv_256_cascudas.append(info['Dataset 256']['gregor']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_256_cascudas.append(info['Dataset 256']['gregor']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_256_cascudas.append(info['Dataset 256']['gregor']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_256_cascudas.append(info['Dataset 256']['gregor']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_256_cascudas.append(str(info['Dataset 256']['grete']['accuracy']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['grete']['precision']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['grete']['recall']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['grete']['f']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['grete']['fpr']))
			linha_csv_256_cascudas.append(info['Dataset 256']['grete']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_256_cascudas.append(info['Dataset 256']['grete']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_256_cascudas.append(info['Dataset 256']['grete']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_256_cascudas.append(info['Dataset 256']['grete']['tp|fn|tn|fp'].split('|')[3])

			linha_csv_256_cascudas.append(str(info['Dataset 256']['samsa']['accuracy']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['samsa']['precision']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['samsa']['recall']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['samsa']['f']))
			linha_csv_256_cascudas.append(str(info['Dataset 256']['samsa']['fpr']))
			linha_csv_256_cascudas.append(info['Dataset 256']['samsa']['tp|fn|tn|fp'].split('|')[0])
			linha_csv_256_cascudas.append(info['Dataset 256']['samsa']['tp|fn|tn|fp'].split('|')[1])
			linha_csv_256_cascudas.append(info['Dataset 256']['samsa']['tp|fn|tn|fp'].split('|')[2])
			linha_csv_256_cascudas.append(info['Dataset 256']['samsa']['tp|fn|tn|fp'].split('|')[3])

			contido_csv_cascudas.append(linha_csv_256_cascudas)


		# gardamos o contido no ficheiro
		gardar_csv(path_inicial+'/'+fich_saida+'_cascudas.csv', contido_csv_cascudas)

if __name__ == '__main__':
	main()
#-----------------------------------------------------------
