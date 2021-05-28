#!/usr/bin/python3
#-----------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	01/05/2021 16:30:52
#+ Editado:	06/05/2021 21:40:05
#-----------------------------------------------------------
import os
import sys
import csv
import json
#-----------------------------------------------------------
def cargar_json(fich):
    if os.path.isfile(fich):
        return json.loads(open(fich).read())
    else:
        open(fich, 'w').write('{}')
        return json.loads(open(fich).read())
#-----------------------------------------------------------
def gardar_csv(nome, contido):
    with open(nome, 'w') as csvfile:
        for row in contido:
            csvfile.write(','.join(row))
            csvfile.write('\n')

#-----------------------------------------------------------
def get_info_filename(nome_fich):
    saida = []
    for ele in nome_fich.split('___')[0].split('_'):
       saida.append(ele.split(';')[1]) 
    return saida
#-----------------------------------------------------------
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
ficheiros = []
for fich in os.listdir():
    ficheiros.append(fich)
# e ordeámolos
ficheiros.sort()

cabeceira_csv = ['train dataset', 'epoch', 'batch size', 'valid dataset', 'tp', 'fn', 'tn', 'fp', 'accuracy', 'precision', 'recall', 'f', 'fpr']
contido_csv = [cabeceira_csv]

# por cada ficheiro da carpeta
for fich in ficheiros:
    linha_csv = get_info_filename(fich)
    
    info = cargar_json(fich)
    
    linha_csv_32 = linha_csv.copy()
    linha_csv_32.append('32')
    linha_csv_32.append(info['Dataset_32']['tp|fn|tn|fp'].split('|')[0])
    linha_csv_32.append(info['Dataset_32']['tp|fn|tn|fp'].split('|')[1])
    linha_csv_32.append(info['Dataset_32']['tp|fn|tn|fp'].split('|')[2])
    linha_csv_32.append(info['Dataset_32']['tp|fn|tn|fp'].split('|')[3])
    linha_csv_32.append(str(info['Dataset_32']['accuracy']))
    linha_csv_32.append(str(info['Dataset_32']['precision']))
    linha_csv_32.append(str(info['Dataset_32']['recall']))
    linha_csv_32.append(str(info['Dataset_32']['f']))
    linha_csv_32.append(str(info['Dataset_32']['fpr']))

    linha_csv_64 = linha_csv.copy()
    linha_csv_64.append('64')
    linha_csv_64.append(info['Dataset_64']['tp|fn|tn|fp'].split('|')[0])
    linha_csv_64.append(info['Dataset_64']['tp|fn|tn|fp'].split('|')[1])
    linha_csv_64.append(info['Dataset_64']['tp|fn|tn|fp'].split('|')[2])
    linha_csv_64.append(info['Dataset_64']['tp|fn|tn|fp'].split('|')[3])
    linha_csv_64.append(str(info['Dataset_64']['accuracy']))
    linha_csv_64.append(str(info['Dataset_64']['precision']))
    linha_csv_64.append(str(info['Dataset_64']['recall']))
    linha_csv_64.append(str(info['Dataset_64']['f']))
    linha_csv_64.append(str(info['Dataset_64']['fpr']))

    linha_csv_128 = linha_csv.copy()
    linha_csv_128.append('128')
    linha_csv_128.append(info['Dataset_128']['tp|fn|tn|fp'].split('|')[0])
    linha_csv_128.append(info['Dataset_128']['tp|fn|tn|fp'].split('|')[1])
    linha_csv_128.append(info['Dataset_128']['tp|fn|tn|fp'].split('|')[2])
    linha_csv_128.append(info['Dataset_128']['tp|fn|tn|fp'].split('|')[3])
    linha_csv_128.append(str(info['Dataset_128']['accuracy']))
    linha_csv_128.append(str(info['Dataset_128']['precision']))
    linha_csv_128.append(str(info['Dataset_128']['recall']))
    linha_csv_128.append(str(info['Dataset_128']['f']))
    linha_csv_128.append(str(info['Dataset_128']['fpr']))

    contido_csv.append(linha_csv_32)
    contido_csv.append(linha_csv_64)
    contido_csv.append(linha_csv_128)
 
# gardamos o contido no ficheiro
gardar_csv(fich_saida+'.csv', contido_csv)

#-----------------------------------------------------------
