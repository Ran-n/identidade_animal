#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	16/06/2021 18:34:16
#+ Editado:	29/07/2021 14:15:14
#------------------------------------------------------------------------------------------------

import sys

# meto este bloque aquí e non despois para non ter que cargar sempre tensorflow se pide axuda e así ter unha resposta rápida
# miramos se ten entradas por comando e se as ten os valores deben ser postos a tal
# quitamos a primeira
__args = sys.argv[1:]
# mensaxe de axuda
if ('-a' in __args) or ('-h' in __args) or ('?' in __args) or ('-?' in __args) or (len(__args) == 0):
    print('\nAxuda -----------')
    print('-h/-a/?\t\t-> Para esta mensaxe')
    print()
    print('-e num\t\t-> epochs\t\t\t\t\t(20)')
    print('-b num\t\t-> batch size\t\t\t\t\t(32)')
    print('-v num\t\t-> Distribución dos % train-val-test\t\t(80-10-10)')
    print('-s num\t\t-> semente\t\t\t\t\t(1)')
    print('----------------\n')
    
    if len(__args) != 0:
        sys.exit()

import os
# eliminar os warnings de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import numpy as np
import sys
import json
import secrets
import math

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# módulo uteis https://www.github.com/Ran-n/uteis.git
from uteis import ficheiro

# ------------------------------------------------------------------------------------------------

# para que me devolva 032 e non 32
def nomenclar(numero):
  if len(str(numero)) == 1:
    return '00'+str(numero)
  elif len(str(numero)) == 2:
    return '0'+str(numero)
  return str(numero)

# calcula a regra de tres
def r3(a, b, c):
    return (b*c)/a

# función encargada de darlle sentido ás opcións de entrada
def lecturaOpcions(args):
    # tan só se pode entrenar con estas dimensións
    ALTURA_IMAXE = ANCHURA_IMAXE = 224
    DIMENSIONS = '224x224'

    # epochs que se realizan
    if '-e' in args:
        EPOCHS = int(args[args.index('-e')+1])
    else:
        EPOCHS=20

    # tamaí±o da batch
    if '-b' in args:
        BATCH_SIZE = int(args[args.index('-b')+1])
    else:
        BATCH_SIZE = 32

    # porcentaxe a usar para a validación
    if '-v' in args:
        try:
            CANTIDADES = [int(ele) for ele in args[args.index('-v')+1].split('-')]
        except:
            raise

        # se non dan un erro
        assert(sum(CANTIDADES) == 100)
        CANTIDADES = [ele/100 for ele in CANTIDADES]
    else:
        CANTIDADES = [0.8, 0.1, 0.1]

    # semente a usar
    if '-s' in args:
        SEMENTE = int(args[args.index('-s')+1])
    else:
        SEMENTE = 1

    return DIMENSIONS, ALTURA_IMAXE, ANCHURA_IMAXE, EPOCHS, BATCH_SIZE, CANTIDADES, SEMENTE

def calculo_metricas(tp, fn, tn, fp):
        acc = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
        precision =  tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
        fpr = fp/(fp+tn) if fp+tn else 0

        return acc, precision, recall, f, fpr

def matriz_confusion3a2(posicion_positivo, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3):
    """
        posicion_positivo --> clase A > 1 ; clase B > 2 ; clase C > 3
    """
    if posicion_positivo == 1:
        tp = f1c1
        fp = f1c2+f1c3
        fn = f2c1+f3c1
        tn = f2c2+f2c3+f3c2+f3c3

    elif posicion_positivo == 2:
        tp = f2c2
        fn = f2c3+f2c1
        fp = f3c2+f1c2
        tn = f3c3+f3c1+f1c3+f1c1

    elif posicion_positivo == 3:
        tn = f1c1+f1c2+f2c1+f2c2
        fp = f1c3+f2c3
        fn = f3c1+f3c2
        tp = f3c3

    else:
        raise Exception('Non existe esa posición nunha matriz 3x3')

    return tp, fn, tn, fp

def medias_metricas_cascudas(dataset, metricas, metrica):
    a = metricas['Dataset '+dataset]['gregor'][metrica]
    b = metricas['Dataset '+dataset]['grete'][metrica]
    c = metricas['Dataset '+dataset]['samsa'][metrica]

    return (a+b+c)/3

#------------------------------------------------------------------------------------------------

# main
if __name__=='__main__':
    # ler as opcións de entrada
    DIMENSIONS, ALTURA_IMAXE, ANCHURA_IMAXE, EPOCHS, BATCH_SIZE, CANTIDADES, SEMENTE = lecturaOpcions(__args)
    NOMENCLATURA = 'dataset-cascudas-resnet50_'+nomenclar(ALTURA_IMAXE)+';epochs_'+nomenclar(EPOCHS)+';batch-size_'+nomenclar(BATCH_SIZE)+';semente_'+nomenclar(SEMENTE)+'___'+str(secrets.token_hex(4))
    CARPETA = 'saidas/cascudas_resnet50/'+NOMENCLATURA
    FICHEIRO = CARPETA+'/'+NOMENCLATURA
    
    # dicc cos distintos datasets de proba a usar
    test_ds = {
        '32': None,
        '64': None,
        '128': None,
        '224': None,
        '256': None
    }

    cant_imaxes_test = {
        '32': None,
        '64': None,
        '128': None,
        '224': None,
        '256': None
    }

    '''
    MODELADO
    '''

    ligazon_conxundo_datos = 'https://bucketfg.blob.core.windows.net/datasets2/Dataset_Cascudas_'+str(ALTURA_IMAXE)+'x'+str(ANCHURA_IMAXE)+'.tar.gz'
    # colle a ligazón e de ahí baixa o cdd
    # feito así para podelo usar en calquer ordenador se problemas
    data_dir = tf.keras.utils.get_file('Dataset_Cascudas_'+str(ALTURA_IMAXE)+'x'+str(ANCHURA_IMAXE), origin=ligazon_conxundo_datos, untar=True)
    data_dir = pathlib.Path(data_dir)

    print()
    print('* Cargando dataset *')
    # cargamos o dataset
    ds = image_dataset_from_directory(
            data_dir,
            seed=SEMENTE,
            image_size=(ALTURA_IMAXE, ANCHURA_IMAXE),
            batch_size=BATCH_SIZE
    )

    # nomes das clases
    nome_clases = ds.class_names

    # mostra a mensaxe de información sobre a configuración usada para a creación do modelo
    print('\nConfiguración a usar:')
    print('---------------------------------------------------------------------------------------------------')
    print('nomenclatura:\t\t', NOMENCLATURA)
    print('dimensións:\t\t', DIMENSIONS)
    print('epochs:\t\t\t', EPOCHS)
    print('batch size:\t\t', BATCH_SIZE)
    print('cant clases:\t\t', len(nome_clases))
    print('semente:\t\t', SEMENTE)
    print('% train-val-test:\t {} - {} - {}'.format(CANTIDADES[0], CANTIDADES[1], CANTIDADES[2]))
    print('---------------------------------------------------------------------------------------------------')

    # gardar os parámetros usados
    ficheiro.gardarJson(FICHEIRO+'.parametros',
        {
            'nomenclatura': NOMENCLATURA,
            'dimensións': DIMENSIONS,
            'epochs': EPOCHS,
            'batch size': BATCH_SIZE,
            'cant clases': len(nome_clases),
            'semente': SEMENTE,
            '% train': CANTIDADES[0],
            '% validation': CANTIDADES[1],
            '% test': CANTIDADES[2]
        })

    # tamaños a coller
    ds_lonx = len(ds)
    cant_imaxes_ds = len(np.concatenate([i for x, i in ds], axis=0))

    tamanho_train_ds = math.ceil(CANTIDADES[0]*ds_lonx)
    tamanho_valid_ds = math.ceil(CANTIDADES[1]*ds_lonx)
    # non preciso o de test porque sempre é o resto dos outros dous
    tamanho_test_ds = ds_lonx-tamanho_train_ds-tamanho_valid_ds

    ds = ds.shuffle(cant_imaxes_ds+1, seed=SEMENTE+1)
    train_ds = ds.take(tamanho_train_ds)
    remaining = ds.skip(tamanho_train_ds)
    valid_ds = remaining.take(tamanho_valid_ds)
    test_ds[str(ALTURA_IMAXE)] = remaining.skip(tamanho_valid_ds)

    # sacar os tests do resto, coa mesma cantidade de imaxes
    # non poño 224 porque o ese test set sempre é o que collo primeiro
    # pq non se pode adestrar con outro
    for ele in ['32', '64', '128', '256']:
        # non fai falla o if do ratas normal porque as dimensións 224 sempre van ser
        # as primeiras en ser cargadas e non fai falla poñelas no bucle
        data_dir2 = tf.keras.utils.get_file(
                                   'Dataset_Cascudas_'+str(ele)+'x'+str(ele),
                                    origin='https://bucketfg.blob.core.windows.net/datasets2/Dataset_Cascudas_'+str(ele)+'x'+str(ele)+'.tar.gz',
                                    untar=True)

        data_dir2 = pathlib.Path(data_dir2)

        print('\n* Baixando tamén o dataset {0}x{0} para sacar o conxunto de proba deste *'.format(ele))
        ds2 = image_dataset_from_directory(
                data_dir2,
                seed=SEMENTE,
                #image_size=(int(ele), int(ele)),
                image_size=(ALTURA_IMAXE, ANCHURA_IMAXE),
                batch_size=BATCH_SIZE
        )

        test_ds[ele] = ds2.take(tamanho_test_ds)
        cant_imaxes_test[ele] = len(np.concatenate([i for x, i in test_ds[ele]], axis=0))

    cant_imaxes_train = len(np.concatenate([i for x, i in train_ds], axis=0))
    cant_imaxes_valid = len(np.concatenate([i for x, i in valid_ds], axis=0))
    cant_imaxes_test[str(ALTURA_IMAXE)] = len(np.concatenate([i for x, i in test_ds[str(ALTURA_IMAXE)]], axis=0))

    print()
    print('\nImaxes separadas tal que:')
    print('----------------------------------')
    print('{} para o train dataset\n{} para o valid dataset\n{} para o test dataset'.format(cant_imaxes_train, cant_imaxes_valid, cant_imaxes_test[str(ALTURA_IMAXE)]))
    print('----------------------------------')
    print('Suman {} imaxes das {} iniciais'.format(cant_imaxes_train+cant_imaxes_valid+cant_imaxes_test[str(ALTURA_IMAXE)], cant_imaxes_ds))
    print()


    # gárdase a distribución de imaxes    
    ficheiro.gardarJson(FICHEIRO+'.distribucion',
                        {'Dimensións':  DIMENSIONS,
                        'Epochs': EPOCHS,
                        'Batch Size': BATCH_SIZE,
                        'Clases': nome_clases,
                        'Cantidade Clases': len(nome_clases),
                        '% train-val-test': CANTIDADES,
                        'semente': SEMENTE,
                        'Distribución Imaxes': {
                            'Train': cant_imaxes_train,
                            'Validation': cant_imaxes_valid,
                            'Test': cant_imaxes_test,
                            'Total': cant_imaxes_ds
                        },
                        'Batchs coas imaxes': {
                            'Train': tamanho_train_ds,
                            'Validation': tamanho_valid_ds,
                            'Test': tamanho_test_ds,
                            'Total': ds_lonx
                        }}, indent=4)

    # precargar o dataset en memoria
    # todo este bloque e para cargar o dataset de buffer e non de memoria para evitar pescozos de botella
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # CREACIÓN da rede ResNet50 #
    base_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")
    base_model.summary()

    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten(name = "flatten")(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(len(nome_clases))(x)

    modelo = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    # CREACIÓN da rede ResNet50 #

    # compilar o modelo
    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # mostra a información sobre o modelo
    # modelo.summary()

    print()
    print('* Adestrando *')
    print()

    # facemos o fit do modelo
    historia = modelo.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS
    )

    # fanse os gráficos
    acc = historia.history['accuracy']
    val_acc = historia.history['val_accuracy']
    loss = historia.history['loss']
    val_loss = historia.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Adestramento')
    plt.plot(epochs_range, val_acc, label='Validación')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.title('Precisión de adestramento e validación')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Adestramento')
    plt.plot(epochs_range, val_loss, label='Validación')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.title('Perda de adestramento e validación')

    # gardamos a gráfica
    for extension in ['png', 'svg', 'pdf']:
        plt.savefig(CARPETA+'/'+NOMENCLATURA+'.'+extension, format=extension)
    #gardamos o modelo
    modelo.save(CARPETA+'/'+NOMENCLATURA)

    # gardamos as clases
    ficheiro.gardarFich(FICHEIRO+'.clases', nome_clases)
    ficheiro.gardarJson(FICHEIRO+'.historia', historia.history)

    # creación das gráficas en inglés
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='$\\itTrain$')
    plt.plot(epochs_range, val_acc, label='$\\itValidation$')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.title('Precisión de $\\it{}$ e $\\it{}$'.format('train', 'validation'))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='$\\itTrain$')
    plt.plot(epochs_range, val_loss, label='$\\itValidation$')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.title('Perda de $\\it{}$ e $\\it{}$'.format('train', 'validation'))

    # gardamos a gráfica
    for extension in ['png', 'svg', 'pdf']:
        plt.savefig(FICHEIRO+' [en].'+extension, format=extension)


    '''
    TEST
    '''

    print()
    print('* Predicindo *')
    print()

    prediccions = {
        'Dataset 32':{
            'gregor': {'acertos': 0, 'erros': {'grete': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'grete': {'acertos': 0, 'erros': {'gregor': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'samsa': {'acertos': 0, 'erros': {'gregor': 0, 'grete':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'total imaxes': 0,
            'porcentaxes imaxes': {'gregor': 0, 'grete': 0, 'samsa': 0},
            'media porcentaxes': {'gregor': {'acertos': 0, 'erros': 0},
                                  'grete': {'acertos': 0, 'erros': 0},
                                  'samsa': {'acertos': 0, 'erros': 0}
                                  }
        },
        'Dataset 64':{
            'gregor': {'acertos': 0, 'erros': {'grete': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'grete': {'acertos': 0, 'erros': {'gregor': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'samsa': {'acertos': 0, 'erros': {'gregor': 0, 'grete':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'total imaxes': 0,
            'porcentaxes imaxes': {'gregor': 0, 'grete': 0, 'samsa': 0},
            'media porcentaxes': {'gregor': {'acertos': 0, 'erros': 0},
                                  'grete': {'acertos': 0, 'erros': 0},
                                  'samsa': {'acertos': 0, 'erros': 0}
                                  }
        },
        'Dataset 128':{
            'gregor': {'acertos': 0, 'erros': {'grete': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'grete': {'acertos': 0, 'erros': {'gregor': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'samsa': {'acertos': 0, 'erros': {'gregor': 0, 'grete':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'total imaxes': 0,
            'porcentaxes imaxes': {'gregor': 0, 'grete': 0, 'samsa': 0},
            'media porcentaxes': {'gregor': {'acertos': 0, 'erros': 0},
                                  'grete': {'acertos': 0, 'erros': 0},
                                  'samsa': {'acertos': 0, 'erros': 0}
                                  }
        },
        'Dataset 224':{
            'gregor': {'acertos': 0, 'erros': {'grete': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'grete': {'acertos': 0, 'erros': {'gregor': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'samsa': {'acertos': 0, 'erros': {'gregor': 0, 'grete':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'total imaxes': 0,
            'porcentaxes imaxes': {'gregor': 0, 'grete': 0, 'samsa': 0},
            'media porcentaxes': {'gregor': {'acertos': 0, 'erros': 0},
                                  'grete': {'acertos': 0, 'erros': 0},
                                  'samsa': {'acertos': 0, 'erros': 0}
                                  }
        },
        'Dataset 256':{
            'gregor': {'acertos': 0, 'erros': {'grete': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'grete': {'acertos': 0, 'erros': {'gregor': 0, 'samsa':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'samsa': {'acertos': 0, 'erros': {'gregor': 0, 'grete':0}, 'total': 0, 'porcentaxes': {'acertos': 0, 'erros': 0}},
            'total imaxes': 0,
            'porcentaxes imaxes': {'gregor': 0, 'grete': 0, 'samsa': 0},
            'media porcentaxes': {'gregor': {'acertos': 0, 'erros': 0},
                                  'grete': {'acertos': 0, 'erros': 0},
                                  'samsa': {'acertos': 0, 'erros': 0}
                                  }
        }
    }

    cronoloxia = {
        'Dataset 32':{'gregor': {'cronoloxía': [], 'confianza': []},
                      'grete':  {'cronoloxía': [], 'confianza': []},
                      'samsa':  {'cronoloxía': [], 'confianza': []}
                        },
        'Dataset 64':{'gregor': {'cronoloxía': [], 'confianza': []},
                      'grete':  {'cronoloxía': [], 'confianza': []},
                      'samsa':  {'cronoloxía': [], 'confianza': []}
                        },
        'Dataset 128':{'gregor': {'cronoloxía': [], 'confianza': []},
                      'grete':   {'cronoloxía': [], 'confianza': []},
                      'samsa':   {'cronoloxía': [], 'confianza': []}
                        },
        'Dataset 224':{'gregor': {'cronoloxía': [], 'confianza': []},
                      'grete':   {'cronoloxía': [], 'confianza': []},
                      'samsa':   {'cronoloxía': [], 'confianza': []}
                        },
        'Dataset 256':{'gregor': {'cronoloxía': [], 'confianza': []},
                      'grete':   {'cronoloxía': [], 'confianza': []},
                      'samsa':   {'cronoloxía': [], 'confianza': []}
                        },

    }

    metricas = {
        'Dataset 32': {
            'gregor': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'grete': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'samsa': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'total': {'f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3': '',
                      'macro': {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      },
                      'micro': {
                        'tp|fn|tn|fp': '',
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      }
            }
        },
        'Dataset 64': {
            'gregor': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'grete': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'samsa': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'total': {'f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3': '',
                      'macro': {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      },
                      'micro': {
                        'tp|fn|tn|fp': '',
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      }
            }
        },
        'Dataset 128': {
            'gregor': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'grete': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'samsa': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'total': {'f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3': '',
                      'macro': {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      },
                      'micro': {
                        'tp|fn|tn|fp': '',
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      }
            }
        },
        'Dataset 224': {
            'gregor': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'grete': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'samsa': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'total': {'f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3': '',
                      'macro': {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      },
                      'micro': {
                        'tp|fn|tn|fp': '',
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      }
            }
        },
        'Dataset 256': {
            'gregor': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'grete': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'samsa': {'tp|fn|tn|fp': '', 'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0},
            'total': {'f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3': '',
                      'macro': {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      },
                      'micro': {
                        'tp|fn|tn|fp': '',
                        'accuracy': 0, 'precision': 0, 'recall': 0, 'f': 0, 'fpr': 0
                      }
            }
        }
    }

    avaliacion = {
        'Dataset 32': None,
        'Dataset 64': None,
        'Dataset 128': None,
        'Dataset 224': None,
        'Dataset 256': None
    }

    for dataset in ['32', '64', '128', '224', '256']:
        print('* Dataset de test {} *'.format(dataset))

        labels = []
        imaxes = []
        for img, label in test_ds[dataset]:
            labels.append(label)
            imaxes.append(img)

        labels = np.concatenate(labels, axis=0)
        imaxes = np.concatenate(imaxes, axis=0)

        clases_verdadeiras = [nome_clases[ele] for ele in labels]

        lista_prediccions = modelo.predict(imaxes)
        clases_preditas = [nome_clases[np.argmax(ele)] for ele in lista_prediccions]
        puntaxes_preditas = [np.max(ele) for ele in lista_prediccions]

        for verdadeiro, predito, puntaxe_predita in zip(clases_verdadeiras, clases_preditas, puntaxes_preditas):
            prediccions['Dataset '+dataset]['total imaxes'] += 1
            prediccions['Dataset '+dataset][verdadeiro]['total'] += 1

            acertou = int(verdadeiro == predito)
            if acertou:
                prediccions['Dataset '+dataset][verdadeiro]['acertos'] += 1
                cronoloxia['Dataset '+dataset][verdadeiro]['cronoloxía'].append(acertou)
            else:
                #print('{} {}'.format(verdadeiro, predito))
                prediccions['Dataset '+dataset][verdadeiro]['erros'][predito] += 1
                cronoloxia['Dataset '+dataset][verdadeiro]['cronoloxía'].append([acertou, verdadeiro])

            cronoloxia['Dataset '+dataset][verdadeiro]['confianza'].append(str(puntaxe_predita))

        for cascuda in ['gregor', 'grete', 'samsa']:
            total = prediccions['Dataset '+dataset][cascuda]['total']
            acertos = prediccions['Dataset '+dataset][cascuda]['acertos']
            erros = 0
            for cascuda2 in ['gregor', 'grete', 'samsa']:
                if cascuda2 != cascuda:
                    erros += prediccions['Dataset '+dataset][cascuda]['erros'][cascuda2]
            
            prediccions['Dataset '+dataset][cascuda]['porcentaxes']['acertos'] = r3(total, 100, acertos)
            prediccions['Dataset '+dataset][cascuda]['porcentaxes']['erros'] = r3(total, 100, erros)

        total_imaxes = prediccions['Dataset '+dataset]['total imaxes']
        total_gregor = prediccions['Dataset '+dataset]['gregor']['total']
        total_grete = prediccions['Dataset '+dataset]['grete']['total']
        total_samsa = prediccions['Dataset '+dataset]['samsa']['total']
        prediccions['Dataset '+dataset]['porcentaxes imaxes']['gregor'] = r3(total_imaxes, 100, total_gregor)
        prediccions['Dataset '+dataset]['porcentaxes imaxes']['grete'] = r3(total_imaxes, 100, total_grete)
        prediccions['Dataset '+dataset]['porcentaxes imaxes']['samsa'] = r3(total_imaxes, 100, total_samsa)

        acertos_gregor = prediccions['Dataset '+dataset]['gregor']['porcentaxes']['acertos']
        acertos_grete = prediccions['Dataset '+dataset]['grete']['porcentaxes']['acertos']
        acertos_samsa = prediccions['Dataset '+dataset]['samsa']['porcentaxes']['acertos']
        erros_gregor = prediccions['Dataset '+dataset]['gregor']['porcentaxes']['erros']
        erros_grete = prediccions['Dataset '+dataset]['grete']['porcentaxes']['erros']
        erros_samsa = prediccions['Dataset '+dataset]['samsa']['porcentaxes']['erros']
        prediccions['Dataset '+dataset]['media porcentaxes']['acertos'] = (acertos_gregor+acertos_grete+acertos_samsa)/3
        prediccions['Dataset '+dataset]['media porcentaxes']['erros'] = (erros_grete+erros_grete+erros_samsa)/3

        # crear a matriz de confusión 3x3
        f1c1 = prediccions['Dataset '+dataset]['gregor']['acertos']
        f1c2 = prediccions['Dataset '+dataset]['gregor']['erros']['grete']
        f1c3 = prediccions['Dataset '+dataset]['gregor']['erros']['samsa']
        f2c1 = prediccions['Dataset '+dataset]['grete']['erros']['gregor']
        f2c2 = prediccions['Dataset '+dataset]['grete']['acertos']
        f2c3 = prediccions['Dataset '+dataset]['grete']['erros']['samsa']
        f3c1 = prediccions['Dataset '+dataset]['samsa']['erros']['gregor']
        f3c2 = prediccions['Dataset '+dataset]['samsa']['erros']['grete']
        f3c3 = prediccions['Dataset '+dataset]['samsa']['acertos']

        # gardamos a matriz de confusión 3x3
        metricas['Dataset '+dataset]['total']['f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3'] = str(f1c1)+'|'+str(f1c2)+'|'+str(f1c3)+'|'+str(f2c1)+'|'+str(f2c2)+'|'+str(f2c3)+'|'+str(f3c1)+'|'+str(f3c2)+'|'+str(f3c3)

        # calculamos as métricas de cada cascuda indiviual
        tp, fn, tn, fp = matriz_confusion3a2(1, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
        metricas['Dataset '+dataset]['gregor']['tp|fn|tn|fp'] = str(tp)+'|'+str(fn)+'|'+str(tn)+'|'+str(fp)
        metricas['Dataset '+dataset]['gregor']['accuracy'] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
        metricas['Dataset '+dataset]['gregor']['precision'] = precision =  tp/(tp+fp) if tp+fp else 0
        metricas['Dataset '+dataset]['gregor']['recall'] = recall = tp/(tp+fn) if tp+fn else 0
        metricas['Dataset '+dataset]['gregor']['f'] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
        metricas['Dataset '+dataset]['gregor']['fpr'] = fp/(fp+tn) if fp+tn else 0


        # calculamos as métricas de cada cascuda indiviual
        tp, fn, tn, fp = matriz_confusion3a2(2, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
        metricas['Dataset '+dataset]['grete']['tp|fn|tn|fp'] = str(tp)+'|'+str(fn)+'|'+str(tn)+'|'+str(fp)
        metricas['Dataset '+dataset]['grete']['accuracy'] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
        metricas['Dataset '+dataset]['grete']['precision'] = precision =  tp/(tp+fp) if tp+fp else 0
        metricas['Dataset '+dataset]['grete']['recall'] = recall = tp/(tp+fn) if tp+fn else 0
        metricas['Dataset '+dataset]['grete']['f'] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
        metricas['Dataset '+dataset]['grete']['fpr'] = fp/(fp+tn) if fp+tn else 0

        # calculamos as métricas de cada cascuda indiviual
        tp, fn, tn, fp = matriz_confusion3a2(3, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
        metricas['Dataset '+dataset]['samsa']['tp|fn|tn|fp'] = str(tp)+'|'+str(fn)+'|'+str(tn)+'|'+str(fp)
        metricas['Dataset '+dataset]['samsa']['accuracy'] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
        metricas['Dataset '+dataset]['samsa']['precision'] = precision =  tp/(tp+fp) if tp+fp else 0
        metricas['Dataset '+dataset]['samsa']['recall'] = recall = tp/(tp+fn) if tp+fn else 0
        metricas['Dataset '+dataset]['samsa']['f'] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
        metricas['Dataset '+dataset]['samsa']['fpr'] = fp/(fp+tn) if fp+tn else 0

        # MACRO AVERAGING #
        metricas['Dataset '+dataset]['total']['macro']['accuracy'] = medias_metricas_cascudas(dataset, metricas, 'accuracy')
        metricas['Dataset '+dataset]['total']['macro']['precision'] = medias_metricas_cascudas(dataset, metricas, 'precision')
        metricas['Dataset '+dataset]['total']['macro']['recall'] = medias_metricas_cascudas(dataset, metricas, 'recall')
        metricas['Dataset '+dataset]['total']['macro']['f'] = medias_metricas_cascudas(dataset, metricas, 'f')
        metricas['Dataset '+dataset]['total']['macro']['fpr'] = medias_metricas_cascudas(dataset, metricas, 'fpr')
        # MACRO AVERAGING #

        # MICRO AVERAGING #
        a_tp, a_fn, a_tn, a_fp = [int(ele) for ele in metricas['Dataset '+dataset]['gregor']['tp|fn|tn|fp'].split('|')]
        b_tp, b_fn, b_tn, b_fp = [int(ele) for ele in metricas['Dataset '+dataset]['grete']['tp|fn|tn|fp'].split('|')]
        c_tp, c_fn, c_tn, c_fp = [int(ele) for ele in metricas['Dataset '+dataset]['samsa']['tp|fn|tn|fp'].split('|')]

        tp = a_tp+b_tp+c_tp
        fn = a_fn+b_fn+c_fn
        tn = a_tn+b_tn+c_tn
        fp = a_fp+b_fp+c_fp

        metricas['Dataset '+dataset]['total']['micro']['tp|fn|tn|fp'] = str(tp)+'|'+str(fn)+'|'+str(tn)+'|'+str(fp)

        metricas['Dataset '+dataset]['total']['micro']['accuracy'] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
        metricas['Dataset '+dataset]['total']['micro']['precision'] = precision =  tp/(tp+fp) if tp+fp else 0
        metricas['Dataset '+dataset]['total']['micro']['recall'] = recall = tp/(tp+fn) if tp+fn else 0
        metricas['Dataset '+dataset]['total']['micro']['f'] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
        metricas['Dataset '+dataset]['total']['micro']['fpr'] = fp/(fp+tn) if fp+tn else 0
        # MICRO AVERAGING #

        # facemos o evaluate para extra datos
        avaliacion['Dataset '+dataset] = modelo.evaluate(test_ds[dataset], return_dict=True)

    ficheiro.gardarJson(FICHEIRO+'.prediccions', prediccions, indent=4)
    ficheiro.gardarJson(FICHEIRO+'.metricas', metricas, indent=4)
    ficheiro.gardarJson(FICHEIRO+'.cronoloxia', cronoloxia, indent=4)
    ficheiro.gardarJson(FICHEIRO+'.avaliacion', avaliacion,indent=4)

    print()
    print('* Documentos gardados baixo o nome: {} *'.format(NOMENCLATURA))
    print()

#------------------------------------------------------------------------------------------------
