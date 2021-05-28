#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------
#+ Autor:   Ran#
#+ Creado:  24/03/2021 19:04:19
#+ Editado: 04/04/2021 21:42:38
# ----------------------------------------------

import sys

# meto este bloque aquí e non despois para non ter que cargar sempre tensorflow se pide axuda e así ter unha resposta rápida
# miramos se ten entradas por comando e se as ten os valores deben ser postos a tal
# quitamos a primeira
args = sys.argv[1:]

# mensaxe de axuda
if ('-a' in args) or('-h' in args) or (len(args) == 0):
    print("\nAxuda -----------")

    print("-h/-a\t\t-> Para esta mensaxe")
    print()
    print("-m\t\t-> Tan só facer o modelado\t\t\t(ambas)")
    print("-p catex\t-> Facer as prediccións dun modelo previo\t(ambas)")
    print()
    print("-d num\t\t-> dimensións\t\t\t\t\t(32x32)")
    print("-e num\t\t-> epochs\t\t\t\t\t(50)")
    print("-b num\t\t-> batch size\t\t\t\t\t(32)")
    print("-c num\t\t-> número de clases\t\t\t\t(2)")
    print("-v num\t\t-> % validación entre 0 e 1\t\t\t(0.2)")
    print("-s num\t\t-> semente\t\t\t\t\t(1)")
    
    print("----------------\n")
    
    if len(args) != 0:
        sys.exit()


import matplotlib.pyplot as plt
import numpy as np
import secrets
import json
import os
# eliminar os warnings de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import PIL
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ----------------------------------------------
# CONFIGURACION
# ----------------------------------------------

if ('-m' in args) or ('-p' in args):
    #
    if '-m' in args:
        TIPO = 'modelar'
        FICHEIRO = None

    #
    if '-p' in args:
        FICHEIRO = args[args.index('-p')+1]
        TIPO = 'predicir'

else:
    TIPO = "ambas"
    FICHEIRO = None

# altura da imaxe
if '-d' in args:
    DIMENSIONS = args[args.index('-d')+1]
    if 'x' in DIMENSIONS:
        ALTURA_IMAXE, ANCHURA_IMAXE = [int(dimension) for dimension in DIMENSIONS.split('x')]
    else: 
        ALTURA_IMAXE = ANCHURA_IMAXE = int(DIMENSIONS)
else:
    # 128, 64 ou 32
    ALTURA_IMAXE = ANCHURA_IMAXE = 32
    DIMENSIONS = "32x32"

# epochs que se realizan
if '-e' in args:
    EPOCHS = int(args[args.index('-e')+1])
else:
    EPOCHS=50

# tamaí±o da batch
if '-b' in args:
    BATCH_SIZE = int(args[args.index('-b')+1])
else:
    BATCH_SIZE = 32

# número de clases que ten
if '-c' in args:
    CANT_CLASES = int(args[args.index('-c')+1])
else:
    CANT_CLASES = 2

# porcentaxe a usar para a validación
if '-v' in args:
    CANT_VALIDACION = float(args[args.index('-v')+1])
    # se nos meten algo que non esté no rango kk
    if CANT_VALIDACION > 1 or CANT_VALIDACION < 0:
        CANT_VALIDACION = 0.2
else:
    CANT_VALIDACION = 0.2

# semente a usar
if '-s' in args:
    SEMENTE = int(args[args.index('-s')+1])
else:
    SEMENTE = 1

# mostra por pantalla a configuración a usar
print('\nConfiguración a usar:')
print('----------------------------------')
dimensions = str(ALTURA_IMAXE)+'x'+str(ANCHURA_IMAXE)
print('dimensións:\t', dimensions, '\nepochs:\t\t', EPOCHS, '\nbatch size:\t', BATCH_SIZE, '\ncant clases:\t', CANT_CLASES, '\n% validación:\t', CANT_VALIDACION, '\nsemente:\t', SEMENTE)
print('----------------------------------')

# como se van chamar os arquivos de saí­da
CARPETA = "saida/"
if not os.path.exists(CARPETA):
    os.makedirs(CARPETA)

nomenclatura = 'dataset;'+DIMENSIONS+'_epochs;'+str(EPOCHS)+'_batch-size;'+str(BATCH_SIZE)+'___'+str(secrets.token_hex(4))
ruta = CARPETA+nomenclatura

# ----------------------------------------------

# optouse por usar un url de orixe en lugar dunha ruta para que se poda usar facilmente en calquer ordenador
lig_cdd = "https://bucketfg.blob.core.windows.net/datasets/Dataset_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE)+".tar.gz"

# ----------------------------------------------
# MODELADO
# ----------------------------------------------

# colle a ligazón e de ahí­ baixa o cdd
data_dir = tf.keras.utils.get_file("Dataset_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE), origin=lig_cdd, untar=True)
data_dir = pathlib.Path(data_dir)

train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=CANT_VALIDACION,
        subset="training",
        seed=SEMENTE,
        image_size=(ALTURA_IMAXE, ANCHURA_IMAXE),
        batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=CANT_VALIDACION,
        subset="validation",
        seed=SEMENTE,
        image_size=(ALTURA_IMAXE, ANCHURA_IMAXE),
        batch_size=BATCH_SIZE
)

nome_clases = train_ds.class_names

# todo este bloque e para cargar o dataset de buffer e non de memoria para evitar pescozos de botella
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------------------------

# creamos a rede
model = Sequential([
    # as cores van do 0 ao 255, con isto fai que vaian do 0 ó 1
    # faise en todos pero tamén nestas en branco e negro polos todos de grises
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(ALTURA_IMAXE, ANCHURA_IMAXE, 3)),

    # bloque 1
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # bloque 2
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # bloque3
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # cabeza
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(CANT_CLASES)
])

#"""
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.1),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
#"""

# información sobre a rede
model.summary()

# facemos o fit
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

#"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
"""
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
#"""

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamento')
plt.plot(epochs_range, val_acc, label='Validación')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.title('Precisión de entrenamento e validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamento')
plt.plot(epochs_range, val_loss, label='Validación')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

plt.title('Perda de entrenamento e validación')
#plt.show()

#gardamos a gráfica
plt.savefig(ruta)
#gardamos o modelo
model.save(ruta)

# ----------------------------------------------
# PREDICCIÓN
# ----------------------------------------------

# diccionario onde se gardan as prediccións
prediccions = {
"Dataset_32":
    {
    "gonzales": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}},
    "speedy": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}}
    },
"Dataset_64":
    {
    "gonzales": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}},
    "speedy": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}}
    },
"Dataset_128":
    {
    "gonzales": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}},
    "speedy": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}}
    }
}

# diccionario onde se gardan as métricas
metricas = {
"Dataset_32": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
"Dataset_64": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
"Dataset_128": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0}
}

# imos recorrendo tódolos identisets para ver como reacciona o modelo
for dataset in ["32", "64", "128"]:
    for rata in ["gonzales", "speedy"]:
        for imaxe in range(4900, 5000, 1):
            url = "https://bucketfg.blob.core.windows.net/identiset-"+dataset+"-"+rata+"/"+dataset+"_"+rata+"_"+str(imaxe)+".jpg"
            path = tf.keras.utils.get_file(dataset+"_"+rata+"_"+str(imaxe), origin=url)

            imx = keras.preprocessing.image.load_img(
                path, target_size=(ALTURA_IMAXE, ANCHURA_IMAXE)
            )
            
            imaxe_array = keras.preprocessing.image.img_to_array(imx)
            imaxe_array = tf.expand_dims(imaxe_array, 0)

            prediccion = modelo.predict(imaxe_array)
            puntaxe = tf.nn.softmax(prediccion[0])

            clase_predita = nome_clases[np.argmax(puntaxe)]
            puntaxe_predita = np.max(puntaxe)

            acertou = int(clase_predita == rata)

            if acertou:
                prediccions["Dataset_"+dataset][rata]["acertos"] += 1
                #print("A imaxe é {} cun {:.4f}% confidence. Acertou :)".format(clase_predita, 100 * puntaxe_predita))
            else:
                prediccions["Dataset_"+dataset][rata]["erros"] += 1
                #print("A imaxe é {} cun {:.4f}% confidence. Errou :(".format(clase_predita, 100 * puntaxe_predita))

            prediccions["Dataset_"+dataset][rata]["historia"]["cronoloxia"].append(acertou)
            prediccions["Dataset_"+dataset][rata]["historia"]["confianza"].append(str(puntaxe_predita))

    tp = prediccions["Dataset_"+dataset]["gonzales"]["acertos"]
    fn = prediccions["Dataset_"+dataset]["gonzales"]["erros"]
    tn = prediccions["Dataset_"+dataset]["speedy"]["acertos"]
    fp = prediccions["Dataset_"+dataset]["speedy"]["erros"]

    # calculamos as métricas
    metricas["Dataset_"+dataset]["accuracy"] = (tp+tn)/(tp+tn+fp+fn)
    metricas["Dataset_"+dataset]["precision"] = precision =  tp/(tp+fp)
    metricas["Dataset_"+dataset]["recall"] = recall = tp/(tp+fn)
    metricas["Dataset_"+dataset]["f"] = 2*((precision*recall)/(precision+recall))
    metricas["Dataset_"+dataset]["fpr"] = fp/(fp+tn)


# gardamos tódolos resultados
open(ruta+".prediccions", 'w').write(json.dumps(prediccions, indent=1, sort_keys=False, ensure_ascii=False))
open(ruta+".metricas", 'w').write(json.dumps(metricas, indent=1, sort_keys=False, ensure_ascii=False))

print('----------------------------------')
print('Datos gardados na ruta: ',ruta)
print('----------------------------------')

# ----------------------------------------------