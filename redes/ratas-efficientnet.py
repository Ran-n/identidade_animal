#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------
#+ Autor:   Ran#
#+ Creado:  21/04/2021 20:18:27
#+ Editado:	21/04/2021 20:22:06
#------------------------------------------------------------------------------------------------
import os
# eliminar os warnings de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import numpy as np
import sys
import json

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Dropout
import efficientnet.keras as efn

#------------------------------------------------------------------------------------------------
SEMENTE = 1
ALTURA_IMAXE = ANCHURA_IMAXE = 224
CANT_VALIDACION = 0.2
BATCH_SIZE = 32
EPOCHS = 15
PASOS = 100

NUMERO = "1"
ALGORITMO = "efficientnet"
CARPETA = "saida_/"+ALGORITMO
NOME_FICH = "ratas-"+ALGORITMO+"_"+NUMERO

if not os.path.exists(CARPETA):
    os.makedirs(CARPETA)

ligazon = "https://bucketfg.blob.core.windows.net/datasets/Dataset_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE)+".tar.gz"
#------------------------------------------------------------------------------------------------

data_dir = tf.keras.utils.get_file("Dataset_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE), origin=ligazon, untar=True)
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

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
model.summary()

for layer in model.layers:
    layer.trainable = False


x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)
model = Model(input = base_model.input, output = predictions)

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='binary_crossentropy',metrics=['accuracy'])


#history = model.fit(train_ds, validation_data = val_ds, steps_per_epoch = PASOS, epochs = EPOCHS)
history = model.fit_generator(train_ds, validation_data = val_ds, epochs = EPOCHS)


acc = history.history['acc']
val_acc = history.history['val_acc']

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

plt.savefig(NOME_FICH)
model.save(NOME_FICH)

ficheiro = open(NOME_FICH+".clases", 'w')
for cousas in nome_clases:
    ficheiro.writelines(cousas+'\n')
ficheiro.close()
open(NOME_FICH+".historia", 'w').write(json.dumps(history.history, indent=1, sort_keys=False, ensure_ascii=False))

#------------------------------------------------------------------------------------------------

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
    },
"Dataset_224":
    {
    "gonzales": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}},
    "speedy": {"acertos": 0, "erros": 0, "historia": {"cronoloxia": [], "confianza": []}}
    }
}

metricas = {
"Dataset_32": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
"Dataset_64": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
"Dataset_128": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
"Dataset_224": {"accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0}
}

for dataset in ["32", "64", "128", "224"]:
    for rata in ["gonzales", "speedy"]:
        for imaxe in range(4900, 5000, 1):
            url = "https://bucketfg.blob.core.windows.net/identiset-"+dataset+"-"+rata+"/"+dataset+"_"+rata+"_"+str(imaxe)+".jpg"
            path = tf.keras.utils.get_file(dataset+"_"+rata+"_"+str(imaxe), origin=url)

            imx = keras.preprocessing.image.load_img(
                path, target_size=(ALTURA_IMAXE, ANCHURA_IMAXE)
            )

            imaxe_array = keras.preprocessing.image.img_to_array(imx)
            imaxe_array = tf.expand_dims(imaxe_array, 0)

            prediccion = model.predict(imaxe_array)
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

    metricas["Dataset_"+dataset]["accuracy"] = (tp+tn)/(tp+tn+fp+fn)
    metricas["Dataset_"+dataset]["precision"] = precision =  tp/(tp+fp)
    metricas["Dataset_"+dataset]["recall"] = recall = tp/(tp+fn)
    metricas["Dataset_"+dataset]["f"] = 2*((precision*recall)/(precision+recall))
    metricas["Dataset_"+dataset]["fpr"] = fp/(fp+tn)

open(NOME_FICH+".prediccions", 'w').write(json.dumps(prediccions, indent=1, sort_keys=False, ensure_ascii=False))
open(NOME_FICH+".metricas", 'w').write(json.dumps(metricas, indent=1, sort_keys=False, ensure_ascii=False))

#------------------------------------------------------------------------------------------------