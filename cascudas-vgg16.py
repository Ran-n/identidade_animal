#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------------
#+ Autor:	Ran#
#+ Creado:	05/06/2021 17:06:17
#+ Editado:	17/06/2021 00:17:17
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
from tensorflow.keras.applications.vgg16 import VGG16

#------------------------------------------------------------------------------------------------

def avaliar(dataset, cascuda, metricas, tp, fn, tn, fp):
    
    metricas["Dataset_"+dataset][cascuda]["tp|fn|tn|fp"] = str(tp)+"|"+str(fn)+"|"+str(tn)+"|"+str(fp)
    metricas["Dataset_"+dataset][cascuda]["accuracy"] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
    metricas["Dataset_"+dataset][cascuda]["precision"] = precision =  tp/(tp+fp) if tp+fp else 0
    metricas["Dataset_"+dataset][cascuda]["recall"] = recall = tp/(tp+fn) if tp+fn else 0
    metricas["Dataset_"+dataset][cascuda]["f"] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
    metricas["Dataset_"+dataset][cascuda]["fpr"] = fp/(fp+tn) if fp+tn else 0

    return metricas

#------------------------------------------------------------------------------------------------
def avaliar2(dataset, metricas, tp, fn, tn, fp):
    
    metricas["Dataset_"+dataset]["total"]["micro"]["tp|fn|tn|fp"] = str(tp)+"|"+str(fn)+"|"+str(tn)+"|"+str(fp)
    metricas["Dataset_"+dataset]["total"]["micro"]["accuracy"] = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0
    metricas["Dataset_"+dataset]["total"]["micro"]["precision"] = precision =  tp/(tp+fp) if tp+fp else 0
    metricas["Dataset_"+dataset]["total"]["micro"]["recall"] = recall = tp/(tp+fn) if tp+fn else 0
    metricas["Dataset_"+dataset]["total"]["micro"]["f"] = 2*((precision*recall)/(precision+recall)) if precision+recall else 0
    metricas["Dataset_"+dataset]["total"]["micro"]["fpr"] = fp/(fp+tn) if fp+tn else 0

    return metricas

#------------------------------------------------------------------------------------------------
def matriz_confusion3a2(posicion_positivo, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3):
    """
        posicion_positivo --> clase A > 1 ; clase B > 2 ; clase C > 3
    """

    if posicion_positivo == 1:
        # tp fn tn fp
        return f1c1, f1c2+f1c3, f2c2+f2c3+f3c2+f3c3, f2c1+f3c1

    elif posicion_positivo == 2:
        # tp fn tn fp
        return f2c2, f2c3+f2c1, f3c3+f3c1+f1c3+f1c1, f3c2+f1c2

    elif posicion_positivo == 3:
        # tp fn tn fp
        return f3c3, f3c1+f3c2, f1c1+f1c2+f2c1+f2c2, f1c3+f2c3

    else:
        raise Exception("Non existe esa posición nunha matriz 3x3")
#------------------------------------------------------------------------------------------------
def medias_metricas_cascudas(dataset, metricas, metrica):
    a = metricas["Dataset_"+dataset]["gregor"][metrica]
    b = metricas["Dataset_"+dataset]["grete"][metrica]
    c = metricas["Dataset_"+dataset]["samsa"][metrica]

    return (a+b+c)/3

#------------------------------------------------------------------------------------------------

SEMENTE = 1
ALTURA_IMAXE = ANCHURA_IMAXE = 224
CANT_VALIDACION = 0.2
BATCH_SIZE = 256
EPOCHS = 15
PASOS = 100
NUM_CLASES = 3
specs = 'epochs;'+str(EPOCHS)+'_batch-size;'+str(BATCH_SIZE)

NUMERO = "1"
ALGORITMO = "vgg16"
CARPETA = "saida-cascudas-transfer-learning/"+"cascudas-"+ALGORITMO+"_"+NUMERO+"_"+specs
NOME_FICH = CARPETA+"cascudas-"+ALGORITMO+"_"+NUMERO+"_"+specs

if not os.path.exists(CARPETA):
    os.makedirs(CARPETA)


ligazon = "https://bucketfg.blob.core.windows.net/datasets/Dataset_Cascudas_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE)+".tar.gz"

#------------------------------------------------------------------------------------------------

data_dir = tf.keras.utils.get_file("Dataset_Cascudas_"+str(ALTURA_IMAXE)+"x"+str(ANCHURA_IMAXE), origin=ligazon, untar=True)
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

#------------------------------------------------------------------------------------------------

base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
base_model.summary()

for layer in base_model.layers:
    layer.trainable = False


x = layers.Flatten(name = "flatten")(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(NUM_CLASES)(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

#------------------------------------------------------------------------------------------------

#model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#history = model.fit(train_ds, validation_data = val_ds, steps_per_epoch = PASOS, epochs = EPOCHS)
history = model.fit(train_ds, validation_data = val_ds, epochs = EPOCHS)

#------------------------------------------------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

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
#------------------------------------------------------------------------------------------------
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
        "gregor": {"acertos": 0, "erros": {"grete": 0, "samsa": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "grete": {"acertos": 0, "erros": {"samsa": 0, "gregor": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "samsa": {"acertos": 0, "erros": {"gregor": 0, "grete": 0}, "historia": {"cronoloxia": [], "confianza": []}}
        },
    "Dataset_64":
        {
        "gregor": {"acertos": 0, "erros": {"grete": 0, "samsa": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "grete": {"acertos": 0, "erros": {"samsa": 0, "gregor": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "samsa": {"acertos": 0, "erros": {"gregor": 0, "grete": 0}, "historia": {"cronoloxia": [], "confianza": []}}
        },
    "Dataset_128":
        {
        "gregor": {"acertos": 0, "erros": {"grete": 0, "samsa": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "grete": {"acertos": 0, "erros": {"samsa": 0, "gregor": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "samsa": {"acertos": 0, "erros": {"gregor": 0, "grete": 0}, "historia": {"cronoloxia": [], "confianza": []}}
        },
    "Dataset_224":
        {
        "gregor": {"acertos": 0, "erros": {"grete": 0, "samsa": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "grete": {"acertos": 0, "erros": {"samsa": 0, "gregor": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "samsa": {"acertos": 0, "erros": {"gregor": 0, "grete": 0}, "historia": {"cronoloxia": [], "confianza": []}}
        },
    "Dataset_256":
        {
        "gregor": {"acertos": 0, "erros": {"grete": 0, "samsa": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "grete": {"acertos": 0, "erros": {"samsa": 0, "gregor": 0}, "historia": {"cronoloxia": [], "confianza": []}},
        "samsa": {"acertos": 0, "erros": {"gregor": 0, "grete": 0}, "historia": {"cronoloxia": [], "confianza": []}}
        }
}

metricas = {
    "Dataset_32": {
        "gregor": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "grete": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "samsa": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "total": {"f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3": "", "macro": {
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }, "micro": {
                                                                            "tp|fn|tn|fp": "",
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }}
    },
    "Dataset_64": {
        "gregor": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "grete": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "samsa": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "total": {"f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3": "", "macro": {
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }, "micro": {
                                                                            "tp|fn|tn|fp": "",
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }}
    },
    "Dataset_128": {
        "gregor": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "grete": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "samsa": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "total": {"f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3": "", "macro": {
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }, "micro": {
                                                                            "tp|fn|tn|fp": "",
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }}
    },
    "Dataset_224": {
        "gregor": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "grete": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "samsa": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "total": {"f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3": "", "macro": {
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }, "micro": {
                                                                            "tp|fn|tn|fp": "",
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }}
    },
    "Dataset_256": {
        "gregor": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "grete": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "samsa": {"tp|fn|tn|fp": "", "accuracy": 0, "precision": 0, "recall": 0, "f": 0, "fpr": 0},
        "total": {"f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3": "", "macro": {
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }, "micro": {
                                                                            "tp|fn|tn|fp": "",
                                                                            "accuracy": 0,
                                                                            "precision": 0,
                                                                            "recall": 0,
                                                                            "f": 0,
                                                                            "fpr": 0
                                                                    }}
    }
}

for dataset in ["32", "64", "128", "224", "256"]:
    for cascuda in ["gregor", "grete", "samsa"]:
        for imaxe in range(0, 100, 1):
            url = "https://bucketfg.blob.core.windows.net/identiset-cascudas-"+dataset+"-"+cascuda+"/"+dataset+"_"+cascuda+"_"+str(imaxe)+".jpg"
            path = tf.keras.utils.get_file(dataset+"_"+cascuda+"_"+str(imaxe), origin=url)

            imx = keras.preprocessing.image.load_img(
                path, target_size=(ALTURA_IMAXE, ANCHURA_IMAXE)
            )
            
            imaxe_array = keras.preprocessing.image.img_to_array(imx)
            imaxe_array = tf.expand_dims(imaxe_array, 0)

            prediccion = model.predict(imaxe_array)
            puntaxe = tf.nn.softmax(prediccion[0])

            clase_predita = nome_clases[np.argmax(puntaxe)]
            puntaxe_predita = np.max(puntaxe)

            acertou = int(clase_predita == cascuda)

            if acertou:
                prediccions["Dataset_"+dataset][cascuda]["acertos"] += 1
            else:
                prediccions["Dataset_"+dataset][cascuda]["erros"][clase_predita] += 1

            # acertou > 0/1 ; cascuda > a clase que era ; clase_predita > o que prediciu o model
            prediccions["Dataset_"+dataset][cascuda]["historia"]["cronoloxia"].append([acertou, cascuda, clase_predita])
            prediccions["Dataset_"+dataset][cascuda]["historia"]["confianza"].append(str(puntaxe_predita))

    # crear a matriz de confusión 3x3
    f1c1 = prediccions["Dataset_"+dataset]["gregor"]["acertos"]
    f1c2 = prediccions["Dataset_"+dataset]["gregor"]["erros"]["grete"]
    f1c3 = prediccions["Dataset_"+dataset]["gregor"]["erros"]["samsa"]
    f2c1 = prediccions["Dataset_"+dataset]["grete"]["erros"]["gregor"]
    f2c2 = prediccions["Dataset_"+dataset]["grete"]["acertos"]
    f2c3 = prediccions["Dataset_"+dataset]["grete"]["erros"]["samsa"]
    f3c1 = prediccions["Dataset_"+dataset]["samsa"]["erros"]["gregor"]
    f3c2 = prediccions["Dataset_"+dataset]["samsa"]["erros"]["grete"]
    f3c3 = prediccions["Dataset_"+dataset]["samsa"]["acertos"]

    # gardamos a matriz de confusión 3x3
    # matriz_confusion_3x3 = str(f1c1)+"|"+str(f1c2)+"|"+str(f1c3)+"|"+str(f2c1)+"|"+str(f2c2)+"|"+str(f2c3)+"|"+str(f3c1)+"|"+str(f3c2)+"|"+str(f3c3)
    metricas["Dataset_"+dataset]["total"]["f1c1|f1c2|f1c3|f2c1|f2c2|f2c3|f3c1|f3c2|f3c3"] = str(f1c1)+"|"+str(f1c2)+"|"+str(f1c3)+"|"+str(f2c1)+"|"+str(f2c2)+"|"+str(f2c3)+"|"+str(f3c1)+"|"+str(f3c2)+"|"+str(f3c3)

    # calculamos as métricas de cada cascuda indiviual
    tp, fn, tn, fp = matriz_confusion3a2(1, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
    # avaliar(dataset, cascuda, metricas, tp, fn, tn, fp)
    metricas = avaliar(dataset, "gregor", metricas, tp, fn, tn, fp)

    # calculamos as métricas de cada cascuda indiviual
    tp, fn, tn, fp = matriz_confusion3a2(2, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
    metricas = avaliar(dataset, "grete", metricas, tp, fn, tn, fp)

    # calculamos as métricas de cada cascuda indiviual
    tp, fn, tn, fp = matriz_confusion3a2(3, f1c1, f1c2, f1c3, f2c1, f2c2, f2c3, f3c1, f3c2, f3c3)
    metricas = avaliar(dataset, "samsa", metricas, tp, fn, tn, fp)

    # metricas facendo o macro averaging
    metricas["Dataset_"+dataset]["total"]["macro"]["accuracy"] = medias_metricas_cascudas(dataset, metricas, "accuracy")
    metricas["Dataset_"+dataset]["total"]["macro"]["precision"] = medias_metricas_cascudas(dataset, metricas, "precision")
    metricas["Dataset_"+dataset]["total"]["macro"]["recall"] = medias_metricas_cascudas(dataset, metricas, "recall")
    metricas["Dataset_"+dataset]["total"]["macro"]["f"] = medias_metricas_cascudas(dataset, metricas, "f")
    metricas["Dataset_"+dataset]["total"]["macro"]["fpr"] = medias_metricas_cascudas(dataset, metricas, "fpr")

    # metricas facendo o micro averaging

    a_tp, a_fn, a_tn, a_fp = metricas["Dataset_"+dataset]["gregor"]["tp|fn|tn|fp"].split("|")
    b_tp, b_fn, b_tn, b_fp = metricas["Dataset_"+dataset]["grete"]["tp|fn|tn|fp"].split("|")
    c_tp, c_fn, c_tn, c_fp = metricas["Dataset_"+dataset]["samsa"]["tp|fn|tn|fp"].split("|")

    tp = int(a_tp)+int(b_tp)+int(c_tp)
    fn = int(a_fn)+int(b_fn)+int(c_fn)
    tn = int(a_tn)+int(b_tn)+int(c_tn)
    fp = int(a_fp)+int(b_fp)+int(c_fp)

    metricas = avaliar2(dataset, metricas, tp, fn, tn, fp)

    """
    metricas["Dataset_"+dataset]["total"]["micro"]["tp|fn|tn|fp"] = avaliar2(dataset, metricas, tp, fn, tn, fp)
    metricas["Dataset_"+dataset]["total"]["micro"]["accuracy"] = 
    metricas["Dataset_"+dataset]["total"]["micro"]["precision"] = 
    metricas["Dataset_"+dataset]["total"]["micro"]["recall"] = 
    metricas["Dataset_"+dataset]["total"]["micro"]["f"] = 
    metricas["Dataset_"+dataset]["total"]["micro"]["fpr"] = 
    """
    """
    tp = prediccions["Dataset_"+dataset]["gonzales"]["acertos"]
    fn = prediccions["Dataset_"+dataset]["gonzales"]["erros"]
    tn = prediccions["Dataset_"+dataset]["speedy"]["acertos"]
    fp = prediccions["Dataset_"+dataset]["speedy"]["erros"]

    metricas = avaliar(FICHEIRO, dataset, metricas, tp, fn, tn, fp)
    """
open(NOME_FICH+".prediccions", 'w').write(json.dumps(prediccions, indent=1, sort_keys=False, ensure_ascii=False))
open(NOME_FICH+".metricas", 'w').write(json.dumps(metricas, indent=1, sort_keys=False, ensure_ascii=False))

#------------------------------------------------------------------------------------------------