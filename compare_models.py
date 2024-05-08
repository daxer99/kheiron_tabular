import pandas as pd
import shutil
import pycaret
import matplotlib.pyplot as plt
from pycaret.classification import *

dataset = pd.read_csv('data.csv', sep=',')
# header = list(dataset.columns)
# print(header)
numeric_features = ['Días de cultivo celular', '% Confluencia', 'Cantidad de Pasajes bool', 'Cantidad de Pasajes numeric', 'Tiempo con bajo suero (DMEM 0,5% SFB) en horas','% Maduración','% Clivaje','Día cambio de medio','ø Eje menor', 'ø Eje mayor',]
categorical_features = ['Tipo celular', 'Origen','Calidad','Medio Placas del día','Día de evolución', 'Grado embrionario','Fragmentación celular','Tipo de ovocito', 'Tipo embrión']
target = 'Preñez'
experiment_name = 'kheiron'
clf1 = setup(data=dataset,
             target = target,
             log_experiment = True,
             experiment_name = experiment_name,
             ignore_features=['Fecha Transferencia', 'Fecha Producción', 'Fecha Descongelamiento','Línea celular','Lente objetivo', 'Magnificación óptica', 'Yegua receptora N°'],
             numeric_features = numeric_features,
             categorical_features = categorical_features,
             train_size = 0.75,
             normalize = False,
             normalize_method = 'zscore',#'minmax', 'maxabs', 'robust'
             pca = False
            )

# model_name = models().index.tolist()
model_name = ['lightgbm', 'catboost', 'dummy']
# print(model_name)
for i in range(len(model_name)):
    model = create_model(model_name[i])
    confusion_matrix = plot_model(model, plot='confusion_matrix', save=True)
    auc = plot_model(model, plot='auc', save=True)
    shutil.copyfile(confusion_matrix, "/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/" + model_name[i]+"_confusion_matrix.png")
    shutil.copyfile(auc, "/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/" + model_name[i]+"_auc.png")
    metrics = pull()
    metrics.to_csv("/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/"+model_name[i]+"_metrics.csv")

    model_tuned = tune_model(model, n_iter = 100)
    confusion_matrix = plot_model(model_tuned, plot='confusion_matrix', save=True)
    auc = plot_model(model_tuned, plot='auc', save=True)
    shutil.copyfile(confusion_matrix, "/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/" + model_name[i]+"_tuned"+"_confusion_matrix.png")
    shutil.copyfile(auc, "/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/" + model_name[i]+"_tuned"+"_auc.png")
    metrics = pull()
    metrics.to_csv("/media/rodrigo/Data1/Kheiron/kheiron_tabular/compare_models/"+ model_name[i]+"_tuned"+"_metrics.csv")

    print("Finished: ",model_name[i])