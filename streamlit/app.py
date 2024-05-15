import streamlit as st
import pandas as pd
import numpy as np

from pycaret.regression import load_model, predict_model

model = load_model("/media/rodrigo/Data1/Kheiron/kheiron_tabular/kheiron-pipeline_tabular")

def predict(model,input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open("logo.png")
    image_horse = Image.open("birra_01.jpg")

    st.image(image,use_column_width=False)

    # add_selectbox = st.sidebar.selectbox(
    #     "¿Como desea añadir datos a predecir?",
    #     ("Carga Online","Desde Archivo"))
    #
    # st.sidebar.info("Esta aplicacion esta diseñada para predecir...")
    # st.sidebar.success("https://www.pycaret.org")
    #
    # st.sidebar.image(image_horse)
    #
    # st.title("Prediccion de estado de preñez")
    #
    # if add_selectbox == "Carga Online":
    #     Fecha_Transferencia = 0
    #     Fecha_Produccion = 0
    #     Fecha_Descongelamiento = 0
    #     Linea_celular = 0
    #     Tipo_celular = 0
    #     Dias_de_cultivo_celular = 0
    #     Confluencia = 0
    #     Cantidad_de_Pasajes_bool = 0
    #     Cantidad_de_Pasajes_numeric = 0
    #     Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas = 0
    #     Origen = 0
    #     Maduracion = 0
    #     Calidad = 0
    #     Clivaje =0
    #     Medio_Placas_del_dia = 0
    #     Dia_cambio_de_medio = 0
    #     Dia_de_evolucion = 0
    #     Grado_embrionario = 0
    #     Fragmentacion_celular = 0
    #     Lente_objetivo = 0
    #     Magnificacion_optica = 0
    #     Eje_menor = 0
    #     Eje_mayor = 0
    #     Tipo_de_ovocito = 0
    #     Tipo_embrion = 0
    #     Yegua_receptora = 0
    #
    #     input_dict = {'Fecha Transferencia':Fecha_Transferencia, 'Fecha Producción':Fecha_Produccion, 'Fecha Descongelamiento':Fecha_Descongelamiento,
    #     'Línea celular':Linea_celular, 'Tipo celular':Tipo_celular, 'Días de cultivo celular':Dias_de_cultivo_celular, '% Confluencia':Confluencia,
    #     'Cantidad de Pasajes bool':Cantidad_de_Pasajes_bool, 'Cantidad de Pasajes numeric':Cantidad_de_Pasajes_numeric,
    #     'Tiempo con bajo suero (DMEM 0,5% SFB) en horas':Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas, 'Origen':Origen, '% Maduración':Maduracion, 'Calidad':Calidad,
    #     '% Clivaje':Clivaje, 'Medio Placas del día':Medio_Placas_del_dia, 'Día cambio de medio':Dia_cambio_de_medio, 'Día de evolución':Dia_de_evolucion, 'Grado embrionario':Grado_embrionario,
    #     'Fragmentación celular':Fragmentacion_celular, 'Lente objetivo':Lente_objetivo, 'Magnificación óptica':Magnificacion_optica, 'ø Eje menor':Eje_menor, 'ø Eje mayor':Eje_mayor,
    #     'Tipo de ovocito':Tipo_de_ovocito, 'Tipo embrión':Tipo_embrion, 'Yegua receptora N°':Yegua_receptora}
    #
    #     input_df = pd.DataFrame([input_dict])
    #
    #     if st.button("Predict"):
    #         output = predict(model,input_df)
    #
    #     st.success("Predict: "+output)
    #
    # if add_selectbox == "Desde Archivo":
    #     file_upload = st.file_uploader("Cargar archivo csv para predecir",type =["csv"])
    #
    #     if file_upload is not None:
    #         data = pd.read_csv(file_upload)
    #         predictions = predict_model(estimator=model, data = data)
    #         st.write(predictions)