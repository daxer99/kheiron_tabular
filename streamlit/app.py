import streamlit as st
import pandas as pd
import numpy as np

from pycaret.regression import load_model, predict_model

model = load_model("kheiron-pipeline_tabular")

def predict(model,input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions

from PIL import Image
image = Image.open("logo.png")
image_horse = Image.open("birra_01.jpg")

st.image(image,use_column_width=False)

add_selectbox = st.sidebar.selectbox(
    "¿Como desea añadir datos a predecir?",
    ("Carga Online","Desde Archivo"))

st.sidebar.info("Esta aplicacion esta diseñada para predecir la capacidad de preñez de un embrion")
st.sidebar.success("https://www.kheiron-biotech.com/")

st.sidebar.image(image_horse)

st.title("Prediccion de estado de preñez")

if add_selectbox == "Carga Online":
    Fecha_Transferencia = st.date_input("Fecha de Transferencia")
    Fecha_Produccion = st.date_input("Fecha de Produccion")
    Fecha_Descongelamiento = st.date_input("Fecha de descongelamiento")
    Linea_celular = st.text_input("Linea Celular")
    Tipo_celular = st.selectbox("Tipo Celular",["MSC-E","FB-E"])
    Dias_de_cultivo_celular = st.number_input('Dias de cultivo celular', 0,100)
    Confluencia = st.number_input('% Confluencia', 0,100)
    Cantidad_de_Pasajes_bool = st.number_input('Cantidad de Pasajes bool', 0,1)
    Cantidad_de_Pasajes_numeric = st.number_input('Cantidad de Pasajes numeric', 0,100)
    Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas = st.number_input('Tiempo con bajo suero (DMEM 0,5% SFB) en horas', 0,1000)
    Origen = st.selectbox("Origen",["Local","Extranjero"])
    Maduracion = st.number_input('% Maduracion', 0,100)
    Calidad = st.selectbox("Calidad",["Sin Observaciones","Feos"])
    Clivaje = st.number_input('% Clivaje', 0,100)
    Medio_Placas_del_dia = st.selectbox("Medio Placas del dia",["F12","Global"])
    Dia_cambio_de_medio = st.number_input('Dia cambio de medio', 0,100)
    Dia_de_evolucion = st.selectbox("Dia de evolucion",["D7","D8","D9"])
    Grado_embrionario = st.selectbox("Grado embrionario",["I","II","III"])
    Fragmentacion_celular = st.selectbox("Fragmentacion celular",["Si","No"])
    Lente_objetivo = st.number_input('Lente objetivo', 0,100)
    Magnificacion_optica = st.number_input('Magnificacion_optica', 0,1000)
    Eje_menor = st.number_input('ø Eje menor', 0,1000)
    Eje_mayor = st.number_input('ø Eje mayor', 0,1000)
    Tipo_de_ovocito = st.selectbox("Tipo de ovocito",["Con ZP","Sin ZP"])
    Tipo_embrion = st.selectbox("Tipoembrion",["Fresco","Vitrificado"])
    Yegua_receptora = st.text_input("Yegua receptora")

    input_dict = {'Fecha Transferencia':Fecha_Transferencia, 'Fecha Producción':Fecha_Produccion, 'Fecha Descongelamiento':Fecha_Descongelamiento,
        'Línea celular':Linea_celular, 'Tipo celular':Tipo_celular, 'Días de cultivo celular':Dias_de_cultivo_celular, '% Confluencia':Confluencia,
        'Cantidad de Pasajes bool':Cantidad_de_Pasajes_bool, 'Cantidad de Pasajes numeric':Cantidad_de_Pasajes_numeric,
        'Tiempo con bajo suero (DMEM 0,5% SFB) en horas':Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas, 'Origen':Origen, '% Maduración':Maduracion, 'Calidad':Calidad,
        '% Clivaje':Clivaje, 'Medio Placas del día':Medio_Placas_del_dia, 'Día cambio de medio':Dia_cambio_de_medio, 'Día de evolución':Dia_de_evolucion, 'Grado embrionario':Grado_embrionario,
        'Fragmentación celular':Fragmentacion_celular, 'Lente objetivo':Lente_objetivo, 'Magnificación óptica':Magnificacion_optica, 'ø Eje menor':Eje_menor, 'ø Eje mayor':Eje_mayor,
        'Tipo de ovocito':Tipo_de_ovocito, 'Tipo embrión':Tipo_embrion, 'Yegua receptora N°':Yegua_receptora}

    input_df = pd.DataFrame([input_dict])
    st.dataframe(input_df)

    if st.button("Predict"):
        output = predict(model,input_df)

        st.success("Predict: "+output)

if add_selectbox == "Desde Archivo":
    file_upload = st.file_uploader("Cargar archivo csv para predecir",type =["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model, data = data)
        st.write(predictions)