import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

def predict(model,input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    predictions_score = predictions_df['prediction_score'][0]
    return predictions,predictions_score

st.title("Prediccion de estado de preñez")

add_selectbox = st.sidebar.selectbox(
    "¿Como desea añadir datos a predecir?",
    ("Carga Online","Desde Archivo"))

add_selectbox_2 = st.sidebar.selectbox("Elija Modelo Predictor",
    ("LDA","LR","RBFSVM"))

if add_selectbox_2 == "LDA":
    model = load_model("lda")
    st.sidebar.info("LDA info...")
if add_selectbox_2 == "LR":
    model = load_model("lr")
    st.sidebar.info("LR info...")
if add_selectbox_2 == "RBFSVM":
    model = load_model("rbfsvm")
    st.sidebar.info("RBFSVM info...")

st.sidebar.header("Link plantilla de carga de datos a predecir (descargar como .csv)")
st.sidebar.success("https://docs.google.com/spreadsheets/d/1n44gqgu6XjUDaVi4g51Ywhm3yRKPawT5I5I3DGB_4BM/edit?usp=sharing")

if add_selectbox == "Carga Online":
    #Fecha_Transferencia = st.date_input("Fecha de Transferencia")
    # Fecha_Produccion = st.date_input("Fecha de Produccion")
    # Fecha_Descongelamiento = st.date_input("Fecha de descongelamiento")
    # Linea_celular = st.text_input("Linea Celular")
    # Lente_objetivo = st.number_input('Lente objetivo', 0,100)
    # Magnificacion_optica = st.number_input('Magnificacion_optica', 0,1000)
    # Yegua_receptora = st.text_input("Yegua receptora")
    Tipo_celular = st.selectbox("Tipo Celular",["MSC-E","FB-E"])
    Dias_de_cultivo_celular = st.number_input('Dias de cultivo celular', 0,100,value=7 )
    Confluencia = st.number_input('% Confluencia', 0,100,value=100)
    Cantidad_de_Pasajes_bool = st.number_input('Cantidad de Pasajes bool', 0,1,value=0)
    Cantidad_de_Pasajes_numeric = st.number_input('Cantidad de Pasajes numeric', 0,100,value=1)
    Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas = st.number_input('Tiempo con bajo suero (DMEM 0,5% SFB) en horas', 0,1000,value=72)
    Origen = st.selectbox("Origen",["Local","Extranjero"])
    Maduracion = st.number_input('% Maduracion', 0.0,100.0,value=63.37)
    Calidad = st.selectbox("Calidad",["Sin Observaciones","Feos"])
    Clivaje = st.number_input('% Clivaje', 0.0,100.0,value=37.38)
    Medio_Placas_del_dia = st.selectbox("Medio Placas del dia",["F12","Global"])
    Dia_cambio_de_medio = st.number_input('Dia cambio de medio', 0,100,value=5)
    Dia_de_evolucion = st.selectbox("Dia de evolucion",["D7","D8","D9"])
    Grado_embrionario = st.selectbox("Grado embrionario",["I","II","III"])
    Fragmentacion_celular = st.selectbox("Fragmentacion celular",["Si","No"])
    Eje_menor = st.number_input('ø Eje menor', 0.0,1000.0,value=541.3)
    Eje_mayor = st.number_input('ø Eje mayor', 0.0,1000.0,value=564.6)
    Tipo_de_ovocito = st.selectbox("Tipo de ovocito",["Con ZP","Sin ZP"])
    Tipo_embrion = st.selectbox("Tipo de embrion",["Fresco","Vitrificado"])

    input_dict = {
        # 'Fecha Transferencia':Fecha_Transferencia, 'Fecha Producción':Fecha_Produccion, 'Fecha Descongelamiento':Fecha_Descongelamiento,
        # 'Línea celular':Linea_celular, 'Lente objetivo':Lente_objetivo, 'Magnificación óptica':Magnificacion_optica, 'Yegua receptora N°':Yegua_receptora
        'Tipo celular':Tipo_celular, 'Días de cultivo celular':Dias_de_cultivo_celular, '% Confluencia':Confluencia,
        'Cantidad de Pasajes bool':Cantidad_de_Pasajes_bool, 'Cantidad de Pasajes numeric':Cantidad_de_Pasajes_numeric,
        'Tiempo con bajo suero (DMEM 0,5% SFB) en horas':Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas, 'Origen':Origen, '% Maduración':Maduracion, 'Calidad':Calidad,
        '% Clivaje':Clivaje, 'Medio Placas del día':Medio_Placas_del_dia, 'Día cambio de medio':Dia_cambio_de_medio, 'Día de evolución':Dia_de_evolucion, 'Grado embrionario':Grado_embrionario,
        'Fragmentación celular':Fragmentacion_celular,  'ø Eje menor':Eje_menor, 'ø Eje mayor':Eje_mayor,
        'Tipo de ovocito':Tipo_de_ovocito, 'Tipo embrión':Tipo_embrion}

    input_df = pd.DataFrame([input_dict])
    st.dataframe(input_df)

    if st.button("Predict"):
        output = predict(model,input_df)

        st.success("Predict: "+output[0])
        st.success("Predict score: "+str(output[1]))

if add_selectbox == "Desde Archivo":
    file_upload = st.file_uploader("Cargar archivo csv para predecir",type =["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model, data = data)
        st.write(predictions)