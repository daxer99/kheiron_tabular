import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

def convertir_cantidad_pasajes_bool(string):
    if "?" in string:
        return 1
    else:
        return 0
def convertir_cantidad_pasajes_numeric(string):
    if len(string) == 1:
        return int(string)
    elif len(string) == 3:
        string = string.split("+")
        return int(string[1])
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
    ("LDA","LR","MLP"))

if add_selectbox_2 == "LDA":
    model = load_model("lda_v3")
    st.sidebar.info("LDA info...")
if add_selectbox_2 == "LR":
    model = load_model("lr_v3")
    st.sidebar.info("LR info...")
if add_selectbox_2 == "MLP":
    model = load_model("mlp_v3")
    st.sidebar.info("MLP info...")

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

    Cantidad_de_Pasajes = st.text_input("Cantidad de Pasajes",value="?+1")
    if '+' in Cantidad_de_Pasajes:
        Cantidad_de_Pasajes = Cantidad_de_Pasajes.split("+")
        if Cantidad_de_Pasajes[0] == "?":
            Cantidad_de_Pasajes_bool = 1
        else:
            Cantidad_de_Pasajes_bool = 0
        try:
            Cantidad_de_Pasajes_numeric = int(Cantidad_de_Pasajes[1])
        except ValueError:
            Cantidad_de_Pasajes_numeric = 1
    else:
        Cantidad_de_Pasajes_bool = 0
        try:
            Cantidad_de_Pasajes_numeric = int(Cantidad_de_Pasajes)
        except ValueError:
            Cantidad_de_Pasajes_numeric = 1

    Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas = st.number_input('Tiempo con bajo suero (DMEM 0,5% SFB) en horas', 0,1000,value=72)
    Origen = st.selectbox("Origen",["Local","Extranjero"])
    Maduracion = st.number_input('% Maduracion', 0.0,100.0,value=63.37)
    Calidad = st.selectbox("Calidad",["Sin Observaciones","Feos"])
    Clivaje = st.number_input('% Clivaje', 0.0,100.0,value=37.38)
    Medio_Placas_del_dia = st.selectbox("Medio Placas del dia",["F12","Global"])
    Dia_cambio_de_medio = st.number_input('Dia cambio de medio', 0,100,value=5)
    Dia_de_evolucion = st.selectbox("Dia de evolucion",["D7","D8","D9"])
    Grado_embrionario = st.selectbox("Grado embrionario",["I","II","III"])
    Fragmentacion_celular = st.selectbox("Fragmentacion celular",["Sí","No"])
    # Eje_menor = st.number_input('ø Eje menor', 0.0,1000.0,value=541.3)
    # Eje_mayor = st.number_input('ø Eje mayor', 0.0,1000.0,value=564.6)
    Tipo_de_ovocito = st.selectbox("Tipo de ovocito",["Con ZP","Sin ZP"])
    Tipo_embrion = st.selectbox("Tipo de embrion",["Fresco","Vitrificado"])
    Dias_post_ovulacion = st.number_input('Días post-ovulación', 0,10,value=5)

    input_dict = {
        # 'Fecha Transferencia':Fecha_Transferencia, 'Fecha Producción':Fecha_Produccion, 'Fecha Descongelamiento':Fecha_Descongelamiento,
        # 'Línea celular':Linea_celular, 'Lente objetivo':Lente_objetivo, 'Magnificación óptica':Magnificacion_optica, 'Yegua receptora N°':Yegua_receptora
        'Tipo celular':Tipo_celular, 'Días de cultivo celular':Dias_de_cultivo_celular, '% Confluencia':Confluencia,
        'Cantidad de Pasajes bool':Cantidad_de_Pasajes_bool, 'Cantidad de Pasajes numeric':Cantidad_de_Pasajes_numeric,
        'Tiempo con bajo suero (DMEM 0,5% SFB) en horas':Tiempo_con_bajo_suero_DMEM_0_5_SFB_en_horas, 'Origen':Origen, '% Maduración':Maduracion, 'Calidad':Calidad,
        '% Clivaje':Clivaje, 'Medio Placas del día':Medio_Placas_del_dia, 'Día cambio de medio':Dia_cambio_de_medio, 'Día de evolución':Dia_de_evolucion, 'Grado embrionario':Grado_embrionario,
        'Fragmentación celular':Fragmentacion_celular,'Tipo de ovocito':Tipo_de_ovocito, 'Tipo embrión':Tipo_embrion,'Días post-ovulación':Dias_post_ovulacion}
    # print(input_dict)

    # input_dict = {
    #     # 'Fecha Transferencia':Fecha_Transferencia, 'Fecha Producción':Fecha_Produccion, 'Fecha Descongelamiento':Fecha_Descongelamiento,
    #     # 'Línea celular':Linea_celular, 'Lente objetivo':Lente_objetivo, 'Magnificación óptica':Magnificacion_optica, 'Yegua receptora N°':Yegua_receptora
    #     'Tipo celular': "MSC-E", 'Días de cultivo celular': 7, '% Confluencia': 100,
    #     'Cantidad de Pasajes bool': 0,
    #     'Cantidad de Pasajes numeric': 1,
    #     'Tiempo con bajo suero (DMEM 0,5% SFB) en horas': 120, 'Origen': "Local",
    #     '% Maduración': 64.5299987792969, 'Calidad': "Sin observaciones",
    #     '% Clivaje': 34.9099998474121, 'Medio Placas del día': "Global", 'Día cambio de medio': 5,
    #     'Día de evolución': "D7", 'Grado embrionario': "III",
    #     'Fragmentación celular': "No", 'ø Eje menor': None, 'ø Eje mayor': None,
    #     'Tipo de ovocito': "Con ZP", 'Tipo embrión': "Fresco"}

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

        data['Cantidad de Pasajes bool'] = data['Cantidad de Pasajes'].apply(convertir_cantidad_pasajes_bool)
        data['Cantidad de Pasajes numeric'] = data['Cantidad de Pasajes'].apply(convertir_cantidad_pasajes_numeric)
        data = data.drop('Cantidad de Pasajes', axis=1)

        predictions = predict_model(estimator=model, data = data)
        st.write(predictions)