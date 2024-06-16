#−−−−−−−−−−−−−−----------
# Grupo N: 6            |
# Integrantes           |
#−−−−−−−−−−−−−-----------
# Cristian Cortés       |
# David González        |
#−−−−−−−−−−−−−-----------

# Importamos las bibliotecas que utilizaremos
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

#Cargamos el set de datos Flights
df = sns.load_dataset("flights")

# Título de la aplicación
st.markdown("<h1 style='text-align: center; color: blue;'>Análisis Interactivo con Set de Datos Flights</h1>", unsafe_allow_html=True)

# Descripción del conjunto de datos, mostrando las primeras 5 filas.
st.markdown("<h2 style='text-align: center; color: grey;'>Qué contiene el conjunto de datos?</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>El conjunto de Datos Flights contiene información acerca de vuelos de un aeropuerto, Incluye datos como Año, Mes y Cantidad de pasajeros.</h4> ", unsafe_allow_html=True)
st.write(df.head()) 
st.write(df.describe())

# Generamos estadística descriptiva del conjunto de Datos.
st.markdown("<h2 style='text-align: center; color: grey;'>Estadística Descriptiva</h2>", unsafe_allow_html=True)
st.write(f"La Media de la cantidad de pasajeros es: {df.passengers.mean():.2f}")
st.write(f"La Mediana de la cantidad de pasajeros es: {df.passengers.median():.2f}")
st.write(f"La Moda de la cantidad de pasajeros es: {df['passengers'].mode().iloc[0]}")
st.write(f"La Desviación Estándar de la cantidad de pasajeros es: {df.passengers.std():.2f}")

# Histograma interactivo
st.markdown("<h2 style='text-align: center; color: grey;'>Histograma Interactivo</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>El Histograma representa la distribución de una variable continua en intervalos (bins), Puedes ver la forma de la distribución y la concentración de valores en diferentes rangos seleccionando la variable del eje X de tu preferencia</h5> ", unsafe_allow_html=True)
#Caja de selección de variables 
column = st.selectbox("Selecciona la columna para el histograma", df.select_dtypes(include='number').columns)
bins = st.slider("Número de bins", 10, 100, 30)
# Crear el Histograma
fig, ax = plt.subplots()
ax.hist(df[column].dropna(), bins=bins, edgecolor='black')
ax.set_title(f'Histograma de {column}')
st.pyplot(fig)

#Diagrama de Caja Interactivo
st.markdown("<h2 style='text-align: center; color: grey;'>Diagrama de Caja Interactivo</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Los box plots nos permiten ver cómo se distribuyen los datos en una variable. Podemos identificar si la distribución es simétrica, sesgada o si hay valores atípicos seleccionando la variable del eje X.</h5> ", unsafe_allow_html=True)
#Caja de selección de variables para el eje x
x_axis = st.selectbox("Selecciona la variable para el eje X", ['month', 'year'], key='boxplot_x_axis')
y_axis = 'passengers'
# Crear el gráfico de caja
fig2, ax2 = plt.subplots()
sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax2)
ax2.set_title(f'Diagrama de Caja de {y_axis} por {x_axis}')
st.pyplot(fig2)
    

#Gráfico de Dispersión
st.markdown("<h2 style='text-align: center; color: grey;'>Gráfico de Dispersión</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Un gráfico de dispersión muestra la relación entre dos variables continuas. Puedes identificar patrones de correlación y detectar valores atípicos o anomalías seleccionando la variable del eje X.</h5> ", unsafe_allow_html=True)
#Caja de selección de variables para el eje x
x_axis_scatter = st.selectbox("Selecciona la variable para el eje X", ['month', 'year'], key='scatter_x_axis')
y_axis_scatter = 'passengers'
# Crear el gráfico de dispersión
fig3, ax3 = plt.subplots()
sns.scatterplot(x=x_axis_scatter, y=y_axis_scatter, data=df, ax=ax3)
ax3.set_title(f'Gráfico de Dispersión de {y_axis_scatter} vs {x_axis_scatter}')
st.pyplot(fig3)

# Gráfico de dispersión con categorías interactivo
st.markdown("<h2 style='text-align: center; color: grey;'>Gráfico de Dispersión con Selección de Meses</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Gráfico de Dispersión donde puede seleccionar los meses para ser considerados en la medición, considerando la variable pasajeros en el eje Y y la variable Año en el eje X.</h5> ", unsafe_allow_html=True)
# Lista de meses
months = df['month'].unique().tolist()
# Checkbox para seleccionar los meses
selected_months = st.multiselect("Selecciona los meses a mostrar", months, default=months)
# Filtrar el DataFrame con los meses seleccionados
filtered_df = df[df['month'].isin(selected_months)]
# Crea el gráfico de dispersión con categorías
fig4, ax4 = plt.subplots()
sns.scatterplot(x='year', y='passengers', hue='month', data=filtered_df, ax=ax4)
ax4.set_title('Gráfico de Dispersión de Pasajeros por Año')
st.pyplot(fig4)

# Paso 1: Preparar los datos
# Convertir la columna 'month' a números
df['month'] = df['month'].apply(lambda x: pd.to_datetime(x, format='%b').month)
# Crear una nueva columna de fecha
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
# Establecer 'date' como índice
df.set_index('date', inplace=True)
# Eliminar las columnas 'year' y 'month' ya que no las necesitamos más
df.drop(['year', 'month'], axis=1, inplace=True)
# Paso 2: Descomponer la serie temporal
result = seasonal_decompose(df['passengers'], model='multiplicative', period=12)
# Paso 3: Modelo de predicción, Crear el modelo ARIMA
model = ARIMA(df['passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# Ajustar el modelo
model_fit = model.fit()
# Realizar la proyección para los próximos 24 meses
forecast = model_fit.get_forecast(steps=24)
forecast_index = pd.date_range(start=df.index[-1], periods=24, freq='M')
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
# Evaluar y visualizar la proyección
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['passengers'], label='Datos históricos')
ax.plot(forecast_series, label='Proyección', color='red')
ax.fill_between(forecast_series.index, 
                forecast.conf_int().iloc[:, 0], 
                forecast.conf_int().iloc[:, 1], 
                color='pink', alpha=0.3)

st.markdown("<h2 style='text-align: center; color: grey;'>Proyección de Número de Pasajeros</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Proyección del número de pasajeros para los próximos 24 meses (1961 y 1962) usando un modelo ARIMA</h5> ", unsafe_allow_html=True)
ax.legend()
ax.set_title('Proyección del número de pasajeros')
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de pasajeros')
st.pyplot(fig)


# Paso 1: Calcular el crecimiento anual de los pasajeros
st.markdown("<h2 style='text-align: center; color: grey;'>Crecimiento Anual de los Pasajeros</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Cálculo del crecimiento anual de los pasajeros y la integración de datos económicos para analizar la correlación entre el crecimiento del PIB y el crecimiento del número de pasajeros.</h5> ", unsafe_allow_html=True)
# Resumir el número de pasajeros por año
passengers_yearly = df['passengers'].resample('Y').sum().reset_index()
passengers_yearly['year'] = passengers_yearly['date'].dt.year
# Calcular la variación porcentual respecto al año anterior
passengers_yearly['growth'] = passengers_yearly['passengers'].pct_change() * 100
# Crear un nuevo DataFrame con los resultados
growth_df = passengers_yearly[['year', 'growth']].dropna()
growth_df['growth'] = growth_df['growth'].round(1)
# Filtrar el DataFrame para mostrar solo desde 1950 hasta 1960
growth_df = growth_df[(growth_df['year'] >= 1950) & (growth_df['year'] <= 1960)]
# Mostrar el nuevo DataFrame
st.write(growth_df)
# Solicitar el contenido de la página web
url = 'https://datosmacro.expansion.com/pib/usa?anio=1970'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
# Encontrar todas las tablas en la página
tables = soup.find_all('table', {'class': 'table tabledat table-striped table-condensed table-hover'})
# Seleccionar la tercera tabla que es la evolución del PIB anual
table = tables[2]
# Crear listas para almacenar las fechas y variaciones del PIB
fechas = []
var_pib = []
# Iterar sobre las filas de la tabla (omitir la primera fila de encabezado)
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    # Verificar si la fila tiene al menos tres celdas
    if len(cells) < 3:
        continue
    fecha = cells[0].text.strip()
    var_pib_valor = cells[2].text.strip()  # Asumiendo que la columna Var. PIB (%) es la tercera columna
    # Verificar si la fecha contiene solo el año
    if fecha.isdigit() and 1948 <= int(fecha) <= 1970:
        fechas.append(fecha)
        var_pib.append(var_pib_valor)
# Crear un DataFrame de pandas con los datos extraídos
econ_data = pd.DataFrame({'Fecha': fechas, 'Var. PIB (%)': var_pib})
# Mostrar el DataFrame en Streamlit
st.markdown("<h2 style='text-align: center; color: grey;'>Datos del PIB desde datosmacro.expansion.com</h2>", unsafe_allow_html=True)
st.write(econ_data)
# Integrar los dos conjuntos de datos
st.markdown("<h2 style='text-align: center; color: grey;'>Integración de los Datos de Crecimiento de Pasajeros y PIB</h2>", unsafe_allow_html=True)
# Convertir la columna 'Fecha' a tipo entero en econ_data
econ_data['Fecha'] = econ_data['Fecha'].astype(int)
# Transformar la columna 'Var. PIB (%)' a número quitando el signo de "%" y reemplazando comas por puntos
econ_data['Var. PIB (%)'] = econ_data['Var. PIB (%)'].str.replace('%', '').str.replace(',', '.').astype(float)
# Filtrar econ_data para que coincida con los años del growth_df (1950-1960)
econ_data_filtered = econ_data[econ_data['Fecha'].isin(growth_df['year'])]
# Renombrar columnas para combinar los datasets
econ_data_filtered.rename(columns={'Fecha': 'year', 'Var. PIB (%)': 'gdp_growth'}, inplace=True)
growth_df.rename(columns={'growth': 'passenger_growth'}, inplace=True)
# Combinar los dos datasets
combined_df = growth_df.merge(econ_data_filtered, on='year')
# Mostrar el DataFrame combinado en Streamlit
st.write(combined_df)
# Gráfico de línea de variación de pasajeros y PIB
st.markdown("<h2 style='text-align: center; color: grey;'>Gráfico de Línea de Variación de Pasajeros y PIB</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.plot(combined_df['year'], combined_df['passenger_growth'], marker='o', label='Crecimiento de Pasajeros')
ax.plot(combined_df['year'], combined_df['gdp_growth'], marker='o', label='Crecimiento del PIB')
ax.set_title('Variación de Pasajeros y PIB (1950-1960)')
ax.set_xlabel('Año')
ax.set_ylabel('Variación (%)')
ax.legend()
st.pyplot(fig)

# Calcular la correlación entre el crecimiento de pasajeros y el crecimiento del PIB
correlation = combined_df['passenger_growth'].corr(combined_df['gdp_growth'])

# Mostrar la correlación en Streamlit
st.markdown("<h2 style='text-align: center; color: grey;'>Correlación entre Crecimiento de Pasajeros y Crecimiento del PIB</h2>", unsafe_allow_html=True)
st.write(f"La correlación entre el crecimiento de pasajeros y el crecimiento del PIB es: {correlation:.2f}")

# Entrenar el modelo de regresión lineal
X = combined_df['gdp_growth'].values.reshape(-1, 1)
y = combined_df['passenger_growth'].values

model = LinearRegression()
model.fit(X, y)

# Filtrar los datos de PIB para los años 1961 a 1970
pib_data_61_70 = econ_data[(econ_data['Fecha'] >= 1961) & (econ_data['Fecha'] <= 1970)]

# Asegurarnos de que todos los valores en 'Var. PIB (%)' sean strings
pib_data_61_70['Var. PIB (%)'] = pib_data_61_70['Var. PIB (%)'].astype(str)

# Transformar la columna 'Var. PIB (%)' a número quitando el signo de "%" y reemplazando comas por puntos
pib_data_61_70['Var. PIB (%)'] = pib_data_61_70['Var. PIB (%)'].str.replace('%', '').str.replace(',', '.').astype(float)

# Solicitar al usuario el año para predecir el número de pasajeros
st.markdown("<h2 style='text-align: center; color: grey;'>Predicción del Número de Pasajeros</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Predicción del número de pasajeros para un año específico (1961-1970), basado en el crecimiento del PIB, usando un modelo de regresión lineal.</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>Se utilizaron esos años dado que en la década del 70 la industria comenzó cambios profundos y solo con la variación del PIB ya no se podría proyectar.</h5>", unsafe_allow_html=True)
year = st.selectbox("Selecciona un año entre 1961 y 1970:", pib_data_61_70['Fecha'])

# Obtener el crecimiento del PIB para el año seleccionado
gdp_growth_input = pib_data_61_70[pib_data_61_70['Fecha'] == year]['Var. PIB (%)'].values[0]

# Predecir el crecimiento de pasajeros basado en el crecimiento del PIB proporcionado
predicted_growth = model.predict(np.array([[gdp_growth_input]]))[0]

# Calcular el número de pasajeros basado en el crecimiento predicho
last_known_passengers = df.resample('Y').sum().loc['1960', 'passengers'].item()
predicted_passengers = last_known_passengers * (1 + predicted_growth / 100)

# Calcular la media, máximo y mínimo del número de pasajeros predicho
mean_passengers = predicted_passengers
max_passengers = mean_passengers * 1.05  # Asumiendo un 5% de variabilidad
min_passengers = mean_passengers * 0.95  # Asumiendo un 5% de variabilidad

st.markdown("<h5 style='text-align: center; color: grey;'>Presentación del número probable de pasajeros, junto con un rango de valores posibles (media, máximo y mínimo).</h5>", unsafe_allow_html=True)
# Mostrar los resultados en Streamlit
st.write(f"El número probable de pasajeros en el año {year} es: **{predicted_passengers:.0f}**")
st.write(f"Media de pasajeros: {mean_passengers:.0f}")
st.write(f"Máximo de pasajeros: {max_passengers:.0f}")
st.write(f"Mínimo de pasajeros: {min_passengers:.0f}")