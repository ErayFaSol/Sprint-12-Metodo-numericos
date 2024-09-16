# %% [markdown]
# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial: especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:
# - la calidad de la predicción;
# - la velocidad de la predicción;
# - el tiempo requerido para el entrenamiento

# %% [markdown]
# ## Preparación de datos

# %%
# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# %%
# Cargar DataSet
df = pd.read_csv("datasets/car_data.csv")

# %%
# Exploracion inicial 
print(('===== Rusty Bargain DataFrame ====='))
print('------------------------------')

print(('==== Informacion general ===='))
print(df.info())
print('------------------------------')

print(('==== Estadisticas descriptivas ===='))
print(df.describe())
print('------------------------------')

print(('==== Muestra de tabla ===='))
print(df.sample(5))

# %% [markdown]
# ## Preprocesamiento de datos

# %%
# Eliminar filas duplicadas
print(f"Numero de duplicados previos: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Verificacion de duplicados: {df.duplicated().sum()}")

# %%
print(f"Numero de valores nulos previos: {df.isnull().sum()}")

# Imputar valores nulos con la moda para las columnas categoricas
df['VehicleType'].fillna(df['VehicleType'].mode()[0], inplace=True)
df['Gearbox'].fillna(df['Gearbox'].mode()[0], inplace=True)
df['Model'].fillna(df['Model'].mode()[0], inplace=True)
df['FuelType'].fillna(df['FuelType'].mode()[0], inplace=True)
df['NotRepaired'].fillna('no', inplace=True)  # Asumimos que la mayoría de los autos no están reparados

print(f"Verificacion de valores nulos : {df.isnull().sum()}")

# %%
# Eliminar filas donde el precio es 0
df = df[df['Price'] > 0]

# %%
# Definir el rango de años de registro válidos
current_year = datetime.now().year
df = df[(df['RegistrationYear'] >= 1900) & (df['RegistrationYear'] <= current_year)]

# %%
# Eliminar filas donde la potencia es 0
df = df[df['Power'] > 0]

# %%
# Convertir columnas de fechas a formato datetime
df['DateCrawled'] = pd.to_datetime(df['DateCrawled'], format='%d/%m/%Y %H:%M')
df['DateCreated'] = pd.to_datetime(df['DateCreated'], format='%d/%m/%Y %H:%M')
df['LastSeen'] = pd.to_datetime(df['LastSeen'], format='%d/%m/%Y %H:%M')


# %%
# Eliminar columnas irrelevantes
# Nota: Reiniciar el kernel despues de la primera ejecucion
df = df.drop(columns=['NumberOfPictures'])


# %%

# Revisión final de los datos
print(('===== Revisión final del DataFrame ====='))
print(df.info())
print(df.describe())
print(df.sample(5))

# %% [markdown]
# ### Preprocesamiento de datos (Explicacion)
# 
# Durante este proceso de preparacion de datos se realizaron varias cosas, por ejemplo
# - Se cargo el archivo car_data.csv en un dataframe y realizamos un analisis de su estructura
# 
# - Se realizo una exploracoon inicial usando metodos como .info() para obtener detalles sobre el numero de entradas, tipos de datos y balores nulos.
# 
# - Generamos estadisticas descriptivas de las columnas numericas usando el metodo .describe()
# 
# - Obtuvimos una muestra aleatoria del dataframe mediante el uso del metodo sample()
# 
# - Se descarto las filas duplicadas con el metodo drop_duplicates
# 
# - Imputamos los valores nulos de las columnas categóricas con la moda (valor más frecuente) usando fillna().
# 
# - Se realizaron cambios en la columnas de fecha convirtiendolo al formato adecuado usando el metodo pd.to_datetime()
# 
# - Tambien se elimino la columna "NumberOfPictures" ya que todos sus valores son 0 y no aportan informacion util al modelo.

# %% [markdown]
# ## Entrenamiento del modelo 

# %%
# Excluir columnas de fechas
columns_to_exclude = ['DateCrawled', 'DateCreated', 'LastSeen']

# Division del conjunto de datos:

X = df.drop(columns=['Price'] + columns_to_exclude)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# %%
# Codificacion de variables categoricas (OHE)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Asegurarse de que ambas matrices tienen las mismas columnas
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# %%
# Regresion linear

# Medir tiempo de entrenamiento 
start_time = time.time()

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Medir el tiempo de entrenamiento
training_time_lin = time.time() - start_time

y_pred_lin = lin_reg.predict(X_test)

#evaluacion
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
print(f'Regresion linear RMSE: {rmse_lin}')
print(f'Regression Linear Training Time: {training_time_lin} seconds')

# %%
# Árbol de decisión 
# Medir el tiempo de entrenamiento
start_time = time.time()

tree_reg = DecisionTreeRegressor(random_state=12345)
tree_reg.fit(X_train, y_train)

# Medir el tiempo de entrenamiento
training_time_tree = time.time() - start_time

# Medir el tiempo de entrenamiento
start_time = time.time()
y_pred_tree = tree_reg.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
print(f'Decision Tree RMSE: {rmse_tree}')
print(f'Decision Tree Training Time: {training_time_tree} seconds')

# %%
# Ajustar los parámetros para reducir el tiempo de entrenamiento
forest_reg = RandomForestRegressor(
    n_estimators=100,       # Reducir el número de árboles
    max_depth=10,           # Limitar la profundidad máxima de los árboles
    max_features='sqrt',    # Reducir el número de características consideradas
    n_jobs=-1,              # Paralelizar el entrenamiento (usa todos los núcleos disponibles)
    random_state=42
)
# Medir el tiempo de entrenamiento
start_time = time.time()

# Entrenar el modelo
forest_reg.fit(X_train, y_train)

# Medir el tiempo de entrenamiento
training_time_forest = time.time() - start_time

# Predicciones
y_pred_forest = forest_reg.predict(X_test)

# Evaluación
mse_forest = mean_squared_error(y_test, y_pred_forest)
rmse_forest = np.sqrt(mse_forest)
print(f'Random Forest RMSE: {rmse_forest}')
print(f'Random Forest Training Time: {training_time_forest} seconds')

# %%
# Preparar datos para LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Medir el tiempo de entrenamiento
start_time = time.time()

# Parámetros
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1
}

# Entrenar el modelo con early stopping
lgbm_reg = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100),
        ],
)
# Medir el tiempo de entrenamiento
training_time_lgbm = time.time() - start_time


# Predicciones
y_pred_lgbm = lgbm_reg.predict(X_test, num_iteration=lgbm_reg.best_iteration)

# Evaluación
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
print(f'LightGBM RMSE: {rmse_lgbm}')
print(f'LightGBM Training Time: {training_time_lgbm} seconds')

# %% [markdown]
# ## Análisis del modelo

# %%
# Comparacion entre modelos
results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'LightGBM'],
    'RMSE': [rmse_lin, rmse_tree, rmse_forest, rmse_lgbm],
    'Training Time (s)': [training_time_lin, training_time_tree, training_time_forest, training_time_lgbm]
}

results_df = pd.DataFrame(results)
print(results_df)


# %%
# Crear gráfico de barras para RMSE
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['RMSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Models')
plt.show()


# %%
# Crear gráfico de barras para Tiempo de Entrenamiento
plt.figure(figsize=(10, 5))
plt.bar(results_df['Model'], results_df['Training Time (s)'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.title('Training Time Comparison of Models')
plt.show()

# %% [markdown]
# ## Conclusion 

# %% [markdown]
# Con base en los resultados obtenidos podemos evaluar que modelo ofrece un balance mejor entre precision y eficiencia. Aqui hay algunas consideraciones
# 
# - Regresion lineal: La de menor tiempo de entrenamiento pero no puede capturar relaciones complejas
# - Arbol de Decision: Puede ser mas preciso pero tiende a sobreajustarse
# - Bosque Aleatorio: Suele ofrecer un buen balance entre precision y tiempo de entrenamieno, mejor que los arboles individuales
# - LightGBM: Generalmente muy preciso y eficiente para los conjuntos de datos mas grandes
# 
# En conclusion el mejor modelo que cumple con los objetivos que les interesa a RustyVargain:
# 
# - Calidad de prediccion: LightGBM ofrece la mejor precisión (menor RMSE), lo que es crucial para una aplicación que determina el valor de mercado de los coches.
# - Velocidad de la prediccion: Aunque el tiempo de entrenamiento es mayor, es aceptable considerando la mejora significativa en la precisión
# - Tiempo requerido para el entremiento: LightGBM ofrece un buen equilibrio entre precision y tiempo de entrenamiento.


