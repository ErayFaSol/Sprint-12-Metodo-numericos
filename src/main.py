from preprocessing.preprocessing import load_data, clean_data
from models.model_trainer import train_linear_regression, train_decision_tree, train_lightGBM, train_random_forest, predict_model
from utils.metrics import calculate_rmse
from utils.visualizations import plot_rmse_comparison, plot_training_time_comparison
from utils.generate_report import save_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Cargar y limpiar datos
df = load_data('datasets/car_data.csv')
df_clean = clean_data(df)
df_clean = df.dropna(subset=['Price'])

# Dividir datos
X = df.drop(columns=['Price', 'DateCrawled', 'DateCreated', 'LastSeen'])  # Excluir fechas y precio
y = df_clean['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# Codificación de variables categóricas (OHE)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Asegurarse de que ambas matrices tienen las mismas columnas
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Entrenar modelos
lin_reg, training_time_lin = train_linear_regression(X_train, y_train)
tree_reg, training_time_tree = train_decision_tree(X_train, y_train)
forest_reg, training_time_forest = train_random_forest(X_train, y_train)
lgbm_reg, training_time_lgbm = train_lightGBM(X_train,y_train, X_test, y_test)

# Evaluar modelos
y_pred_lin = predict_model(lin_reg, X_test)
y_pred_tree = predict_model(tree_reg, X_test)
y_pred_forest = predict_model(forest_reg, X_test)
y_pred_lgbm = predict_model(lgbm_reg, X_test)

# Calcular RMSE
rmse_lin = calculate_rmse(y_test, y_pred_lin)
rmse_tree = calculate_rmse(y_test, y_pred_tree)
rmse_forest = calculate_rmse(y_test, y_pred_forest)
rmse_lgbm = calculate_rmse(y_test, y_pred_lgbm)


# Comparar modelos
results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'LightGBM'],
    'RMSE': [rmse_lin, rmse_tree, rmse_forest, rmse_lgbm],
    'Training Time (s)': [training_time_lin, training_time_tree, training_time_forest, training_time_lgbm]
}

# Graficar RMSE y tiempos de entrenamiento
plot_rmse_comparison([rmse_lin, rmse_tree, rmse_forest, rmse_lgbm], ['Linear Regression', 'Decision Tree', 'Random Forest', 'LightGBM'])
plot_training_time_comparison(['Linear Regression', 'Decision Tree', 'Random Forest', 'LightGBM'], [training_time_lin, training_time_tree, training_time_forest, training_time_lgbm])

# Imprimir resultados finales
for model, rmse, time in zip(results['Model'], results['RMSE'], results['Training Time (s)']):
    print(f"{model} - RMSE: {rmse:.2f}, Training Time: {time:.2f} seconds")
    
# Generar reporte 
analysis_results = """
<h2>1. Introducción</h2>
<p>El objetivo de este análisis fue construir un modelo de predicción para determinar el valor de mercado de automóviles usados utilizando varios algoritmos de aprendizaje automático.</p>

<h2>2. Resultados del Modelo</h2>
<h3>2.1 RMSE por Modelo</h3>
<p>A continuación se presenta la comparación de los diferentes modelos utilizados y sus resultados de error cuadrático medio (RMSE).</p>
<img src='rmse_comparison.png' alt='Comparación de RMSE entre modelos' width='600'>

<h3>2.2 Tiempo de Entrenamiento por Modelo</h3>
<p>El siguiente gráfico muestra el tiempo de entrenamiento de cada uno de los modelos.</p>
<img src='training_time_comparison.png' alt='Comparación de tiempo de entrenamiento entre modelos' width='600'>

<h2>3. Conclusiones</h2>
<p>El modelo <strong>LightGBM</strong> fue el más preciso en términos de RMSE, aunque su tiempo de entrenamiento fue más largo comparado con otros modelos como la regresión lineal y los árboles de decisión.</p>
<p>En general, el <strong>LightGBM</strong> fue el modelo recomendado, ya que ofrece el mejor balance entre precisión y eficiencia para los datos proporcionados.</p>
"""

save_report(analysis_results)