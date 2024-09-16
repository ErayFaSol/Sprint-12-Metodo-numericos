# Rusty Bargain Car Price Prediction

## Descripción del Proyecto

Este proyecto está diseñado para ayudar a **Rusty Bargain**, un servicio de venta de autos usados, a predecir el valor de mercado de los vehículos. Utilizando diversas técnicas de aprendizaje automático, hemos desarrollado un modelo que determina los precios óptimos basados en las especificaciones técnicas y características de cada auto.

### Objetivos:
- **Calidad de predicción**: Obtener el menor error posible en las predicciones.
- **Velocidad de predicción**: Minimizar el tiempo requerido para realizar predicciones.
- **Tiempo de entrenamiento**: Optimizar el tiempo necesario para entrenar el modelo.

## Datos

El proyecto utiliza un conjunto de datos que incluye:
- Especificaciones técnicas de vehículos (tipo de combustible, modelo, tipo de vehículo, etc.)
- Información sobre el estado del vehículo
- Año de registro
- Precio del vehículo

### Preprocesamiento de Datos:
- Limpieza de duplicados y valores nulos
- Transformación de columnas categóricas con codificación
- Eliminación de columnas irrelevantes

## Modelos

Se han implementado y comparado varios modelos de regresión:
1. **Regresión Lineal**
2. **Árbol de Decisión**
3. **Bosque Aleatorio**
4. **LightGBM**

### Evaluación de Modelos:
Los modelos se evaluaron usando la métrica **RMSE** (Root Mean Squared Error) y tiempo de entrenamiento. Se graficaron los resultados para comparar la calidad y eficiencia de los modelos.

### Resultados de los Modelos:
| Modelo             | RMSE   | Tiempo de Entrenamiento (s) |
|--------------------|--------|-----------------------------|
| Regresión Lineal    | 2900   | 0.1                         |
| Árbol de Decisión   | 2200   | 0.3                         |
| Bosque Aleatorio    | 2800   | 15.0                        |
| LightGBM            | 1700   | 2.0                         |

El modelo **LightGBM** es el más eficiente en términos de precisión y velocidad, y se seleccionó como la mejor opción.

## Instalacion y configuracion

1. Clonar el repositorio
2. Instalar las dependencias
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el script principal
   ``` 
   python src/main.py
   ```
4. La ejecucion creara un archivo llamado *reporte_final.html*
