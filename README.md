# Métodos Numéricos

Este proyecto tiene como objetivo mejorar las operaciones de una compañía de seguros mediante el uso de modelos de machine learning y técnicas de análisis numérico. Se desarrollaron modelos predictivos que permitieron a la empresa tomar decisiones más informadas en áreas clave como la gestión de riesgos y optimización de procesos.

## Tecnologías utilizadas
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Objetivo
Optimizar las operaciones internas de una compañía de seguros a través de la aplicación de técnicas de aprendizaje automático y métodos numéricos, mejorando la eficiencia en la gestión de riesgos y recursos.

## Contexto
Las compañías de seguros manejan grandes cantidades de datos sobre clientes, reclamos y riesgos. Utilizando estos datos, se pueden desarrollar modelos predictivos para optimizar procesos y minimizar el riesgo financiero. Este proyecto se centra en la aplicación de métodos numéricos y machine learning para mejorar el análisis de estos datos y ayudar a la empresa a tomar decisiones más precisas.

## Descripción del Proyecto
El proyecto involucra la creación de varios modelos de machine learning para mejorar áreas clave de la compañía, como la predicción de reclamaciones y la gestión de riesgos. El análisis incluye el preprocesamiento de datos, la implementación de algoritmos de aprendizaje supervisado, y la evaluación de la precisión de los modelos.

Pasos del proyecto:
1. **Análisis exploratorio de datos (EDA)**: Identificar patrones y correlaciones en los datos de la compañía.
2. **Preprocesamiento de datos**: Limpieza y transformación de los datos.
3. **Desarrollo de modelos predictivos**: Implementación de modelos de regresión y clasificación.
4. **Evaluación de los modelos**: Medir el rendimiento utilizando métricas como precisión, recall y el puntaje F1.

## Proceso

### Análisis Exploratorio de Datos (EDA)
Se llevó a cabo un análisis exhaustivo de los datos proporcionados por la compañía de seguros utilizando pandas, Matplotlib y Seaborn. Se analizaron variables clave como la frecuencia y el monto de las reclamaciones, la edad de los clientes, y el tipo de póliza contratada. Estas variables se utilizaron posteriormente como insumos para los modelos predictivos.

### Preprocesamiento de Datos
Se realizó la limpieza de datos, incluyendo el manejo de valores faltantes y la codificación de variables categóricas. También se normalizaron los datos para garantizar que todas las variables estén en la misma escala.

### Desarrollo de Modelos
Se entrenaron varios modelos de machine learning, entre ellos:
- **Regresión Lineal**: Para predecir el monto de las reclamaciones.
- **Random Forest**: Para la clasificación de clientes en riesgo de realizar futuras reclamaciones.
- **K-Nearest Neighbors (KNN)**: Utilizado para comparar resultados con modelos supervisados más complejos.

El modelo de **Random Forest** resultó ser el más efectivo, con una precisión del 87% y un puntaje F1 de 0.85.

### Evaluación del Modelo
Las métricas utilizadas para evaluar el modelo fueron:
- **Precisión**: 87%
- **Recall**: 0.84
- **Puntaje F1**: 0.85
- **Matriz de confusión**: Mostró una buena capacidad de predicción, aunque con algunos falsos positivos en las predicciones de reclamaciones menores.

## Resultados
El modelo de **Random Forest** logró identificar correctamente los clientes con mayor probabilidad de realizar futuras reclamaciones, lo que permitió a la compañía ajustar sus estrategias de gestión de riesgos y optimizar sus procesos internos.

## Conclusiones
El uso de métodos numéricos y machine learning proporcionó a la compañía de seguros una herramienta eficiente para predecir el comportamiento de los clientes y mejorar la gestión de riesgos. Los modelos desarrollados ofrecen un enfoque basado en datos que permite optimizar la toma de decisiones en la empresa.

### Futuras mejoras
- Incluir más variables en el análisis para mejorar la robustez del modelo.
- Implementar técnicas de optimización de hiperparámetros para afinar el rendimiento.
- Experimentar con modelos más avanzados como XGBoost o LightGBM para mejorar la precisión de las predicciones.

### Enlace al proyecto
[Métodos Numéricos](https://github.com/ErayFaSol/Sprint-12-Metodo-numericos)
