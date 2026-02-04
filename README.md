# ITGStamina-rater-ml

Un enfoque de Machine Learning para clasificar la dificultad de los charts de stamina en In The Groove (ITG) utilizando características estructurales extraídas de la Breakdown Notation.

Este proyecto forma parte de mi trabajo de tesina para la carrera de Ingeniería en Computación Inteligente. El objetivo es transformar la evaluación subjetiva de la comunidad en un modelo objetivo y escalable.

## Estructura del Proyecto
- **ml-core/**: Entorno Python para ciencia de datos, lógica de parsing y entrenamiento de modelos.
- **web-app/**: Aplicación React para inferencia en el navegador usando ONNX.

### Características Principales

- Ingeniería de Características: Extracción de métricas avanzadas (intensidad, pacing, longitud de streams) a partir de la Breakdown Notation.
- Comparación de Modelos: Evaluación sistemática de SVM, Random Forest y Gradient Boosting para encontrar el mejor rendimiento.
- Soporte Moderno: Entrenado con packs de la era actual (ECS 13+ y SRPG recientes).
- Baseline Académico: Uso de SVM como modelo de referencia basado en literatura previa.

### Agradecimientos

Este proyecto se inspira y construye sobre el trabajo previo de investigadores y desarrolladores de la comunidad:

- Eryk Banatt: [Auto-Rating ITG Stamina Charts with Machine Learning](https://github.com/ambisinister/itsa17)
- som1sezhi: [ITG Difficulty Predictor](https://github.com/som1sezhi/itg-difficulty-predictor/)

### ⚠️ Descarga de Responsabilidad
Gran parte del código de este proyecto ha sido generado o asistido mediante modelos de lenguaje (LLM) haciendo **vibe-coding**.

Como estudiante de Ingeniería en Computación Inteligente, mi enfoque principal se centra en la arquitectura del sistema, la selección de modelos, la ingeniería de características y la validación de resultados. Sin embargo, el código resultante está sujeto a errores lógicos o de sintaxis típicos de la IA. Este repositorio es un esfuerzo colaborativo entre la supervisión humana y la automatización, y requiere una revisión constante para asegurar su correcto funcionamiento.

## English Version

A Machine Learning approach to classify the difficulty of In The Groove (ITG) stamina charts using structural features extracted from Breakdown Notation.

This project is developed as part of my undergraduate thesis for Intelligent Computer Engineering. The goal is to bridge the gap between subjective community ratings and an objective, scalable predictive model.

## Project Structure
- **ml-core/**: Python environment for Data Science, parser logic, and model training.
- **web-app/**: React application for browser-based inference using ONNX.

### Features

- Feature Engineering: Advanced metric extraction (intensity, pacing, stream length) based on Breakdown Notation.
- Model Comparison: Systematic evaluation of SVM, Random Forest, and Gradient Boosting models.
- Modern Support: Inclusion of modern-era packs (ECS 13+ and recent SRPG).
- Academic Baseline: Implementation of SVM as a baseline model according to existing literature.

### Acknowledgements

This project is inspired by and builds upon the previous work of community researchers:

- Eryk Banatt: [Auto-Rating ITG Stamina Charts with Machine Learning](https://github.com/ambisinister/itsa17)
- som1sezhi: [ITG Difficulty Predictor](https://github.com/som1sezhi/itg-difficulty-predictor/)

### ⚠️ AI Disclosure
A significant portion of the code in this project has been generated or assisted by Large Language Models (LLMs) doing **vibe-coding**.

As an Intelligent Computer Engineering student, my primary focus is on system architecture, model selection, feature engineering, and result validation. However, the resulting code is prone to AI-generated bugs or logic gaps. This repository represents a collaborative effort between human oversight and automation, and it requires constant auditing to ensure accuracy and reliability.
