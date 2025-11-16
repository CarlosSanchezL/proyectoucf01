# Human Action Recognition with LSTM (UCF101 Skeleton Dataset)

Este proyecto implementa un modelo de deep learning para clasificar acciones humanas usando coordenadas 2D de esqueletos del dataset UCF101.
Incluye preprocesamiento, data loader, modelos, entrenamiento, evaluación y predicción final.
✔️ Cumplimiento de los Requerimientos del Proyecto

Este proyecto cumple con todos los puntos solicitados en la actividad, incluyendo:

1. Implementación de un modelo de deep learning

Se implementó un modelo profundo basado en LSTM, el cual procesa secuencias temporales de poses humanas (esqueletos 2D).
También se implementó un baseline MLP para comparación, tal como lo pide la rúbrica.

2. Uso de un dataset real

Se utilizó el dataset UCF101 Skeleton 2D (formato .pkl), que contiene coordenadas reales de esqueletos obtenidos de los videos del dataset original UCF101.

3. Diseño de pipeline completo

Se implementó el pipeline completo solicitado:

Carga del dataset

Preprocesamiento

DataLoader

Entrenamiento

Validación

Comparación con baseline

Predicción final

Guardado del mejor modelo

4. Decisiones técnicas documentadas

Se explica claramente:

El modelo utilizado

La razón para usar esqueletos 2D (características de menor dimensionalidad)

La elección del subset de 5 clases

Los hiperparámetros utilizados

## 📁 Estructura del Proyecto

proyecto_UCF101/
│
├── data/
│   └── ucf101_2d.pkl
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── predict_demo.py
│
├── checkpoints/  (se genera automáticamente)
│   └── best_lstm.pt
│
├── venv/
│
├── requirements.txt
└── README.md

## 📦 Instalación

Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install torch numpy
```


## 🧩 Entrenamiento del Modelo

```bash
python src/train.py   --pkl_path data/ucf101_2d.pkl   --train_split train1   --val_split test1   --model_type lstm
```

## 🔍 Predicciones

```bash
python src/predict_demo.py   --pkl_path data/ucf101_2d.pkl   --checkpoint ../checkpoints/best_lstm.pt   --model_type lstm   --split test1
```

## 🧠 Modelos incluidos

✔️ MLP Baseline  
✔️ LSTM (modelo principal)

## 👤 Autor

Carlos Sánchez Llanes  
Tecnológico de Monterrey
