# Human Action Recognition with LSTM (UCF101 Skeleton Dataset)

Este proyecto implementa un modelo de deep learning para clasificar acciones humanas usando coordenadas 2D de esqueletos del dataset UCF101.
Incluye preprocesamiento, data loader, modelos, entrenamiento, evaluación y predicción final.

Este proyecto cumple con todos los puntos solicitados:

✔️ 1. Modelo de deep learning

-Implementado: LSTM (modelo principal)
-Baseline incluido: MLP (comparación requerida)

✔️ 2. Uso de un dataset real
-Se utiliza UCF101 Skeleton 2D (.pkl) proveniente del dataset oficial.

✔️ 3. Pipeline completo
-Carga del dataset
-Preprocesamiento
-DataLoader
-Entrenamiento
-Validación
-Comparación baseline
-Predicción final
-Guardado del mejor modelo

✔️ 4. Entrenamiento y mejoras
-Se entrenó baseline y luego se mejoró con LSTM (mayor accuracy).
-Se usó regularización (weight decay) y clipping de gradiente.

✔️ 5. Predicciones funcionales
-El modelo genera predicciones reales desde consola.

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
