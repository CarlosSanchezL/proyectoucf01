# Human Action Recognition with LSTM (UCF101 Skeleton Dataset)

Este proyecto implementa un modelo de deep learning para clasificar acciones humanas usando coordenadas 2D de esqueletos del dataset UCF101.
Incluye preprocesamiento, data loader, modelos, entrenamiento, evaluación y predicción final.

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

## 📥 Dataset

Colocar el archivo:

```
data/ucf101_2d.pkl
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
