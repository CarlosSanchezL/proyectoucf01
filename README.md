# Human Action Recognition with LSTM (UCF101 Skeleton Dataset)

Este proyecto implementa un sistema completo para **clasificación de acciones humanas** usando coordenadas 2D de esqueletos del dataset **UCF101 Skeleton**.  
Incluye preprocesamiento, carga del dataset, modelos baseline y avanzados, entrenamiento, evaluación, mejoras y predicción desde consola.

---

## ✅ puntos a corregir solicitados por el profesor

### **1. Modelo de Deep Learning (Requerido)**
- **Modelo principal:** LSTM  
- **Baseline:** MLP (comparación obligatoria)

Ambos fueron entrenados y evaluados correctamente.

---

### **2. Uso de un dataset real**
Se utiliza el archivo real:

```
data/ucf101_2d.pkl
```

Con los splits originales del dataset:

```
train1, train2, train3, test1, test2, test3
```

---

### **3. Pipeline completo**
El proyecto contiene:

- Carga del dataset  
- Preprocesamiento (normalización, padding/truncado)  
- DataLoader  
- Entrenamiento  
- Validación  
- Evaluación en test  
- Comparación baseline vs mejoras  
- Generación de predicciones  
- Guardado automático del mejor modelo  

---

### **4. Entrenamiento + Mejoras (Requisito del profesor)**
me pidio mejorar:

> “Evalúa el desempeño del modelo en su aproximación inicial y realiza ajustes para mejorar su desempeño.”

Se cumplió mediante:

| Modelo | Mejora implementada | Resultado |
|--------|----------------------|-----------|
| **MLP (baseline)** | Ninguna | Base para comparación |
| **LSTM base** | Ninguna | Mejor que baseline |
| **LSTM mejorado** | `weight_decay` + `grad clipping` | Mejor estabilidad y mejor val_acc |

---

### **5. Predicciones desde consola (Requisito del profesor)**
Se agregó la opción:

```
--video_name NOMBRE_DEL_VIDEO
```

Para predecir un video específico.

---

# 📁 Estructura del Proyecto

```
proyectoucf01/
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
├── checkpoints/ #Se genera automaticamente
│   ├── best_mlp.pt
│   ├── best_lstm.pt
│   ├── results_mlp.json
│   ├── results_lstm_base.json
│   └── results_lstm_mejorado.json
│
├── venv/
├── requirements.txt
└── README.md
```

---

# 📦 Instalación

### 1. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm
```

---

# 🧠 Entrenamiento de los Modelos

## **1. MLP Baseline**
```bash
python src/train.py   --pkl_path data/ucf101_2d.pkl   --train_split train1   --val_split test1   --test_split test2   --model_type mlp   --save_results
```

## **2. LSTM Base**
```bash
python src/train.py   --pkl_path data/ucf101_2d.pkl   --train_split train1   --val_split test1   --test_split test2   --model_type lstm   --save_results
```

## **3. LSTM Mejorado (Clipping + Weight Decay)**
```bash
python src/train.py   --pkl_path data/ucf101_2d.pkl   --train_split train1   --val_split test1   --test_split test2   --model_type lstm   --weight_scale 1e-4   --clip_grad 5.0   --save_results
```

---

# 📊 Resultados (Reales)

| Modelo | Val Acc | Test Acc |
|--------|---------|-----------|
| **MLP baseline** | 0.7104 | 0.6532 |
| **LSTM base** | 0.7377 | 0.7746 |
| **LSTM mejorado** | 0.7596 | 0.6879 |

---

# 🔍 Predicciones desde consola

### Por índice:
```bash
python src/predict_demo.py   --pkl_path data/ucf101_2d.pkl   --checkpoint checkpoints/best_lstm.pt   --model_type lstm   --split test1   --index 0
```

### Por nombre de video:
```bash
python src/predict_demo.py   --pkl_path data/ucf101_2d.pkl   --checkpoint checkpoints/best_lstm.pt   --model_type lstm   --video_name v_ApplyEyeMakeup_g01_c01
```

---

# 👤 Autor
**Carlos Sánchez Llanes**  
Tecnológico de Monterrey
