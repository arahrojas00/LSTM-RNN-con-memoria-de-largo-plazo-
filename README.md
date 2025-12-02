# LSTM-RNN-con-memoria-de-largo-plazo-
Modelo basado en LSTM (RNN) sobre secuencias de coordenadas de esqueletos 2D.
# Detección de Acciones Humanas en Videos del Dataset UCF101 Usando Esqueletos 2D

## Objetivo

Este proyecto implementa un modelo de Deep Learning para **clasificar acciones humanas** en videos del dataset **UCF101**, utilizando como entrada coordenadas de **esqueletos 2D** extraídas previamente. Se desarrolla una arquitectura basada en **LSTM** que modela la dinámica temporal del movimiento humano.

---

## Arquitectura del Modelo

- Tipo: **LSTM (Long Short-Term Memory)**
- Entrada: Secuencias de esqueletos 2D normalizados, en forma `(frames, 34)`
- Capas:
  - 2 capas LSTM con 64 unidades
  - Dropout (0.5) para regularización
  - Capa final densa con activación `softmax` para clasificación multiclase
- Salida: 5 clases de acciones humanas

---

## Dataset

- **Fuente:** [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- **Representación usada:** Coordenadas 2D de esqueletos por frame en archivo `.pkl` (se proporciona enlace oficial en las instrucciones del curso)
- **Clases seleccionadas (subset):**
  - `ApplyEyeMakeup`
  - `ApplyLipstick`
  - `BrushingTeeth`
  - `BlowDryHair`
  - `HeadMassage`
  
Estas clases fueron elegidas por su similitud visual y gestual, representando un reto real para el modelo.

---

## Pipeline

1. **Carga de datos:**
   - Lectura del archivo `.pkl` con anotaciones de poses 2D
2. **Preprocesamiento:**
   - Filtrado de las 5 clases
   - Normalización de coordenadas `(x, y)` en rango [0, 1]
   - Reformateo a vectores de 34 features por frame
   - Ajuste a longitud fija (300 frames) por padding/truncamiento
3. **División del dataset:**
   - Entrenamiento (70%)
   - Validación (15%)
   - Prueba (15%) — estratificado por clase
4. **Modelo:**
   - Arquitectura LSTM con regularización
   - Compilado con `Adam` y `categorical_crossentropy`
5. **Entrenamiento:**
   - Hasta 50 épocas con `early stopping` (paciencia = 5)
6. **Evaluación:**
   - Precisión sobre conjunto de prueba
   - Matriz de confusión
7. **Predicción:**
   - Se muestran predicciones de ejemplos reales

---

## Baseline

**rat:**
- Prgene

---

## Resultados

**Precisión:**
- Precisión en validación: ~62%
- Precisión en el conjunto de prueba: 63.37%
- Las confusiones se dan mayormente entre acciones similares como `HeadMassage` y `BlowDryHair`

---

## Requisitos del Entorno

Este proyecto está diseñado para ejecutarse en **Google Colab**.

### Librerías necesarias

Instala las siguientes bibliotecas:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
