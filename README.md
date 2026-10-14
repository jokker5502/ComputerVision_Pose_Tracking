# Proyecto: Estimación de Postura y Analítica de Video con YOLO (Tracking)

**Materia:** Visión por Computadora  
**Autor:** Estudiante de Visión por Computadora  
**Tecnologías:** Python 3.10+, OpenCV, YOLOv8 (Ultralytics), ByteTrack, NumPy, Matplotlib  

Este repositorio contiene la solución completa para las dos prácticas y retos solicitados:
1. **Parte 1:** Estimación de Pose Humana (COCO 17 Keypoints) y Reconocimiento de Actividades Posturales en Tiempo Real con Cámara Web.
2. **Parte 2:** Reto de Seguimiento Multiobjeto (*Multi-Object Tracking*) sobre el video `store.mp4` implementando **Motion Trail** (trayectorias con desvanecimiento), **Heatmap de Actividad** (mapa de calor térmico) y **Smart Entry Counter** (contador inteligente de entradas y salidas por línea virtual sin duplicados).

---

## Estructura del Repositorio

```text
Tarea_Pose_y_Tracking/
│
├── 1_pose_estimation.ipynb           # Notebook interactivo de la Parte 1 (Pose Estimation + Actividad)
├── 1_pose_estimation_webcam.py       # Script ejecutable en tiempo real con cámara web para la presentación
│
├── 2_tracking_analytics.ipynb        # Notebook interactivo de la Parte 2 (Reto de Tracking completo)
├── 2_tracking_analytics.py           # Script ejecutable para procesar 'store.mp4' con Trails, Heatmap y Conteo
│
├── store.mp4                         # Video de prueba proporcionado para el reto de analítica
├── store_output_analytics.mp4        # Video generado con las tres analíticas en acción
│
├── zuck_running.jpg                  # Imagen de muestra para validación de pose
├── body_coco.png                     # Diagrama de referencia de los 17 puntos clave de COCO
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore                        # Exclusiones de archivos pesados y cachés
└── README.md                         # Documentación técnica y guía de sustentación oral
```

---

## 🛠️ Instalación y Configuración del Entorno

1. **Clonar o descargar este repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd Tarea_Pose_y_Tracking
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   * En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * En Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar las dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧍 PARTE 1: Estimación de Postura y Reconocimiento de Actividades (5 Puntos)

### 1. Justificación y Relevancia de la Actividad (1 pt)
Para la actividad individual se implementó un **Sistema Inteligente de Clasificación de Posturas y Gestos Corporales** en tiempo real capaz de distinguir:
* **Mano Derecha / Izquierda / Ambas Levantadas:** Identifica cuando las muñecas superan la altura de la cabeza (nariz) o los hombros.  
  * *Relevancia:* Aplicación en aulas inteligentes para pedir la palabra sin periféricos, interacción humano-computador (HCI) sin contacto y sistemas de votación rápida.
* **Brazos en Cruz (T-Pose):** Detecta cuando los brazos se encuentran extendidos horizontalmente alineados con el eje de los hombros.  
  * *Relevancia:* Calibración de captura de movimiento (*mocap*), sesiones de fisioterapia guiada y control por gestos en simuladores.
* **Sentadilla / Flexión de Rodillas (Squat):** Calcula en tiempo real el ángulo entre la cadera, la rodilla y el tobillo. Cuando el ángulo desciende por debajo de 140°, clasifica la acción como sentadilla.  
  * *Relevancia:* Asistente virtual de entrenamiento deportivo, corrección de técnica en gimnasios y prevención de lesiones osteomusculares.
* **De Pie / Reposo:** Estado erguido neutral de referencia.

### 2. Explicación del Código Línea por Línea (2 pts)
El núcleo del cálculo geométrico y de clasificación se encuentra en `1_pose_estimation_webcam.py` y `1_pose_estimation.ipynb`:

```python
def calculate_angle(p1, p2, p3):
    # 1. Creamos los vectores directores v1 (vértice a p1) y v2 (vértice a p3)
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)

    # 2. Obtenemos las normas (módulos) de cada vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 180.0

    # 3. Calculamos el producto punto y el coseno del ángulo theta
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Evita imprecisiones numéricas fuera de [-1, 1]

    # 4. Obtenemos el ángulo en grados mediante el arcocoseno
    return float(np.degrees(np.arccos(cosine_angle)))
```

Lógica de decisión:
* `wrist[1] < nose[1]`: En coordenadas de imagen de OpenCV, el eje Y crece hacia abajo. Por lo tanto, un valor de Y menor indica una posición más alta en el espacio vertical.
* `angle(hip, knee, ankle) < 140°`: Indica flexión articular suficiente para registrar una sentadilla o posición sentada.

### 3. Ejecución de la Demostración en Vivo con Webcam (2 pts)
Para ejecutar la presentación en vivo ante el docente:
```bash
python 1_pose_estimation_webcam.py --source 0
```
* **Controles:**
  * Presiona `q` para cerrar la ventana.
  * Presiona `s` para capturar una imagen de evidencia con la postura detectada.

---

## 📹 PARTE 2: Reto de Tracking en Video (`store.mp4`) (10 Puntos)

El reto consiste en aplicar **Multi-Object Tracking (MOT)** combinando la velocidad de detección de **YOLOv8** con la asociación temporal de **ByteTrack** sobre `store.mp4`.

### Componentes Implementados y Rúbrica:

#### 1. Motion Trail (Trayectoria con Desvanecimiento) (2 pts)
* **Punto de anclaje:** Se calcula el punto inferior central de cada caja delimitadora `cx = (x1 + x2) // 2`, `cy = y2` (posición de los pies en el plano del suelo de la tienda).
* **Historial temporal:** Se almacena una cola circular `deque(maxlen=30)` por cada `track_id`.
* **Efecto Fading:** Al dibujar las líneas entre puntos consecutivos, el grosor (`thickness`) y la luminosidad del color aumentan proporcionalmente a la cercanía con el fotograma actual (`fraction = i / num_points`). Los puntos antiguos se desvanecen suavemente.

#### 2. Heatmap de Actividad (Mapa de Calor Térmico) (2 pts)
* **Matriz de Acumulación 2D:** Se crea una matriz de ceros del mismo tamaño del fotograma (`height x width`, tipo `float32`).
* **Acumulación de densidad:** En cada fotograma, cada persona detectada suma intensidad en su coordenada mediante `cv2.circle(heatmap_accum, (cx, cy), radius=18, color=1.5, thickness=-1)`.
* **Filtro Gaussiano y Colormap:** Se aplica `cv2.GaussianBlur` para suavizar la densidad térmica y se mapea con `cv2.applyColorMap(norm, cv2.COLORMAP_JET)`.
* **Fusión Alfa:** Las áreas de alta densidad se superponen sobre el video original respetando la transparencia, permitiendo ver a la vez el video y los puntos calientes de tráfico del pasillo.

#### 3. Smart Entry Counter (Contador Inteligente) (2 pts)
* **Línea Virtual de Referencia:** Se define una línea horizontal en `y = 460` que cruza el pasillo de la tienda.
* **Vector de Movimiento:** Se compara la posición anterior (`prev_y`) con la actual (`cy`):
  * **ENTRADA (IN):** `prev_y < line_y <= cy` (la persona avanza hacia el primer plano / entra a la tienda).
  * **SALIDA (OUT):** `prev_y > line_y >= cy` (la persona camina hacia el fondo / sale de la tienda).
* **Prevención de Doble Conteo:** Se utilizan conjuntos `set()` de identificadores (`counted_in_ids` y `counted_out_ids`). Cada ID único solo puede contabilizarse una vez en cada dirección.

#### 4. Sustentación Oral y Explicación Lógica (4 pts)
La arquitectura está modularizada y documentada para facilitar la exposición oral de 3 a 5 minutos requerida en la rúbrica.

### Ejecución del Script de Tracking:
```bash
python 2_tracking_analytics.py --video store.mp4 --output store_output_analytics.mp4
```

**Atajos de teclado durante la reproducción interactiva:**
* `h`: Alterna entre modos de visualización:
  * *Modo 0:* Video con BBoxes + Trails + Heatmap superpuesto + HUD.
  * *Modo 1:* Pantalla dividida (*Side-by-Side*: Video a la izquierda y Heatmap térmico a la derecha).
  * *Modo 2:* Video limpio con Trails y Conteo.
* `s`: Guarda una captura de pantalla del fotograma y analíticas en curso.
* `q`: Detiene la ejecución de forma segura y cierra el archivo de video.

---

## 🚀 Cómo Subir este Proyecto a tu Repositorio de GitHub

Sigue estos sencillos pasos desde la terminal dentro de esta carpeta para subir tu tarea a GitHub:

1. **Crea un nuevo repositorio en GitHub:**
   * Entra a [github.com/new](https://github.com/new).
   * Asígnale un nombre (por ejemplo: `ComputerVision_Pose_Tracking`).
   * Déjalo **Público** y **NO marques** la opción de inicializar con README ni con .gitignore (ya los tenemos creados aquí).
   * Haz clic en **Create repository**.

2. **Inicializa Git y sube tus archivos:**
   Abre una terminal en esta carpeta (`Tarea_Pose_y_Tracking`) y ejecuta:

   ```bash
   # 1. Inicializar repositorio local
   git init

   # 2. Agregar todos los archivos
   git add .

   # 3. Crear el primer commit
   git commit -m "Entrega: Estimacion de Pose y Analitica de Video Tracking con YOLO"

   # 4. Renombrar la rama principal a main
   git branch -M main

   # 5. Vincular con tu repositorio remoto de GitHub (reemplaza con tu URL real)
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git

   # 6. Subir los archivos
   git push -u origin main
   ```

3. **Copia el link de tu repositorio:**  
   ¡Listo! Copia la URL de tu repositorio (`https://github.com/TU_USUARIO/TU_REPOSITORIO`) y envíala a la plataforma de entrega de tu universidad.
