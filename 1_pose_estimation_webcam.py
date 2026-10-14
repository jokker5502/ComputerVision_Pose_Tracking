"""
=============================================================================
ACTIVIDAD INDIVIDUAL: ESTIMACIÓN DE POSTURA Y RECONOCIMIENTO DE ACTIVIDADES
=============================================================================
Materia: Visión por Computadora
Tecnologías: Python, OpenCV, YOLOv8-Pose (Ultralytics), NumPy

DESCRIPCIÓN DE LA ACTIVIDAD:
Este script implementa un sistema en tiempo real para estimación de pose
humana y clasificación de actividades corporales a partir de los 17 puntos
clave (keypoints) del formato COCO detectados por YOLO-Pose:

Actividades Reconocidas:
1. MANO DERECHA LEVANTADA: Muñeca derecha sobre la altura de la cabeza/nariz.
2. MANO IZQUIERDA LEVANTADA: Muñeca izquierda sobre la altura de la cabeza/nariz.
3. AMBAS MANOS LEVANTADAS: Ambas muñecas simultáneamente elevadas.
4. BRAZOS EN CRUZ (T-POSE): Brazos extendidos horizontalmente a la altura de los hombros.
5. SENTADILLA / SQUAT: Ángulo de las rodillas (cadera-rodilla-tobillo) < 140°.
6. DE PIE / NEUTRAL: Posición erguida estándar en reposo.

APLICACIONES Y RELEVANCIA:
- Ergonomía y corrección postural en puestos de trabajo.
- Monitoreo en tiempo real de ejercicios físicos y conteo de repeticiones de fitness.
- Interacción humano-computador (HCI) sin contacto y sistemas de votación/gestos.
- Detección de caídas y emergencias en adultos mayores o pacientes.
=============================================================================
"""

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO

# Índices estándar de Keypoints COCO (17 puntos)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def calculate_angle(p1, p2, p3):
    """
    Calcula el ángulo (en grados) en el vértice p2 formado por los puntos p1, p2 y p3.
    p1: (x, y) - Punto inicial (ej. Cadera)
    p2: (x, y) - Vértice (ej. Rodilla)
    p3: (x, y) - Punto final (ej. Tobillo)
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 180.0

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)


def classify_activity(kpts_xy, kpts_conf, conf_thresh=0.4):
    """
    Clasifica la actividad corporal analizando las coordenadas (x, y) y
    confianzas de los puntos clave detectados.
    """
    def is_visible(idx):
        return kpts_conf[idx] > conf_thresh

    activity = "DE PIE / NEUTRAL"
    color = (200, 200, 200)  # Gris claro
    details = ""

    # Extraer puntos clave
    has_nose = is_visible(NOSE)
    has_l_wrist = is_visible(LEFT_WRIST)
    has_r_wrist = is_visible(RIGHT_WRIST)
    has_l_shoulder = is_visible(LEFT_SHOULDER)
    has_r_shoulder = is_visible(RIGHT_SHOULDER)
    has_l_elbow = is_visible(LEFT_ELBOW)
    has_r_elbow = is_visible(RIGHT_ELBOW)

    has_l_leg = is_visible(LEFT_HIP) and is_visible(LEFT_KNEE) and is_visible(LEFT_ANKLE)
    has_r_leg = is_visible(RIGHT_HIP) and is_visible(RIGHT_KNEE) and is_visible(RIGHT_ANKLE)

    # 1. Análisis de Sentadilla / Squat (Ángulos de Rodillas)
    left_knee_angle = 180.0
    right_knee_angle = 180.0

    if has_l_leg:
        left_knee_angle = calculate_angle(kpts_xy[LEFT_HIP], kpts_xy[LEFT_KNEE], kpts_xy[LEFT_ANKLE])
    if has_r_leg:
        right_knee_angle = calculate_angle(kpts_xy[RIGHT_HIP], kpts_xy[RIGHT_KNEE], kpts_xy[RIGHT_ANKLE])

    # Condición de sentadilla: rodilla flexionada significativamente (< 140°)
    is_squat = False
    if has_l_leg and has_r_leg:
        if left_knee_angle < 140.0 and right_knee_angle < 140.0:
            is_squat = True
            details = f"Rodillas: I={int(left_knee_angle)}° D={int(right_knee_angle)}°"
    elif has_l_leg and left_knee_angle < 140.0:
        is_squat = True
        details = f"Rodilla Izq: {int(left_knee_angle)}°"
    elif has_r_leg and right_knee_angle < 140.0:
        is_squat = True
        details = f"Rodilla Der: {int(right_knee_angle)}°"

    if is_squat:
        activity = "SENTADILLA / SQUAT"
        color = (0, 140, 255)  # Naranja
        return activity, color, details

    # 2. Análisis de Manos Levantadas
    ref_y = kpts_xy[NOSE][1] if has_nose else (
        (kpts_xy[LEFT_SHOULDER][1] + kpts_xy[RIGHT_SHOULDER][1]) / 2 if (has_l_shoulder and has_r_shoulder) else 0
    )

    left_raised = has_l_wrist and (kpts_xy[LEFT_WRIST][1] < ref_y)
    right_raised = has_r_wrist and (kpts_xy[RIGHT_WRIST][1] < ref_y)

    if left_raised and right_raised:
        activity = "AMBAS MANOS LEVANTADAS"
        color = (0, 255, 0)  # Verde brillante
        details = "Manos por encima de la cabeza"
        return activity, color, details
    elif right_raised:
        activity = "MANO DERECHA LEVANTADA"
        color = (0, 220, 100)  # Verde esmeralda
        details = "Muñeca derecha por encima de referencia"
        return activity, color, details
    elif left_raised:
        activity = "MANO IZQUIERDA LEVANTADA"
        color = (0, 220, 100)
        details = "Muñeca izquierda por encima de referencia"
        return activity, color, details

    # 3. Análisis de Brazos en Cruz (T-Pose)
    if has_l_shoulder and has_r_shoulder and has_l_wrist and has_r_wrist and has_l_elbow and has_r_elbow:
        # Alineación vertical con los hombros
        y_shoulder_avg = (kpts_xy[LEFT_SHOULDER][1] + kpts_xy[RIGHT_SHOULDER][1]) / 2.0
        diff_l = abs(kpts_xy[LEFT_WRIST][1] - y_shoulder_avg)
        diff_r = abs(kpts_xy[RIGHT_WRIST][1] - y_shoulder_avg)

        # Extensión horizontal (brazos separados hacia los costados)
        l_extended = kpts_xy[LEFT_WRIST][0] > kpts_xy[LEFT_SHOULDER][0] or kpts_xy[LEFT_WRIST][0] < kpts_xy[LEFT_SHOULDER][0]
        r_extended = kpts_xy[RIGHT_WRIST][0] > kpts_xy[RIGHT_SHOULDER][0] or kpts_xy[RIGHT_WRIST][0] < kpts_xy[RIGHT_SHOULDER][0]

        shoulder_width = abs(kpts_xy[LEFT_SHOULDER][0] - kpts_xy[RIGHT_SHOULDER][0])

        if diff_l < (shoulder_width * 0.45) and diff_r < (shoulder_width * 0.45):
            activity = "BRAZOS EN CRUZ / T-POSE"
            color = (255, 200, 0)  # Cian / Azul cielo
            details = "Brazos alineados a la horizontal"
            return activity, color, details

    if has_l_leg or has_r_leg:
        details = f"Postura erguida (Rodilla: ~{int(left_knee_angle if has_l_leg else right_knee_angle)}°)"
    else:
        details = "Cuerpo en reposo"

    return activity, color, details


def main():
    parser = argparse.ArgumentParser(description="Reconocimiento de Actividades con YOLO-Pose en Tiempo Real")
    parser.add_argument("--source", type=str, default="0", help="Fuente de video: '0' para webcam o ruta a archivo de video/imagen")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Modelo YOLO-Pose de Ultralytics (ej. yolov8n-pose.pt)")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza del detector")
    args = parser.parse_args()

    # Manejar webcam (entero) o archivo
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 70)
    print(" INICIANDO ESTIMACIÓN DE POSE Y RECONOCIMIENTO DE ACTIVIDADES")
    print(f" Modelo: {args.model}")
    print(f" Fuente: {source}")
    print(" Presiona 'q' para salir | Presiona 's' para guardar captura")
    print("=" * 70)

    # 1. Cargar el modelo YOLO Pose
    model = YOLO(args.model)

    # 2. Iniciar captura de video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video {source}")
        return

    prev_time = time.time()
    screenshot_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o no se pudo leer fotograma.")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30.0
        prev_time = current_time

        # 3. Inferencia de Pose con YOLO
        results = model(frame, conf=args.conf, verbose=False)
        result = results[0]

        # Dibujar esqueleto y cajas anotadas por defecto de Ultralytics
        annotated_frame = result.plot()

        h, w, _ = frame.shape

        # Si se detectaron personas con keypoints
        main_activity = "NINGUNA PERSONA DETECTADA"
        main_color = (120, 120, 120)
        main_details = "Póngase frente a la cámara"

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            num_people = len(result.keypoints.xy)

            # Clasificar la actividad de la persona principal (mayor área/primera detectada)
            kpts_xy = result.keypoints.xy[0].cpu().numpy()
            kpts_conf = result.keypoints.conf[0].cpu().numpy()

            main_activity, main_color, main_details = classify_activity(kpts_xy, kpts_conf)

            # Dibujar ángulo de rodilla si está disponible
            if kpts_conf[LEFT_KNEE] > 0.4 and kpts_conf[LEFT_HIP] > 0.4 and kpts_conf[LEFT_ANKLE] > 0.4:
                ang = calculate_angle(kpts_xy[LEFT_HIP], kpts_xy[LEFT_KNEE], kpts_xy[LEFT_ANKLE])
                pos = (int(kpts_xy[LEFT_KNEE][0]) + 10, int(kpts_xy[LEFT_KNEE][1]))
                cv2.putText(annotated_frame, f"{int(ang)} deg", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 4. Dibujar Banner Superior Informativo (HUD)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

        # Barra lateral indicadora de estado
        cv2.rectangle(annotated_frame, (10, 10), (22, 75), main_color, -1)

        # Textos informativos
        cv2.putText(annotated_frame, f"ACTIVIDAD: {main_activity}", (35, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, main_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Detalle: {main_details} | FPS: {fps:.1f}", (35, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

        # Pie de página con atajos
        cv2.putText(annotated_frame, "[q] Salir | [s] Captura", (w - 220, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Mostrar ventana
        cv2.imshow("Reconocimiento de Actividades con Pose - YOLOv8", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"captura_pose_{screenshot_idx}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"[+] Captura guardada como: {filename}")
            screenshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada correctamente.")


if __name__ == "__main__":
    main()
