"""
=============================================================================
RETO DE TRACKING: MOTION TRAIL + HEATMAP + SMART ENTRY COUNTER
=============================================================================
Materia: Visión por Computadora
Tecnologías: Python, OpenCV, YOLO (Ultralytics) + ByteTrack, NumPy

OBJETIVO DEL RETO:
Construir un sistema analítico de seguimiento multiobjeto sobre el video 'store.mp4'
que implementa tres funcionalidades fundamentales para analítica de video:

1. MOTION TRAIL (TRAYECTORIA DE MOVIMIENTO):
   - Calcula el punto central / base de cada persona detectada.
   - Dibuja la trayectoria temporal del movimiento.
   - Aplica efecto de desvanecimiento (fading) para que los puntos antiguos desaparezcan gradualmente.

2. HEATMAP DE ACTIVIDAD (MAPA DE CALOR):
   - Acumula los puntos espaciales visitados por todas las personas en una matriz 2D.
   - Aplica suavizado Gaussiano y mapa de color COLORMAP_JET.
   - Se superpone sobre el video destacando áreas de alto y bajo tráfico.

3. SMART ENTRY COUNTER (CONTADOR INTELIGENTE DE ENTRADA/SALIDA):
   - Define una línea virtual de control de paso.
   - Cuenta cruces en ambas direcciones (ENTRADA / SALIDA).
   - Previene el doble conteo registrando de forma única los identificadores (Track IDs).
=============================================================================
"""

import cv2
import numpy as np
import time
import argparse
import os
from collections import defaultdict, deque
from ultralytics import YOLO


class SmartTrackingSystem:
    def __init__(self, video_path, output_path=None, model_name="yolov8n.pt", line_y=460, trail_len=30, max_frames=None):
        self.video_path = video_path
        self.output_path = output_path
        self.model_name = model_name
        self.line_y = line_y
        self.trail_len = trail_len
        self.max_frames = max_frames

        # Estructuras de datos para Tracking
        # Historial de posiciones por cada ID: deque de (cx, cy)
        self.track_history = defaultdict(lambda: deque(maxlen=self.trail_len))

        # Contadores y prevención de doble conteo
        self.count_in = 0
        self.count_out = 0
        self.counted_in_ids = set()
        self.counted_out_ids = set()

        # Matriz de acumulación para el Heatmap
        self.heatmap_accum = None

        # Modos de visualización:
        # 0: Overlay completo (Video + BBoxes + Trails + Heatmap + HUD)
        # 1: Pantalla Dividida (Side by Side: Video con Trails a la izquierda | Heatmap puro a la derecha)
        # 2: Solo Video con Trails y Conteo (sin mapa de calor superpuesto)
        self.view_mode = 0

    def process(self):
        print("=" * 75)
        print(" INICIANDO SISTEMA MULTIOBJETO: TRAIL + HEATMAP + SMART COUNTER")
        print(f" Video fuente: {self.video_path}")
        print(f" Modelo detector: {self.model_name} (ByteTrack)")
        print(f" Posición línea virtual (Y): {self.line_y}")
        print(f" Guardar salida en: {self.output_path if self.output_path else 'Solo visualización'}")
        print(" Teclas: [h] Cambiar modo de vista | [q] Salir | [s] Guardar captura")
        print("=" * 75)

        # 1. Cargar modelo YOLO
        print("[+] Cargando modelo YOLO...")
        model = YOLO(self.model_name)

        # 2. Abrir captura de video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[!] Error crítico: No se pudo abrir el archivo de video: {self.video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[i] Propiedades del video: {width}x{height} a {fps:.1f} FPS ({total_video_frames} fotogramas totales)")

        # Inicializar matriz de calor acumulativa (float32 con ceros)
        self.heatmap_accum = np.zeros((height, width), dtype=np.float32)

        # Configurar VideoWriter si se solicitó exportar
        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"[+] Exportando video procesado a: {self.output_path}")

        frame_count = 0
        prev_time = time.time()
        start_process_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[i] Fin del flujo de video.")
                break

            frame_count += 1
            if self.max_frames and frame_count > self.max_frames:
                print(f"\n[i] Límite de {self.max_frames} fotogramas alcanzado.")
                break

            current_time = time.time()
            instant_fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else fps
            prev_time = current_time

            # -------------------------------------------------------------
            # INFERENCIA Y TRACKING CON YOLOV8 + BYTETRACK
            # -------------------------------------------------------------
            # persist=True mantiene los IDs consistentes entre fotogramas
            # classes=[0] filtra únicamente personas (clase 0 del dataset COCO)
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False
            )

            # Capas de visualización
            trails_layer = np.zeros_like(frame)
            active_ids = set()

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_coords = boxes.xyxy.cpu().numpy()

                for track_id, box in zip(track_ids, xyxy_coords):
                    active_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)

                    # Punto base de la persona (pies): óptimo para trayectoria y posición en el suelo
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    # Guardar posición en el historial de trayectorias
                    prev_positions = list(self.track_history[track_id])
                    self.track_history[track_id].append((cx, cy))

                    # ---------------------------------------------------------
                    # TAREA 2: ACUMULACIÓN DE HEATMAP
                    # ---------------------------------------------------------
                    # Sumamos intensidad en la coordenada visitada
                    if 0 <= cx < width and 0 <= cy < height:
                        cv2.circle(self.heatmap_accum, (cx, cy), radius=18, color=1.5, thickness=-1)

                    # ---------------------------------------------------------
                    # TAREA 3: SMART ENTRY COUNTER (LÍNEA VIRTUAL)
                    # ---------------------------------------------------------
                    if len(prev_positions) >= 2:
                        prev_y = prev_positions[-1][1]
                        # Cruce hacia abajo (Entrada / Hacia primer plano)
                        if prev_y < self.line_y <= cy:
                            if track_id not in self.counted_in_ids:
                                self.count_in += 1
                                self.counted_in_ids.add(track_id)

                        # Cruce hacia arriba (Salida / Hacia el fondo)
                        elif prev_y > self.line_y >= cy:
                            if track_id not in self.counted_out_ids:
                                self.count_out += 1
                                self.counted_out_ids.add(track_id)

                    # Dibujar Bounding Box y etiqueta de ID
                    color_id = self._get_id_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 2)
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_id, 2)

            # -------------------------------------------------------------
            # TAREA 1: DIBUJAR MOTION TRAILS CON DESVANECIMIENTO (FADING)
            # -------------------------------------------------------------
            for track_id, points in self.track_history.items():
                if len(points) < 2:
                    continue
                color_id = self._get_id_color(track_id)
                num_points = len(points)

                for i in range(1, num_points):
                    pt1 = points[i - 1]
                    pt2 = points[i]

                    # El grosor e intensidad aumentan con los puntos más recientes (desvanecimiento gradual)
                    fraction = i / float(num_points)
                    thickness = max(1, int(4 * fraction))

                    # Color desvanecido (alfa mediante escala de valor RGB)
                    faded_color = (
                        int(color_id[0] * (0.3 + 0.7 * fraction)),
                        int(color_id[1] * (0.3 + 0.7 * fraction)),
                        int(color_id[2] * (0.3 + 0.7 * fraction))
                    )
                    cv2.line(frame, pt1, pt2, faded_color, thickness, cv2.LINE_AA)

                # Punto brillante en la posición actual
                cv2.circle(frame, points[-1], 5, color_id, -1)

            # -------------------------------------------------------------
            # TAREA 2: PROCESAMIENTO Y GENERACIÓN DEL HEATMAP
            # -------------------------------------------------------------
            # Aplicar filtro Gaussiano para difuminar los puntos acumulados
            blurred_heatmap = cv2.GaussianBlur(self.heatmap_accum, (35, 35), 0)
            norm_heatmap = cv2.normalize(blurred_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

            # -------------------------------------------------------------
            # DIBUJAR LÍNEA VIRTUAL DE CONTEO
            # -------------------------------------------------------------
            line_color = (0, 255, 255)  # Amarillo
            cv2.line(frame, (80, self.line_y), (width - 80, self.line_y), line_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "LINEA DE CONTEO (CRUCE)", (90, self.line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "v ENTRADA v", (90, self.line_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "^ SALIDA ^", (width - 240, self.line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)

            # -------------------------------------------------------------
            # COMBINAR CON MAPA DE CALOR SEGÚN MODO DE VISTA
            # -------------------------------------------------------------
            if self.view_mode == 0:
                # Superposición con transparencia en zonas transitadas (norm_heatmap > 15)
                mask = norm_heatmap > 15
                blended = frame.copy()
                blended[mask] = cv2.addWeighted(frame, 0.60, color_heatmap, 0.40, 0)[mask]
                display_frame = blended
            elif self.view_mode == 1:
                # Pantalla dividida (Izquierda: Video + Trails | Derecha: Mapa de Calor)
                half_w = width // 2
                left_view = cv2.resize(frame, (half_w, height))
                right_view = cv2.resize(color_heatmap, (half_w, height))
                display_frame = np.hstack((left_view, right_view))
            else:
                display_frame = frame

            # -------------------------------------------------------------
            # DIBUJAR HUD (DASHBOARD INFORMATIVO SUPERIOR)
            # -------------------------------------------------------------
            disp_h, disp_w = display_frame.shape[:2]
            hud_overlay = display_frame.copy()
            cv2.rectangle(hud_overlay, (0, 0), (disp_w, 80), (15, 15, 15), -1)
            cv2.addWeighted(hud_overlay, 0.78, display_frame, 0.22, 0, display_frame)

            # Indicadores de Conteo y Analítica
            cv2.putText(display_frame, f"ENTRADA (IN): {self.count_in}", (25, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"SALIDA (OUT): {self.count_out}", (280, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"ACTIVOS: {len(active_ids)}", (520, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"FPS: {instant_fps:.1f}", (720, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

            # Subtítulo explicativo
            mode_names = ["Overlay Video + Heatmap", "Pantalla Dividida", "Solo Video + Trails"]
            cv2.putText(display_frame,
                        f"Modo: {mode_names[self.view_mode]} | Frame: {frame_count}/{total_video_frames} | [h] Cambiar vista | [q] Salir",
                        (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

            # Escribir frame al video de salida si está habilitado
            if writer:
                # Asegurar tamaño consistente para el writer
                if display_frame.shape[:2] != (height, width):
                    write_frame = cv2.resize(display_frame, (width, height))
                else:
                    write_frame = display_frame
                writer.write(write_frame)

            # Mostrar ventana en vivo
            cv2.imshow("Tracking Inteligente: Motion Trail + Heatmap + Smart Counter", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[i] Detención solicitada por el usuario.")
                break
            elif key == ord('h'):
                self.view_mode = (self.view_mode + 1) % 3
                print(f"[i] Modo de visualización cambiado a: {mode_names[self.view_mode]}")
            elif key == ord('s'):
                screenshot_name = f"captura_tracking_frame_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"[+] Captura guardada como: {screenshot_name}")

            if frame_count % 60 == 0:
                elapsed = time.time() - start_process_time
                print(f"Progreso: Frame {frame_count}/{total_video_frames} | IN: {self.count_in} | OUT: {self.count_out} | Tiempo: {elapsed:.1f}s", end="\r")

        cap.release()
        if writer:
            writer.release()
            print(f"\n[+] Video de salida completado con éxito: {self.output_path}")
        cv2.destroyAllWindows()

        print("\n" + "=" * 75)
        print(" RESUMEN FINAL DE LA EJECUCIÓN:")
        print(f" Total Fotogramas Procesados: {frame_count}")
        print(f" Total Personas que Entraron (IN): {self.count_in}")
        print(f" Total Personas que Salieron (OUT): {self.count_out}")
        print(f" Total Personas Únicas Rastreadas: {len(self.track_history)}")
        print("=" * 75)

    def _get_id_color(self, track_id):
        """Genera un color pseudoaleatorio consistente y distinguible para cada ID."""
        np.random.seed(int(track_id) * 37)
        color = np.random.randint(60, 255, size=3).tolist()
        return tuple(color)


def main():
    parser = argparse.ArgumentParser(description="Reto de Tracking: Motion Trail + Heatmap + Smart Counter")
    parser.add_argument("--video", type=str, default="store.mp4", help="Ruta al video de entrada (store.mp4)")
    parser.add_argument("--output", type=str, default="store_output_analytics.mp4", help="Ruta para exportar el video resultante")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLO preentrenado")
    parser.add_argument("--line_y", type=int, default=460, help="Posición en eje Y de la línea virtual de conteo")
    parser.add_argument("--max_frames", type=int, default=None, help="Límite máximo de fotogramas a procesar (None para todo el video)")
    args = parser.parse_args()

    tracker = SmartTrackingSystem(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model,
        line_y=args.line_y,
        max_frames=args.max_frames
    )
    tracker.process()


if __name__ == "__main__":
    main()
