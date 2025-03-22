"""
Модуль для синхронной записи видео с нескольких веб-камер.

Предоставляет функции для одновременного захвата видео
с нескольких веб-камер и сохранения результатов.
"""

import cv2
import time
import threading
import queue
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class CameraDevice:
    """
    Класс для работы с отдельной камерой.

    Attributes:
        camera_id (int): Идентификатор камеры
        name (str): Имя камеры
        capture (cv2.VideoCapture): Объект захвата OpenCV
        width (int): Ширина кадра
        height (int): Высота кадра
        fps (float): Частота кадров
    """

    def __init__(
            self,
            camera_id: int,
            width: int = 1280,
            height: int = 720,
            fps: float = 30.0,
            name: Optional[str] = None
    ):
        """
        Инициализирует устройство камеры.

        Args:
            camera_id: Идентификатор камеры (обычно 0, 1, ...)
            width: Желаемая ширина кадра
            height: Желаемая высота кадра
            fps: Желаемая частота кадров
            name: Имя камеры. Если None, используется "camera_{camera_id}"
        """
        self.camera_id = camera_id
        self.name = name or f"camera_{camera_id}"
        self.capture = None
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        self.last_frame = None
        self.last_timestamp = 0

    def open(self) -> bool:
        """
        Открывает камеру.

        Returns:
            bool: True если камера успешно открыта, иначе False
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_id)

            # Установка разрешения и FPS
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

            # Проверка успешного открытия
            if not self.capture.isOpened():
                logger.error(f"Не удалось открыть камеру {self.name} (ID: {self.camera_id})")
                return False

            # Получение фактических параметров
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Камера {self.name} (ID: {self.camera_id}) успешно открыта. "
                f"Разрешение: {self.width}x{self.height}, FPS: {self.fps}"
            )
            return True
        except Exception as e:
            logger.error(f"Ошибка при открытии камеры {self.name} (ID: {self.camera_id}): {e}")
            return False

    def close(self) -> None:
        """
        Закрывает камеру.
        """
        if self.capture is not None:
            self.capture.release()
            logger.info(f"Камера {self.name} (ID: {self.camera_id}) закрыта")
        self.capture = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Считывает кадр с камеры.

        Returns:
            Tuple[bool, Optional[np.ndarray], float]:
                - успех чтения
                - кадр (или None если чтение не удалось)
                - временная метка
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None, 0

        ret, frame = self.capture.read()
        timestamp = time.time()

        if ret:
            self.last_frame = frame
            self.last_timestamp = timestamp

        return ret, frame, timestamp

    def get_camera_info(self) -> Dict:
        """
        Возвращает информацию о камере.

        Returns:
            Dict: Словарь с информацией о камере
        """
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps
        }


class MultiCameraRecorder:
    """
    Класс для одновременной записи с нескольких камер.

    Attributes:
        cameras (List[CameraDevice]): Список устройств камер
        recording (bool): Флаг записи
        output_dir (Path): Директория для сохранения видео
    """

    def __init__(
            self,
            camera_ids: List[int],
            output_dir: Union[str, Path],
            width: int = 1280,
            height: int = 720,
            fps: float = 30.0,
            camera_names: Optional[List[str]] = None
    ):
        """
        Инициализирует рекордер для нескольких камер.

        Args:
            camera_ids: Список идентификаторов камер
            output_dir: Директория для сохранения видео
            width: Желаемая ширина кадра
            height: Желаемая высота кадра
            fps: Желаемая частота кадров
            camera_names: Список имен камер. Если None, используются автоматические имена
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Создаем объекты камер
        self.cameras = []
        for i, cam_id in enumerate(camera_ids):
            name = camera_names[i] if camera_names and i < len(camera_names) else None
            camera = CameraDevice(cam_id, width, height, fps, name)
            self.cameras.append(camera)

        self.recording = False
        self.writers = {}
        self.frame_queues = {}
        self.recording_threads = {}
        self.frame_counts = {}
        self.timestamp_files = {}
        self.stop_event = threading.Event()

    def open_cameras(self) -> bool:
        """
        Открывает все камеры.

        Returns:
            bool: True если все камеры успешно открыты, иначе False
        """
        success = True
        for camera in self.cameras:
            if not camera.open():
                success = False

        return success

    def close_cameras(self) -> None:
        """
        Закрывает все камеры.
        """
        for camera in self.cameras:
            camera.close()

    def camera_recording_thread(self, camera: CameraDevice, video_path: str) -> None:
        """
        Поток для записи с отдельной камеры.

        Args:
            camera: Объект камеры
            video_path: Путь для сохранения видео
        """
        queue = self.frame_queues[camera.name]
        writer = self.writers[camera.name]
        timestamp_file = self.timestamp_files[camera.name]
        frame_count = 0

        logger.info(f"Запущен поток записи для камеры {camera.name}")

        # Записываем заголовок для файла временных меток
        with open(timestamp_file, 'w') as f:
            f.write("frame_number,timestamp\n")

        while not self.stop_event.is_set():
            try:
                frame, timestamp = queue.get(timeout=0.1)
                writer.write(frame)

                # Записываем временную метку
                with open(timestamp_file, 'a') as f:
                    f.write(f"{frame_count},{timestamp}\n")

                frame_count += 1
                self.frame_counts[camera.name] = frame_count

                queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в потоке записи камеры {camera.name}: {e}")
                break

        logger.info(f"Поток записи для камеры {camera.name} завершен. Записано {frame_count} кадров.")

    def start_recording(self, session_name: Optional[str] = None) -> Dict[str, str]:
        """
        Начинает запись со всех камер.

        Args:
            session_name: Имя сессии для именования файлов

        Returns:
            Dict[str, str]: Словарь с путями к сохраненным видеофайлам
        """
        if self.recording:
            logger.warning("Запись уже запущена")
            return {}

        if not session_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"recording_{timestamp}"

        session_dir = self.output_dir / session_name
        session_dir.mkdir(exist_ok=True)

        # Проверяем, что все камеры открыты
        for camera in self.cameras:
            if camera.capture is None or not camera.capture.isOpened():
                if not camera.open():
                    logger.error(f"Не удалось открыть камеру {camera.name} для записи")
                    return {}

        # Создаем записывающие устройства и очереди для каждой камеры
        video_paths = {}
        for camera in self.cameras:
            video_path = str(session_dir / f"{camera.name}.mp4")
            timestamp_path = str(session_dir / f"{camera.name}_timestamps.csv")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                camera.fps,
                (camera.width, camera.height)
            )

            if not writer.isOpened():
                logger.error(f"Не удалось создать видеофайл для камеры {camera.name}")
                # Освобождаем уже созданные ресурсы
                for w in self.writers.values():
                    w.release()
                self.writers.clear()
                return {}

            self.writers[camera.name] = writer
            self.frame_queues[camera.name] = queue.Queue()
            self.timestamp_files[camera.name] = timestamp_path
            self.frame_counts[camera.name] = 0

            # Запускаем поток записи
            recording_thread = threading.Thread(
                target=self.camera_recording_thread,
                args=(camera, video_path),
                daemon=True
            )
            self.recording_threads[camera.name] = recording_thread
            recording_thread.start()

            video_paths[camera.name] = video_path

        self.recording = True
        self.stop_event.clear()

        logger.info(f"Запись начата. Видеофайлы будут сохранены в {session_dir}")

        return video_paths

    def capture_frames(self) -> None:
        """
        Захватывает кадры со всех камер и добавляет их в очереди.
        """
        for camera in self.cameras:
            ret, frame, timestamp = camera.read_frame()
            if ret and self.recording:
                self.frame_queues[camera.name].put((frame, timestamp))

    def stop_recording(self) -> Dict[str, int]:
        """
        Останавливает запись со всех камер.

        Returns:
            Dict[str, int]: Словарь с количеством записанных кадров для каждой камеры
        """
        if not self.recording:
            logger.warning("Запись не запущена")
            return {}

        # Сигнализируем потокам записи о необходимости завершения
        self.stop_event.set()

        # Ждем завершения потоков
        for name, thread in self.recording_threads.items():
            thread.join(timeout=2.0)
            logger.info(f"Поток записи для камеры {name} завершен")

        # Освобождаем ресурсы
        for writer in self.writers.values():
            writer.release()

        self.writers.clear()
        self.frame_queues.clear()
        self.recording_threads.clear()

        self.recording = False

        # Возвращаем количество записанных кадров
        return self.frame_counts.copy()

    def preview_cameras(self, window_name: str = "Camera Preview", max_fps: float = 30.0) -> None:
        """
        Отображает предпросмотр камер в реальном времени.

        Args:
            window_name: Имя окна предпросмотра
            max_fps: Максимальная частота кадров для предпросмотра
        """
        if not all(camera.capture is not None and camera.capture.isOpened() for camera in self.cameras):
            if not self.open_cameras():
                logger.error("Не удалось открыть камеры для предпросмотра")
                return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                start_time = time.time()

                # Считываем кадры со всех камер
                frames = []
                for camera in self.cameras:
                    ret, frame, _ = camera.read_frame()
                    if ret:
                        # Добавляем метку с именем камеры
                        cv2.putText(
                            frame,
                            camera.name,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                        frames.append(frame)

                if not frames:
                    logger.error("Не удалось получить кадры с камер")
                    break

                # Создаем составное изображение
                if len(frames) == 1:
                    display_frame = frames[0]
                elif len(frames) == 2:
                    # Располагаем кадры горизонтально
                    display_frame = np.hstack(frames)
                else:
                    # Для более чем двух камер нужно создать сетку
                    rows = int(np.ceil(len(frames) / 2))
                    display_frame = None
                    for row in range(rows):
                        row_frames = frames[row * 2:min((row + 1) * 2, len(frames))]
                        if len(row_frames) == 1:
                            row_frames.append(np.zeros_like(row_frames[0]))
                        row_combined = np.hstack(row_frames)
                        if display_frame is None:
                            display_frame = row_combined
                        else:
                            display_frame = np.vstack((display_frame, row_combined))

                # Показываем составное изображение
                cv2.imshow(window_name, display_frame)

                # Если запись активна, добавляем кадры в очереди
                if self.recording:
                    self.capture_frames()

                # Проверяем нажатие клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q или Esc
                    break
                elif key == ord('r'):  # r - начать/остановить запись
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()

                # Ограничение FPS
                elapsed = time.time() - start_time
                sleep_time = max(1.0 / max_fps - elapsed, 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cv2.destroyWindow(window_name)

    def record_for_duration(self, duration_seconds: float, session_name: Optional[str] = None) -> Dict[str, str]:
        """
        Записывает видео с камер в течение указанного времени.

        Args:
            duration_seconds: Длительность записи в секундах
            session_name: Имя сессии для именования файлов

        Returns:
            Dict[str, str]: Словарь с путями к сохраненным видеофайлам
        """
        if not all(camera.capture is not None and camera.capture.isOpened() for camera in self.cameras):
            if not self.open_cameras():
                logger.error("Не удалось открыть камеры для записи")
                return {}

        # Начинаем запись
        video_paths = self.start_recording(session_name)
        if not video_paths:
            return {}

        start_time = time.time()
        end_time = start_time + duration_seconds

        try:
            while time.time() < end_time:
                self.capture_frames()
                remaining = end_time - time.time()
                logger.info(f"Запись: осталось {remaining:.1f} секунд")
                time.sleep(0.01)  # Небольшая задержка для снижения нагрузки на процессор

        finally:
            # Останавливаем запись
            frame_counts = self.stop_recording()
            logger.info(f"Запись завершена. Количество кадров: {frame_counts}")

        return video_paths


def list_available_cameras(max_id: int = 10) -> List[int]:
    """
    Находит доступные камеры в системе.

    Args:
        max_id: Максимальный ID для проверки

    Returns:
        List[int]: Список ID доступных камер
    """
    available_cameras = []

    for camera_id in range(max_id):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available_cameras.append(camera_id)
            cap.release()

    return available_cameras


if __name__ == "__main__":
    from openmocap.utils.logger import configure_logging, LogLevel

    # Настраиваем логирование
    configure_logging(LogLevel.INFO)

    # Находим доступные камеры
    cameras = list_available_cameras()
    if not cameras:
        logger.error("Камеры не найдены")
        exit(1)

    logger.info(f"Найдены камеры с ID: {cameras}")

    # Создаем директорию для записи
    output_dir = Path.home() / "openmocap_data" / "recordings"

    # Создаем рекордер
    recorder = MultiCameraRecorder(
        camera_ids=cameras,
        output_dir=output_dir,
        width=1280,
        height=720,
        fps=30.0
    )

    # Показываем предпросмотр (нажмите 'r' для начала/остановки записи, 'q' для выхода)
    logger.info("Запуск предпросмотра. Нажмите 'r' для начала/остановки записи, 'q' для выхода")
    recorder.preview_cameras()

    # Освобождаем ресурсы
    recorder.close_cameras()