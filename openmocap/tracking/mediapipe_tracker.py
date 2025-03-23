"""
Трекер точек на основе MediaPipe.
"""

import cv2
import logging
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from tqdm import tqdm

from openmocap.tracking.base_tracker import BaseTracker
from openmocap.tracking.skeleton_model import SkeletonModel
from openmocap.utils.video_utils import get_video_properties, video_frame_generator


logger = logging.getLogger(__name__)


class MediaPipeTracker(BaseTracker):
    """
    Трекер точек на основе MediaPipe.

    Использует библиотеку MediaPipe для отслеживания точек тела, рук и лица.

    Attributes:
        name (str): Имя трекера
        num_tracked_points (int): Количество отслеживаемых точек
        landmark_names (List[str]): Список имен точек (ориентиров)
        config (Dict): Словарь с конфигурационными параметрами трекера
        model_complexity (int): Сложность модели (0, 1 или 2)
        enable_segmentation (bool): Включение сегментации
        min_detection_confidence (float): Минимальный порог уверенности для обнаружения
        min_tracking_confidence (float): Минимальный порог уверенности для отслеживания
        track_face (bool): Отслеживать точки лица
        track_hands (bool): Отслеживать точки рук
        smooth_landmarks (bool): Сглаживать координаты точек
    """

    # Словарь с именами ориентиров MediaPipe Pose
    POSE_LANDMARK_NAMES = [
        'nose',
        'left_eye_inner',
        'left_eye',
        'left_eye_outer',
        'right_eye_inner',
        'right_eye',
        'right_eye_outer',
        'left_ear',
        'right_ear',
        'mouth_left',
        'mouth_right',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_pinky',
        'right_pinky',
        'left_index',
        'right_index',
        'left_thumb',
        'right_thumb',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        'left_heel',
        'right_heel',
        'left_foot_index',
        'right_foot_index'
    ]

    # Индексы соединений костей MediaPipe Pose
    POSE_CONNECTIONS = [
        # Туловище
        # Соединяем центр бедер с центром плеч (spine)
        (23, 11), (23, 12),  # левое бедро - левое/правое плечо
        (24, 11), (24, 12),  # правое бедро - левое/правое плечо
        (23, 24),  # соединение бедер (pelvis/hips)
        (11, 12),  # соединение плеч (chest)

        # Левая рука
        (11, 13),  # левое плечо - левый локоть (upper_arm.L)
        (13, 15),  # левый локоть - левое запястье (forearm.L)
        # Пальцы левой руки
        (15, 17),  # запястье - мизинец
        (15, 19),  # запястье - указательный
        (15, 21),  # запястье - большой

        # Правая рука
        (12, 14),  # правое плечо - правый локоть (upper_arm.R)
        (14, 16),  # правый локоть - правое запястье (forearm.R)
        # Пальцы правой руки
        (16, 18),  # запястье - мизинец
        (16, 20),  # запястье - указательный
        (16, 22),  # запястье - большой

        # Левая нога
        (23, 25),  # левое бедро - левое колено (thigh.L)
        (25, 27),  # левое колено - левая лодыжка (shin.L)
        (27, 31),  # левая лодыжка - левый носок (foot.L to toes)
        (27, 29),  # левая лодыжка - левая пятка

        # Правая нога
        (24, 26),  # правое бедро - правое колено (thigh.R)
        (26, 28),  # правое колено - правая лодыжка (shin.R)
        (28, 32),  # правая лодыжка - правый носок (foot.R to toes)
        (28, 30),  # правая лодыжка - правая пятка

        # Голова и шея
        # Соединяем центр плеч с носом (для представления шеи и головы)
        (0, 11), (0, 12),  # нос - левое/правое плечо

        # Лицо (опционально, можно убрать для упрощения)
        (0, 1), (1, 2), (2, 3), (3, 7),  # Левая сторона лица
        (0, 4), (4, 5), (5, 6), (6, 8),  # Правая сторона лица
        (9, 10)  # Рот
    ]

    def __init__(
            self,
            model_complexity: int = 1,
            enable_segmentation: bool = False,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
            track_face: bool = True,
            track_hands: bool = True,
            smooth_landmarks: bool = True,
            name: str = "mediapipe_tracker",
            config: Optional[Dict] = None,
    ):
        """
        Инициализирует объект трекера MediaPipe.

        Args:
            model_complexity: Сложность модели (0, 1 или 2)
            enable_segmentation: Включение сегментации
            min_detection_confidence: Минимальный порог уверенности для обнаружения
            min_tracking_confidence: Минимальный порог уверенности для отслеживания
            track_face: Отслеживать точки лица
            track_hands: Отслеживать точки рук
            smooth_landmarks: Сглаживать координаты точек
            name: Имя трекера
            config: Словарь с конфигурационными параметрами
        """
        super().__init__(name=name, config=config or {})

        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.track_face = track_face
        self.track_hands = track_hands
        self.smooth_landmarks = smooth_landmarks

        # Инициализация моделей MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        # Инициализация точек и имен
        self.landmark_names = self.POSE_LANDMARK_NAMES.copy()
        self.num_tracked_points = len(self.landmark_names)

        # Создание экземпляров моделей
        self._initialize_models()

        # Загрузка информации о скелетной модели
        self.skeleton_model = SkeletonModel(
            landmark_names=self.landmark_names,
            connections=self.POSE_CONNECTIONS
        )

        logger.info(f"Инициализирован трекер MediaPipe: {name}")
        logger.debug(f"Количество отслеживаемых точек: {self.num_tracked_points}")

    def _initialize_models(self):
        """
        Инициализирует модели MediaPipe.
        """
        # Модель Pose для отслеживания точек тела
        self.pose = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            smooth_landmarks=self.smooth_landmarks
        )

        # Модели для рук и лица, если включены
        if self.track_hands:
            self.hands = self.mp_hands.Hands(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

        if self.track_face:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

    def track_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Отслеживает точки на одном кадре.

        Args:
            frame: Кадр изображения в формате NumPy array (BGR)

        Returns:
            np.ndarray: Массив координат точек shape (num_tracked_points, 3)
                        с координатами x, y, вероятностью
        """
        # Преобразуем BGR в RGB для MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        # Массив для хранения результатов
        landmarks = np.full((self.num_tracked_points, 3), np.nan)

        # Обработка кадра моделью Pose
        results = self.pose.process(frame_rgb)

        # Если точки тела найдены, извлекаем их координаты
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Координаты нормализованы от 0 до 1, умножаем на размеры кадра
                landmarks[i, 0] = landmark.x * frame_width
                landmarks[i, 1] = landmark.y * frame_height
                landmarks[i, 2] = landmark.visibility

        # TODO: Добавить обработку рук и лица если они включены

        return landmarks

    def track_video(
            self,
            video_path: Union[str, Path],
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1,
            show_progress: bool = True,
            visualize: bool = False,
            output_video_path: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> np.ndarray:
        """
        Отслеживает точки на всех кадрах видео.

        Args:
            video_path: Путь к видеофайлу
            start_frame: Начальный кадр для обработки
            end_frame: Конечный кадр для обработки
            step: Шаг между обрабатываемыми кадрами
            show_progress: Показывать индикатор прогресса
            visualize: Создавать визуализацию отслеживания
            output_video_path: Путь для сохранения видео с визуализацией
            **kwargs: Дополнительные параметры

        Returns:
            np.ndarray: Массив координат точек shape (num_frames, num_tracked_points, 3)
                        с координатами x, y, вероятностью
        """
        video_path = Path(video_path)
        logger.info(f"Отслеживание точек на видео: {video_path}")

        # Получаем свойства видео
        video_props = get_video_properties(video_path)
        frame_count = video_props["frame_count"]
        fps = video_props["fps"]
        frame_width = video_props["width"]
        frame_height = video_props["height"]

        # Определяем диапазон кадров для обработки
        if end_frame is None or end_frame > frame_count:
            end_frame = frame_count

        # Вычисляем количество кадров для обработки
        num_frames = (end_frame - start_frame + step - 1) // step

        # Инициализируем массив для хранения результатов
        landmarks_sequence = np.full((num_frames, self.num_tracked_points, 3), np.nan)

        # Настраиваем видеозапись с визуализацией, если нужно
        video_writer = None
        if visualize and output_video_path:
            output_video_path = Path(output_video_path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (frame_width, frame_height)
            )

        # Итератор для прогресс-бара
        frame_iterator = range(num_frames)
        if show_progress:
            frame_iterator = tqdm(frame_iterator, desc=f"Отслеживание точек на {video_path.name}")

        # Открываем видеофайл
        cap = cv2.VideoCapture(str(video_path))

        try:
            # Перемещаемся к начальному кадру
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Обрабатываем кадры
            frame_idx = 0
            while cap.isOpened() and frame_idx < num_frames:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Не удалось прочитать кадр {start_frame + frame_idx * step}")
                    break

                # Пропускаем кадры в соответствии с шагом
                for _ in range(step - 1):
                    cap.read()

                # Отслеживаем точки на кадре
                landmarks = self.track_frame(frame)
                landmarks_sequence[frame_idx] = landmarks

                # Визуализируем результаты, если нужно
                if visualize:
                    frame_with_landmarks = self._draw_landmarks(frame, landmarks)

                    if video_writer:
                        video_writer.write(frame_with_landmarks)

                    # Показываем кадр (если не в фоновом режиме)
                    if 'background' not in kwargs or not kwargs['background']:
                        cv2.imshow('MediaPipe Tracking', frame_with_landmarks)
                        if cv2.waitKey(1) & 0xFF == 27:  # Выход по Esc
                            break

                frame_idx += 1

                # Обновляем прогресс-бар
                if show_progress:
                    frame_iterator.update(1)

        finally:
            # Освобождаем ресурсы
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        logger.info(f"Обработано {frame_idx} кадров на видео {video_path.name}")

        return landmarks_sequence

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Рисует точки и соединения на кадре.

        Args:
            frame: Исходный кадр
            landmarks: Массив координат точек shape (num_tracked_points, 3)

        Returns:
            np.ndarray: Кадр с нарисованными точками и соединениями
        """
        # Создаем копию кадра для рисования
        vis_frame = frame.copy()

        # Рисуем соединения
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection

            # Проверяем, что обе точки найдены
            if not np.isnan(landmarks[start_idx, 0]) and not np.isnan(landmarks[end_idx, 0]):
                start_point = (int(landmarks[start_idx, 0]), int(landmarks[start_idx, 1]))
                end_point = (int(landmarks[end_idx, 0]), int(landmarks[end_idx, 1]))

                # Рисуем линию
                cv2.line(vis_frame, start_point, end_point, (0, 255, 0), 2)

        # Рисуем точки
        for i, (x, y, conf) in enumerate(landmarks):
            if not np.isnan(x) and not np.isnan(y):
                # Цвет зависит от уверенности (от красного до зеленого)
                if not np.isnan(conf):
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                else:
                    color = (0, 255, 0)

                # Рисуем точку
                cv2.circle(vis_frame, (int(x), int(y)), 4, color, -1)

                # Добавляем номер точки
                cv2.putText(
                    vis_frame,
                    str(i),
                    (int(x) + 5, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return vis_frame

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели трекера.

        Returns:
            Dict[str, Any]: Словарь с информацией о модели трекера
        """
        return {
            "name": self.name,
            "type": "mediapipe",
            "landmark_names": self.landmark_names,
            "num_tracked_points": self.num_tracked_points,
            "connections": self.POSE_CONNECTIONS,
            "model_complexity": self.model_complexity,
            "enable_segmentation": self.enable_segmentation,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "track_face": self.track_face,
            "track_hands": self.track_hands,
            "smooth_landmarks": self.smooth_landmarks
        }

    def __del__(self):
        """
        Освобождает ресурсы при уничтожении объекта.
        """
        if hasattr(self, 'pose'):
            self.pose.close()

        if hasattr(self, 'hands') and self.track_hands:
            self.hands.close()

        if hasattr(self, 'face_mesh') and self.track_face:
            self.face_mesh.close()


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем объект трекера
    tracker = MediaPipeTracker(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Отслеживаем точки на видео
    video_path = r"C:\Users\vczyp\openmocap_data\recordings\recording_20250322_235323\camera_0.mp4"
    if Path(video_path).exists():
        landmarks = tracker.track_video(
            video_path,
            visualize=True,
            output_video_path="output_video.mp4"
        )

        print(f"Tracked landmarks shape: {landmarks.shape}")