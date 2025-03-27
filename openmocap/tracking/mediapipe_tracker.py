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
    SKELETON_LANDMARK_NAMES = [
        # Центральная линия (позвоночник)
        'pelvis',  # 0 - таз (центр бедер)
        'spine',  # 1 - нижняя часть позвоночника
        'spine.001',  # 2 - позвоночник (лумбарный)
        'spine.002',  # 3 - позвоночник (грудной)
        'spine.003',  # 4 - позвоночник (верхний грудной)
        'spine.004',  # 5 - позвоночник (нижний шейный)
        'spine.005',  # 6 - шея
        'spine.006',  # 7 - голова

        # Левая рука
        'shoulder.L',  # 8 - левое плечо
        'upper_arm.L',  # 9 - левое предплечье
        'forearm.L',  # 10 - левое предплечье
        'hand.L',  # 11 - левая кисть

        # Правая рука
        'shoulder.R',  # 12 - правое плечо
        'upper_arm.R',  # 13 - правое предплечье
        'forearm.R',  # 14 - правое предплечье
        'hand.R',  # 15 - правая кисть

        # Левая нога
        'thigh.L',  # 16 - левое бедро
        'shin.L',  # 17 - левая голень
        'foot.L',  # 18 - левая стопа
        'heel.L',  # 19 - левая пятка
        'toe.L',  # 20 - левые пальцы ноги

        # Правая нога
        'thigh.R',  # 21 - правое бедро
        'shin.R',  # 22 - правая голень
        'foot.R',  # 23 - правая стопа
        'heel.R',  # 24 - правая пятка
        'toe.R'  # 25 - правые пальцы ноги
    ]

    # Соединения между точками (определение костей)
    SKELETON_CONNECTIONS = [
        # Позвоночник
        (0, 1),  # pelvis -> spine
        (1, 2),  # spine -> spine.001
        (2, 3),  # spine.001 -> spine.002
        (3, 4),  # spine.002 -> spine.003
        (4, 5),  # spine.003 -> spine.004
        (5, 6),  # spine.004 -> spine.005 (шея)
        (6, 7),  # spine.005 -> spine.006 (голова)

        # Левая рука
        (4, 8),  # spine.003 -> shoulder.L
        (8, 9),  # shoulder.L -> upper_arm.L
        (9, 10),  # upper_arm.L -> forearm.L
        (10, 11),  # forearm.L -> hand.L

        # Правая рука
        (4, 12),  # spine.003 -> shoulder.R
        (12, 13),  # shoulder.R -> upper_arm.R
        (13, 14),  # upper_arm.R -> forearm.R
        (14, 15),  # forearm.R -> hand.R

        # Левая нога
        (0, 16),  # pelvis -> thigh.L
        (16, 17),  # thigh.L -> shin.L
        (17, 18),  # shin.L -> foot.L
        (18, 19),  # foot.L -> heel.L
        (18, 20),  # foot.L -> toe.L

        # Правая нога
        (0, 21),  # pelvis -> thigh.R
        (21, 22),  # thigh.R -> shin.R
        (22, 23),  # shin.R -> foot.R
        (23, 24),  # foot.R -> heel.R
        (23, 25)  # foot.R -> toe.R
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
            SKELETON_LANDMARK_NAMES=SKELETON_LANDMARK_NAMES,
            SKELETON_CONNECTIONS=SKELETON_CONNECTIONS):
        """
        Инициализирует объект трекера MediaPipe.

        Args:
            ...
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

        # Инициализация точек и имен для скелета Blender
        self.landmark_names = SKELETON_LANDMARK_NAMES
        self.num_tracked_points = len(self.landmark_names)

        # Создание экземпляров моделей
        self._initialize_models()

        # Загрузка информации о скелетной модели
        self.skeleton_model = SkeletonModel(
            landmark_names=self.landmark_names,
            connections=SKELETON_CONNECTIONS
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

    def map_mediapipe_to_blender_skeleton(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Конвертирует 33 точки MediaPipe в 26 точек скелета Blender.

        Args:
            landmarks: Массив точек MediaPipe формы (33, 3)

        Returns:
            np.ndarray: Массив точек Blender-скелета формы (26, 3)
        """
        # Создаем массив для новых точек
        blender_points = np.full((26, 3), np.nan)

        # Для упрощения доступа
        mp = landmarks  # MediaPipe точки

        # Таз (центр между бедрами)
        if not (np.isnan(mp[23]).any() or np.isnan(mp[24]).any()):
            blender_points[0] = (mp[23] + mp[24]) / 2  # pelvis

        # Позвоночник
        if not (np.isnan(mp[23]).any() or np.isnan(mp[24]).any() or np.isnan(mp[11]).any() or np.isnan(mp[12]).any()):
            pelvis = (mp[23] + mp[24]) / 2
            chest = (mp[11] + mp[12]) / 2
            spine_vector = chest - pelvis

            # Основной позвоночник
            blender_points[1] = pelvis + spine_vector * 0.2  # spine
            blender_points[2] = pelvis + spine_vector * 0.4  # spine.001
            blender_points[3] = pelvis + spine_vector * 0.6  # spine.002
            blender_points[4] = pelvis + spine_vector * 0.8  # spine.003
            blender_points[5] = chest  # spine.004

        # Шея и голова
        if not (np.isnan(mp[0]).any() or np.isnan(mp[11]).any() or np.isnan(mp[12]).any()):
            chest = (mp[11] + mp[12]) / 2
            head_vector = mp[0] - chest

            blender_points[6] = chest + head_vector * 0.3  # spine.005 (шея)
            blender_points[7] = mp[0]  # spine.006 (голова)

        # Плечи
        blender_points[8] = mp[11]  # shoulder.L
        blender_points[12] = mp[12]  # shoulder.R

        # Руки
        blender_points[9] = mp[13]  # upper_arm.L
        blender_points[10] = mp[15]  # forearm.L
        blender_points[11] = mp[15]  # hand.L (используем запястье MediaPipe)

        blender_points[13] = mp[14]  # upper_arm.R
        blender_points[14] = mp[16]  # forearm.R
        blender_points[15] = mp[16]  # hand.R (используем запястье MediaPipe)

        # Ноги
        blender_points[16] = mp[23]  # thigh.L
        blender_points[17] = mp[25]  # shin.L
        blender_points[18] = mp[27]  # foot.L

        # Обработка пятки и пальцев для левой ноги
        if not np.isnan(mp[27]).any() and not np.isnan(mp[31]).any():
            ankle_l = mp[27]
            toe_l = mp[31]
            foot_vector = toe_l - ankle_l

            # Пятка находится в противоположном направлении от пальцев
            heel_vector = -foot_vector * 0.4
            blender_points[19] = ankle_l + heel_vector  # heel.L
            blender_points[20] = toe_l  # toe.L

        # Правая нога
        blender_points[21] = mp[24]  # thigh.R
        blender_points[22] = mp[26]  # shin.R
        blender_points[23] = mp[28]  # foot.R

        # Обработка пятки и пальцев для правой ноги
        if not np.isnan(mp[28]).any() and not np.isnan(mp[32]).any():
            ankle_r = mp[28]
            toe_r = mp[32]
            foot_vector = toe_r - ankle_r

            # Пятка находится в противоположном направлении от пальцев
            heel_vector = -foot_vector * 0.4
            blender_points[24] = ankle_r + heel_vector  # heel.R
            blender_points[25] = toe_r  # toe.R

        return blender_points

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

        # Массив для хранения результатов MediaPipe
        mp_landmarks = np.full((33, 3), np.nan)

        # Обработка кадра моделью Pose
        results = self.pose.process(frame_rgb)

        # Если точки тела найдены, извлекаем их координаты
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Координаты нормализованы от 0 до 1, умножаем на размеры кадра
                mp_landmarks[i, 0] = landmark.x * frame_width
                mp_landmarks[i, 1] = landmark.y * frame_height
                mp_landmarks[i, 2] = landmark.visibility

        # Преобразуем координаты MediaPipe в Blender-скелет
        blender_landmarks = self.map_mediapipe_to_blender_skeleton(mp_landmarks)

        return blender_landmarks

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
        for connection in self.SKELETON_CONNECTIONS:
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
            "landmark_names": self.SKELETON_LANDMARK_NAMES,
            "num_tracked_points": self.num_tracked_points,
            "connections": self.SKELETON_CONNECTIONS,
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