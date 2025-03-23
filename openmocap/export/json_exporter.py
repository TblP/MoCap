"""
Модуль для экспорта данных в формате JSON.

Предоставляет функциональность для сохранения данных отслеживания движения в JSON-файлы
с поддержкой оптимизированного формата для визуализации и дальнейшей обработки.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    Класс для экспорта данных отслеживания движения в формате JSON.

    Attributes:
        encoding (str): Кодировка файла
    """

    def __init__(self, encoding: str = 'utf-8'):
        """
        Инициализирует экспортер JSON.

        Args:
            encoding: Кодировка файла
        """
        self.encoding = encoding
        logger.info(f"Инициализирован JSON экспортер")

    def export_points_3d(
            self,
            points_3d: np.ndarray,
            output_path: Union[str, Path],
            landmark_names: Optional[List[str]] = None,
            fps: float = 30.0,
            include_confidence: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует 3D-координаты точек в JSON-файл.

        Args:
            points_3d: Массив 3D-координат точек shape (num_frames, num_landmarks, 3)
                      или (num_frames, num_landmarks, 4) с уверенностью
            output_path: Путь для сохранения файла
            landmark_names: Список имен ориентиров (ключевых точек)
            fps: Частота кадров
            include_confidence: Включать ли значения уверенности в экспорт
            metadata: Дополнительные метаданные для включения

        Raises:
            ValueError: Если формат данных некорректен
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Проверяем формат данных
        if len(points_3d.shape) != 3:
            raise ValueError(f"Неверный формат данных. Ожидается массив формы (num_frames, num_landmarks, 3 или 4), "
                             f"получено {points_3d.shape}")

        num_frames, num_landmarks, coords_dim = points_3d.shape

        if coords_dim not in [3, 4]:
            raise ValueError(f"Неверное количество координат. Ожидается 3 (x, y, z) или 4 (x, y, z, conf), "
                             f"получено {coords_dim}")

        has_confidence = coords_dim == 4

        # Если имена ориентиров не предоставлены, создаем стандартные
        if landmark_names is None:
            landmark_names = [f"landmark_{i}" for i in range(num_landmarks)]
        elif len(landmark_names) != num_landmarks:
            logger.warning(f"Количество имен ориентиров ({len(landmark_names)}) не соответствует "
                           f"количеству ориентиров в данных ({num_landmarks}). "
                           f"Будут использованы стандартные имена.")
            landmark_names = [f"landmark_{i}" for i in range(num_landmarks)]

        # Создаем словарь соответствия имен и индексов для метаданных
        joint_mapping = {name: i for i, name in enumerate(landmark_names)}

        # Подготавливаем метаданные
        json_metadata = {
            "fps": fps,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": num_frames,
            "duration": num_frames / fps if fps > 0 else 0,
            "joint_mapping": joint_mapping
        }

        # Добавляем пользовательские метаданные, если они предоставлены
        if metadata:
            for key, value in metadata.items():
                if key not in json_metadata:
                    json_metadata[key] = value

        # Создаем фреймы данных
        frames = []

        for frame_idx in range(num_frames):
            frame_data = {}

            for landmark_idx, landmark_name in enumerate(landmark_names):
                # Проверяем, есть ли NaN значения
                if np.any(np.isnan(points_3d[frame_idx, landmark_idx, :3])):
                    continue

                # Заполняем координаты
                coordinates = {
                    "x": float(points_3d[frame_idx, landmark_idx, 0]),
                    "y": float(points_3d[frame_idx, landmark_idx, 1]),
                    "z": float(points_3d[frame_idx, landmark_idx, 2])
                }

                # Добавляем уверенность, если она есть и нужна
                if has_confidence and include_confidence:
                    coordinates["confidence"] = float(points_3d[frame_idx, landmark_idx, 3])

                frame_data[landmark_name] = coordinates

            # Добавляем кадр только если в нем есть данные
            if frame_data:
                frames.append(frame_data)

        # Формируем итоговую структуру JSON
        json_data = {
            "metadata": json_metadata,
            "frames": frames
        }

        # Сохраняем в JSON-файл
        with open(output_path, 'w', encoding=self.encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Данные 3D-точек экспортированы в {output_path}")

    def export_skeleton_model(
            self,
            landmark_names: List[str],
            connections: List[tuple],
            output_path: Union[str, Path],
            segment_lengths: Optional[Dict[tuple, float]] = None,
            segment_names: Optional[Dict[tuple, str]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует модель скелета в JSON-файл.

        Args:
            landmark_names: Список имен ориентиров
            connections: Список соединений между ориентирами (ребра скелета)
            output_path: Путь для сохранения файла
            segment_lengths: Словарь длин сегментов
            segment_names: Словарь имен сегментов
            metadata: Дополнительные метаданные для включения
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Создаем словарь соответствия имен и индексов
        joint_mapping = {name: i for i, name in enumerate(landmark_names)}

        # Формируем список соединений
        skeleton_connections = []
        for start, end in connections:
            connection = {
                "start": start,
                "end": end,
                "start_name": landmark_names[start] if start < len(landmark_names) else f"unknown_{start}",
                "end_name": landmark_names[end] if end < len(landmark_names) else f"unknown_{end}"
            }

            # Добавляем имя сегмента, если доступно
            if segment_names and (start, end) in segment_names:
                connection["name"] = segment_names[(start, end)]

            # Добавляем длину сегмента, если доступна
            if segment_lengths and (start, end) in segment_lengths:
                connection["length"] = segment_lengths[(start, end)]

            skeleton_connections.append(connection)

        # Формируем метаданные
        json_metadata = {
            "num_landmarks": len(landmark_names),
            "num_connections": len(connections),
            "joint_mapping": joint_mapping
        }

        # Добавляем пользовательские метаданные, если они предоставлены
        if metadata:
            for key, value in metadata.items():
                if key not in json_metadata:
                    json_metadata[key] = value

        # Формируем итоговую структуру JSON
        json_data = {
            "metadata": json_metadata,
            "landmarks": [{"index": i, "name": name} for i, name in enumerate(landmark_names)],
            "connections": skeleton_connections
        }

        # Сохраняем в JSON-файл
        with open(output_path, 'w', encoding=self.encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Модель скелета экспортирована в {output_path}")

    def export_joint_angles(
            self,
            joint_angles: Dict[str, np.ndarray],
            output_path: Union[str, Path],
            fps: float = 30.0,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует углы суставов в JSON-файл.

        Args:
            joint_angles: Словарь {имя_сустава: массив_углов}
            output_path: Путь для сохранения файла
            fps: Частота кадров
            metadata: Дополнительные метаданные для включения
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Получаем количество кадров
        num_frames = max(len(angles) for angles in joint_angles.values())

        # Формируем метаданные
        json_metadata = {
            "fps": fps,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": num_frames,
            "duration": num_frames / fps if fps > 0 else 0,
            "joint_angles": list(joint_angles.keys())
        }

        # Добавляем пользовательские метаданные, если они предоставлены
        if metadata:
            for key, value in metadata.items():
                if key not in json_metadata:
                    json_metadata[key] = value

        # Создаем фреймы данных
        frames = []

        for frame_idx in range(num_frames):
            frame_data = {}

            for joint_name, angles in joint_angles.items():
                if frame_idx < len(angles) and not np.isnan(angles[frame_idx]):
                    frame_data[joint_name] = float(angles[frame_idx])

            # Добавляем кадр только если в нем есть данные
            if frame_data:
                frames.append(frame_data)

        # Формируем итоговую структуру JSON
        json_data = {
            "metadata": json_metadata,
            "frames": frames
        }

        # Сохраняем в JSON-файл
        with open(output_path, 'w', encoding=self.encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Углы суставов экспортированы в {output_path}")


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    frames = 100
    landmarks = 33  # MediaPipe Pose имеет 33 ориентира

    # 3D данные
    points_3d = np.random.rand(frames, landmarks, 3)

    # Углы суставов
    joint_angles = {
        "left_elbow": np.random.rand(frames) * 180,
        "right_elbow": np.random.rand(frames) * 180,
        "left_knee": np.random.rand(frames) * 180,
        "right_knee": np.random.rand(frames) * 180
    }

    # Модель скелета
    landmark_names = [f"landmark_{i}" for i in range(landmarks)]
    connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]

    # Создаем экспортер
    exporter = JSONExporter()

    # Экспортируем данные
    exporter.export_points_3d(points_3d, "test_3d_points.json", landmark_names)
    exporter.export_joint_angles(joint_angles, "test_joint_angles.json")
    exporter.export_skeleton_model(landmark_names, connections, "test_skeleton_model.json")