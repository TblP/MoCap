"""
Модуль для экспорта данных в формате CSV.

Предоставляет функциональность для сохранения данных отслеживания движения в CSV-файлы.
"""

import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)


class CSVExporter:
    """
    Класс для экспорта данных отслеживания движения в формате CSV.

    Attributes:
        delimiter (str): Разделитель для CSV-файла
        encoding (str): Кодировка файла
    """

    def __init__(self, delimiter: str = ',', encoding: str = 'utf-8'):
        """
        Инициализирует экспортер CSV.

        Args:
            delimiter: Разделитель для CSV-файла
            encoding: Кодировка файла
        """
        self.delimiter = delimiter
        self.encoding = encoding
        logger.info(f"Инициализирован CSV экспортер с разделителем '{delimiter}'")

    def export_points_3d(
            self,
            points_3d: np.ndarray,
            output_path: Union[str, Path],
            landmark_names: Optional[List[str]] = None,
            include_confidence: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует 3D-координаты точек в CSV-файл.

        Args:
            points_3d: Массив 3D-координат точек shape (num_frames, num_landmarks, 3)
                      или (num_frames, num_landmarks, 4) с уверенностью
            output_path: Путь для сохранения файла
            landmark_names: Список имен ориентиров (ключевых точек)
            include_confidence: Включать ли значения уверенности в экспорт
            metadata: Дополнительные метаданные для включения в заголовок

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

        # Создаем заголовки столбцов
        headers = ["frame"]
        for name in landmark_names:
            headers.append(f"{name}_x")
            headers.append(f"{name}_y")
            headers.append(f"{name}_z")
            if has_confidence and include_confidence:
                headers.append(f"{name}_confidence")

        # Создаем данные для экспорта
        rows = []
        for frame in range(num_frames):
            row = [frame]
            for landmark in range(num_landmarks):
                row.append(points_3d[frame, landmark, 0])  # x
                row.append(points_3d[frame, landmark, 1])  # y
                row.append(points_3d[frame, landmark, 2])  # z
                if has_confidence and include_confidence:
                    row.append(points_3d[frame, landmark, 3])  # confidence
            rows.append(row)

        # Сохраняем в CSV
        with open(output_path, 'w', newline='', encoding=self.encoding) as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            # Записываем метаданные, если они предоставлены
            if metadata:
                writer.writerow(["# Metadata"])
                for key, value in metadata.items():
                    writer.writerow([f"# {key}", value])
                writer.writerow([])  # Пустая строка после метаданных

            writer.writerow(headers)
            writer.writerows(rows)

        logger.info(f"Данные 3D-точек экспортированы в {output_path}")

    def export_points_2d(
            self,
            points_2d: np.ndarray,
            output_path: Union[str, Path],
            landmark_names: Optional[List[str]] = None,
            camera_names: Optional[List[str]] = None,
            include_confidence: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует 2D-координаты точек в CSV-файл.

        Args:
            points_2d: Массив 2D-координат точек shape (num_cameras, num_frames, num_landmarks, 2)
                      или (num_cameras, num_frames, num_landmarks, 3) с уверенностью
            output_path: Путь для сохранения файла
            landmark_names: Список имен ориентиров (ключевых точек)
            camera_names: Список имен камер
            include_confidence: Включать ли значения уверенности в экспорт
            metadata: Дополнительные метаданные для включения в заголовок

        Raises:
            ValueError: Если формат данных некорректен
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Проверяем формат данных
        if len(points_2d.shape) != 4:
            raise ValueError(
                f"Неверный формат данных. Ожидается массив формы (num_cameras, num_frames, num_landmarks, 2 или 3), "
                f"получено {points_2d.shape}")

        num_cameras, num_frames, num_landmarks, coords_dim = points_2d.shape

        if coords_dim not in [2, 3]:
            raise ValueError(f"Неверное количество координат. Ожидается 2 (x, y) или 3 (x, y, conf), "
                             f"получено {coords_dim}")

        has_confidence = coords_dim == 3

        # Если имена ориентиров не предоставлены, создаем стандартные
        if landmark_names is None:
            landmark_names = [f"landmark_{i}" for i in range(num_landmarks)]
        elif len(landmark_names) != num_landmarks:
            logger.warning(f"Количество имен ориентиров ({len(landmark_names)}) не соответствует "
                           f"количеству ориентиров в данных ({num_landmarks}). "
                           f"Будут использованы стандартные имена.")
            landmark_names = [f"landmark_{i}" for i in range(num_landmarks)]

        # Если имена камер не предоставлены, создаем стандартные
        if camera_names is None:
            camera_names = [f"camera_{i}" for i in range(num_cameras)]
        elif len(camera_names) != num_cameras:
            logger.warning(f"Количество имен камер ({len(camera_names)}) не соответствует "
                           f"количеству камер в данных ({num_cameras}). "
                           f"Будут использованы стандартные имена.")
            camera_names = [f"camera_{i}" for i in range(num_cameras)]

        # Создаем заголовки столбцов для каждой камеры
        all_headers = ["frame"]
        for cam_name in camera_names:
            for name in landmark_names:
                all_headers.append(f"{cam_name}_{name}_x")
                all_headers.append(f"{cam_name}_{name}_y")
                if has_confidence and include_confidence:
                    all_headers.append(f"{cam_name}_{name}_confidence")

        # Создаем данные для экспорта
        rows = []
        for frame in range(num_frames):
            row = [frame]
            for cam in range(num_cameras):
                for landmark in range(num_landmarks):
                    row.append(points_2d[cam, frame, landmark, 0])  # x
                    row.append(points_2d[cam, frame, landmark, 1])  # y
                    if has_confidence and include_confidence:
                        row.append(points_2d[cam, frame, landmark, 2])  # confidence
            rows.append(row)

        # Сохраняем в CSV
        with open(output_path, 'w', newline='', encoding=self.encoding) as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            # Записываем метаданные, если они предоставлены
            if metadata:
                writer.writerow(["# Metadata"])
                for key, value in metadata.items():
                    writer.writerow([f"# {key}", value])
                writer.writerow([])  # Пустая строка после метаданных

            writer.writerow(all_headers)
            writer.writerows(rows)

        logger.info(f"Данные 2D-точек экспортированы в {output_path}")

    def export_joint_angles(
            self,
            joint_angles: Dict[str, np.ndarray],
            output_path: Union[str, Path],
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Экспортирует углы суставов в CSV-файл.

        Args:
            joint_angles: Словарь {имя_сустава: массив_углов}
            output_path: Путь для сохранения файла
            metadata: Дополнительные метаданные для включения в заголовок
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Преобразуем в DataFrame для удобного экспорта
        df = pd.DataFrame(joint_angles)
        df.insert(0, 'frame', range(len(df)))

        # Сохраняем в CSV с метаданными
        with open(output_path, 'w', newline='', encoding=self.encoding) as f:
            # Записываем метаданные, если они предоставлены
            if metadata:
                f.write("# Metadata\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("\n")  # Пустая строка после метаданных

            # Записываем данные
            df.to_csv(f, index=False, sep=self.delimiter)

        logger.info(f"Данные углов суставов экспортированы в {output_path}")

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
        Экспортирует модель скелета в CSV-файл.

        Args:
            landmark_names: Список имен ориентиров
            connections: Список соединений между ориентирами (ребра скелета)
            output_path: Путь для сохранения файла
            segment_lengths: Словарь длин сегментов
            segment_names: Словарь имен сегментов
            metadata: Дополнительные метаданные для включения в заголовок
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Создаем данные для экспорта
        rows = []

        # Записываем информацию об ориентирах
        rows.append(["# Landmarks"])
        for i, name in enumerate(landmark_names):
            rows.append([i, name])

        rows.append([])  # Пустая строка

        # Записываем информацию о соединениях
        rows.append(["# Connections"])
        rows.append(["start_idx", "end_idx", "segment_name", "segment_length"])

        for start, end in connections:
            segment_name = segment_names.get((start, end), "") if segment_names else ""
            segment_length = segment_lengths.get((start, end), "") if segment_lengths else ""
            rows.append([start, end, segment_name, segment_length])

        # Сохраняем в CSV
        with open(output_path, 'w', newline='', encoding=self.encoding) as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            # Записываем метаданные, если они предоставлены
            if metadata:
                writer.writerow(["# Metadata"])
                for key, value in metadata.items():
                    writer.writerow([f"# {key}", value])
                writer.writerow([])  # Пустая строка после метаданных

            writer.writerows(rows)

        logger.info(f"Модель скелета экспортирована в {output_path}")


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    frames = 100
    landmarks = 33  # MediaPipe Pose имеет 33 ориентира

    # 3D данные
    points_3d = np.random.rand(frames, landmarks, 3)

    # 2D данные
    cameras = 2
    points_2d = np.random.rand(cameras, frames, landmarks, 2)

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
    exporter = CSVExporter()

    # Экспортируем данные
    exporter.export_points_3d(points_3d, "test_3d_points.csv", landmark_names)
    exporter.export_points_2d(points_2d, "test_2d_points.csv", landmark_names)
    exporter.export_joint_angles(joint_angles, "test_joint_angles.csv")
    exporter.export_skeleton_model(landmark_names, connections, "test_skeleton_model.csv")