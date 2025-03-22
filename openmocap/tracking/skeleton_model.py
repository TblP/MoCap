"""
Модуль для работы с моделью скелета.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)


class SkeletonModel:
    """
    Класс для работы с моделью скелета.

    Содержит информацию о точках скелета, их соединениях и другие связанные данные.

    Attributes:
        landmark_names (List[str]): Список имен точек (ориентиров)
        connections (List[Tuple[int, int]]): Список соединений между точками (ребра скелета)
        segment_lengths (Dict[Tuple[int, int], float]): Словарь длин сегментов
        segment_names (Dict[Tuple[int, int], str]): Словарь имен сегментов
        joint_hierarchy (Dict[int, List[int]]): Иерархия суставов
        mass_distribution (Dict[int, float]): Распределение массы по сегментам
    """

    def __init__(
            self,
            landmark_names: List[str],
            connections: List[Tuple[int, int]],
            segment_lengths: Optional[Dict[Tuple[int, int], float]] = None,
            segment_names: Optional[Dict[Tuple[int, int], str]] = None,
            joint_hierarchy: Optional[Dict[int, List[int]]] = None,
            mass_distribution: Optional[Dict[int, float]] = None
    ):
        """
        Инициализирует объект модели скелета.

        Args:
            landmark_names: Список имен точек (ориентиров)
            connections: Список соединений между точками (ребра скелета)
            segment_lengths: Словарь длин сегментов
            segment_names: Словарь имен сегментов
            joint_hierarchy: Иерархия суставов
            mass_distribution: Распределение массы по сегментам
        """
        self.landmark_names = landmark_names
        self.connections = connections
        self.segment_lengths = segment_lengths or {}
        self.segment_names = segment_names or {}
        self.joint_hierarchy = joint_hierarchy or {}
        self.mass_distribution = mass_distribution or {}

        # Создаем словарь индексов ориентиров для удобного доступа
        self.landmark_indices = {name: i for i, name in enumerate(landmark_names)}

        # Автоматически создаем имена сегментов, если они не указаны
        if not segment_names:
            self._create_segment_names()

        logger.info(
            f"Инициализирована модель скелета с {len(landmark_names)} точками и {len(connections)} соединениями")

    def _create_segment_names(self):
        """
        Автоматически создает имена сегментов на основе имен точек.
        """
        for start, end in self.connections:
            if (start, end) not in self.segment_names:
                start_name = self.landmark_names[start]
                end_name = self.landmark_names[end]
                self.segment_names[(start, end)] = f"{start_name}_to_{end_name}"

    def calculate_segment_lengths(self, skeleton_data: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Вычисляет средние длины сегментов по данным скелета.

        Args:
            skeleton_data: Массив координат точек shape (num_frames, num_landmarks, 3)

        Returns:
            Dict[Tuple[int, int], float]: Словарь длин сегментов
        """
        # Инициализируем словарь длин сегментов
        segment_lengths = {}

        # Для каждого соединения
        for start, end in self.connections:
            # Извлекаем координаты начальной и конечной точек
            start_points = skeleton_data[:, start, :3]  # Берем только x, y, z
            end_points = skeleton_data[:, end, :3]  # Берем только x, y, z

            # Создаем маски для валидных точек
            valid_start = ~np.isnan(start_points).any(axis=1)
            valid_end = ~np.isnan(end_points).any(axis=1)
            valid = valid_start & valid_end

            # Если нет валидных точек, пропускаем этот сегмент
            if not np.any(valid):
                logger.warning(
                    f"Не найдено валидных точек для сегмента {self.segment_names.get((start, end), f'{start}-{end}')}")
                continue

            # Вычисляем евклидово расстояние для каждого кадра
            distances = np.linalg.norm(
                end_points[valid] - start_points[valid],
                axis=1
            )

            # Вычисляем медианное значение (более устойчиво к выбросам, чем среднее)
            median_distance = np.median(distances)

            # Сохраняем длину сегмента
            segment_lengths[(start, end)] = float(median_distance)

        # Обновляем словарь длин сегментов
        self.segment_lengths.update(segment_lengths)

        return segment_lengths

    def calculate_bone_lengths_robust(self, skeleton_data: np.ndarray, percentile: float = 50) -> Dict[
        Tuple[int, int], float]:
        """
        Вычисляет длины костей с использованием процентилей для устойчивости к выбросам.

        Args:
            skeleton_data: Массив координат точек shape (num_frames, num_landmarks, 3)
            percentile: Процентиль для расчета длины (50 = медиана)

        Returns:
            Dict[Tuple[int, int], float]: Словарь длин сегментов
        """
        # Инициализируем словарь длин сегментов
        segment_lengths = {}

        # Для каждого соединения
        for start, end in self.connections:
            # Извлекаем координаты начальной и конечной точек
            start_points = skeleton_data[:, start, :3]  # Берем только x, y, z
            end_points = skeleton_data[:, end, :3]  # Берем только x, y, z

            # Создаем маски для валидных точек
            valid_start = ~np.isnan(start_points).any(axis=1)
            valid_end = ~np.isnan(end_points).any(axis=1)
            valid = valid_start & valid_end

            # Если нет валидных точек, пропускаем этот сегмент
            if not np.any(valid):
                continue

            # Вычисляем евклидово расстояние для каждого кадра
            distances = np.linalg.norm(
                end_points[valid] - start_points[valid],
                axis=1
            )

            # Вычисляем процентиль
            percentile_distance = np.percentile(distances, percentile)

            # Сохраняем длину сегмента
            segment_lengths[(start, end)] = float(percentile_distance)

        return segment_lengths

    def enforce_bone_lengths(
            self,
            skeleton_data: np.ndarray,
            reference_lengths: Optional[Dict[Tuple[int, int], float]] = None
    ) -> np.ndarray:
        """
        Корректирует положение точек скелета для соблюдения длин костей.

        Args:
            skeleton_data: Массив координат точек shape (num_frames, num_landmarks, 3)
            reference_lengths: Словарь эталонных длин сегментов.
                               Если None, используются текущие длины из self.segment_lengths

        Returns:
            np.ndarray: Скорректированный массив координат точек
        """
        # Если эталонные длины не указаны, используем текущие
        lengths = reference_lengths or self.segment_lengths

        # Если нет длин сегментов, вычисляем их
        if not lengths:
            lengths = self.calculate_segment_lengths(skeleton_data)

        # Копируем исходные данные
        corrected_data = skeleton_data.copy()

        # Для каждого кадра
        for frame in range(skeleton_data.shape[0]):
            # Для каждого соединения
            for (start, end), length in lengths.items():
                # Проверяем, что точки валидны
                if np.isnan(corrected_data[frame, start]).any() or np.isnan(corrected_data[frame, end]).any():
                    continue

                # Получаем текущие координаты
                start_point = corrected_data[frame, start, :3]
                end_point = corrected_data[frame, end, :3]

                # Вычисляем вектор и его длину
                vector = end_point - start_point
                current_length = np.linalg.norm(vector)

                # Если длина близка к эталонной, пропускаем
                if np.isclose(current_length, length, rtol=0.01):
                    continue

                # Корректируем положение конечной точки
                if current_length > 0:  # Избегаем деления на ноль
                    unit_vector = vector / current_length
                    corrected_end = start_point + unit_vector * length
                    corrected_data[frame, end, :3] = corrected_end

                # Можно также корректировать положение обеих точек,
                # но это сложнее и требует учёта иерархии суставов

        return corrected_data

    def calculate_joint_angles(self, skeleton_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисляет углы суставов в скелете.

        Args:
            skeleton_data: Массив координат точек shape (num_frames, num_landmarks, 3)

        Returns:
            Dict[str, np.ndarray]: Словарь углов суставов по кадрам
        """
        num_frames = skeleton_data.shape[0]
        joint_angles = {}

        # Для каждого сустава находим соединенные с ним сегменты
        segments_by_joint = {}
        for start, end in self.connections:
            segments_by_joint.setdefault(start, []).append((start, end))
            segments_by_joint.setdefault(end, []).append((end, start))

        # Для каждого сустава, имеющего более одного сегмента
        for joint, segments in segments_by_joint.items():
            if len(segments) >= 2:
                joint_name = self.landmark_names[joint]

                # Инициализируем массив для углов
                angles = np.full(num_frames, np.nan)

                # Для каждого кадра
                for frame in range(num_frames):
                    # Создаем список векторов от сустава к соседним точкам
                    vectors = []

                    for segment in segments:
                        joint_idx, other_idx = segment

                        # Проверяем, что точки валидны
                        if np.isnan(skeleton_data[frame, joint_idx]).any() or np.isnan(
                                skeleton_data[frame, other_idx]).any():
                            continue

                        # Вычисляем вектор
                        if joint_idx == segment[0]:  # От сустава к другой точке
                            vector = skeleton_data[frame, other_idx, :3] - skeleton_data[frame, joint_idx, :3]
                        else:  # От другой точки к суставу
                            vector = skeleton_data[frame, joint_idx, :3] - skeleton_data[frame, other_idx, :3]

                        vectors.append(vector)

                    # Если у нас есть хотя бы два вектора
                    if len(vectors) >= 2:
                        # Берем первые два и вычисляем угол между ними
                        v1 = vectors[0]
                        v2 = vectors[1]

                        # Нормализуем векторы
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)

                        if v1_norm > 0 and v2_norm > 0:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm

                            # Вычисляем косинус угла и преобразуем в градусы
                            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                            angle = np.arccos(cos_angle) * 180 / np.pi

                            angles[frame] = angle

                # Добавляем в словарь углов
                joint_angles[joint_name] = angles

        return joint_angles

    def save_to_dict(self) -> Dict[str, Any]:
        """
        Сохраняет модель скелета в словарь.

        Returns:
            Dict[str, Any]: Словарь с параметрами модели скелета
        """
        return {
            "landmark_names": self.landmark_names,
            "connections": self.connections,
            "segment_lengths": self.segment_lengths,
            "segment_names": self.segment_names,
            "joint_hierarchy": self.joint_hierarchy,
            "mass_distribution": self.mass_distribution
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkeletonModel":
        """
        Создает модель скелета из словаря.

        Args:
            data: Словарь с параметрами модели скелета

        Returns:
            SkeletonModel: Модель скелета
        """
        return cls(
            landmark_names=data["landmark_names"],
            connections=data["connections"],
            segment_lengths=data.get("segment_lengths"),
            segment_names=data.get("segment_names"),
            joint_hierarchy=data.get("joint_hierarchy"),
            mass_distribution=data.get("mass_distribution")
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Сохраняет модель скелета в файл JSON.

        Args:
            file_path: Путь к файлу
        """
        import json

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            # Преобразуем ключи кортежей в строки
            data = self.save_to_dict()

            # Преобразуем кортежи в строки для segment_lengths и segment_names
            segment_lengths = {}
            for k, v in data["segment_lengths"].items():
                segment_lengths[str(k)] = v
            data["segment_lengths"] = segment_lengths

            segment_names = {}
            for k, v in data["segment_names"].items():
                segment_names[str(k)] = v
            data["segment_names"] = segment_names

            json.dump(data, f, indent=4)

        logger.info(f"Модель скелета сохранена в {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "SkeletonModel":
        """
        Загружает модель скелета из файла JSON.

        Args:
            file_path: Путь к файлу

        Returns:
            SkeletonModel: Модель скелета
        """
        import json
        import ast

        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Преобразуем строки обратно в кортежи для segment_lengths и segment_names
        segment_lengths = {}
        for k, v in data["segment_lengths"].items():
            segment_lengths[ast.literal_eval(k)] = v
        data["segment_lengths"] = segment_lengths

        segment_names = {}
        for k, v in data["segment_names"].items():
            segment_names[ast.literal_eval(k)] = v
        data["segment_names"] = segment_names

        logger.info(f"Модель скелета загружена из {file_path}")

        return cls.from_dict(data)


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем простую модель скелета
    landmark_names = ["head", "shoulder_left", "shoulder_right", "elbow_left", "elbow_right", "wrist_left",
                      "wrist_right"]
    connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

    skeleton_model = SkeletonModel(
        landmark_names=landmark_names,
        connections=connections
    )

    # Создаем тестовые данные скелета
    num_frames = 100
    skeleton_data = np.zeros((num_frames, len(landmark_names), 3))

    # Заполняем данные случайными значениями для теста
    for i, name in enumerate(landmark_names):
        skeleton_data[:, i, 0] = np.random.normal(i * 0.1, 0.01, num_frames)  # x
        skeleton_data[:, i, 1] = np.random.normal(i * 0.05, 0.01, num_frames)  # y
        skeleton_data[:, i, 2] = np.random.normal(i * 0.02, 0.01, num_frames)  # z

    # Вычисляем длины сегментов
    segment_lengths = skeleton_model.calculate_segment_lengths(skeleton_data)
    print("Длины сегментов:")
    for (start, end), length in segment_lengths.items():
        start_name = landmark_names[start]
        end_name = landmark_names[end]
        print(f"  {start_name} -> {end_name}: {length:.3f}")

    # Вычисляем углы суставов
    joint_angles = skeleton_model.calculate_joint_angles(skeleton_data)
    print("\nУглы суставов (среднее):")
    for joint_name, angles in joint_angles.items():
        print(f"  {joint_name}: {np.nanmean(angles):.1f} градусов")

    # Корректируем скелет для соблюдения длин костей
    corrected_data = skeleton_model.enforce_bone_lengths(skeleton_data)

    # Сохраняем и загружаем модель скелета
    skeleton_model.save_to_file("test_skeleton_model.json")
    loaded_model = SkeletonModel.load_from_file("test_skeleton_model.json")

    print("\nЗагруженная модель скелета:")
    print(f"  Точки: {loaded_model.landmark_names}")
    print(f"  Соединения: {loaded_model.connections}")