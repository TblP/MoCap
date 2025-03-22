"""
Модуль для применения ограничений жесткого тела к данным скелета.

Содержит функции и классы для обеспечения постоянства длин костей
и поддержания реалистичных биомеханических ограничений в данных захвата движения.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

from openmocap.tracking.skeleton_model import SkeletonModel

logger = logging.getLogger(__name__)


class RigidBodyConstraints:
    """
    Класс для применения ограничений жесткого тела к данным скелета.

    Attributes:
        skeleton_model (SkeletonModel): Модель скелета с информацией о соединениях
        segment_lengths (Dict[Tuple[int, int], float]): Словарь длин сегментов (костей)
        joint_limits (Dict[str, Dict[str, float]]): Ограничения углов суставов
        root_joint_idx (int): Индекс корневого сустава в иерархии
    """

    def __init__(
            self,
            skeleton_model: SkeletonModel,
            segment_lengths: Optional[Dict[Tuple[int, int], float]] = None,
            joint_limits: Optional[Dict[str, Dict[str, float]]] = None,
            root_joint_idx: int = 23  # Таз (left_hip) как корневой сустав по умолчанию
    ):
        """
        Инициализирует объект ограничений жесткого тела.

        Args:
            skeleton_model: Модель скелета с информацией о соединениях
            segment_lengths: Словарь длин сегментов (костей). Если None, будет вычислен позже.
            joint_limits: Ограничения углов суставов.
                         Например: {'shoulder': {'min_angle': 0, 'max_angle': 180}}
            root_joint_idx: Индекс корневого сустава в иерархии
        """
        self.skeleton_model = skeleton_model
        self.segment_lengths = segment_lengths or {}
        self.joint_limits = joint_limits or {}
        self.root_joint_idx = root_joint_idx

        # Создаем иерархию суставов, если ее нет в модели скелета
        if not self.skeleton_model.joint_hierarchy:
            self._create_joint_hierarchy()

        logger.info("Инициализирован объект ограничений жесткого тела")

    def _create_joint_hierarchy(self):
        """
        Создает иерархию суставов на основе соединений в модели скелета.
        Корневой сустав - таз (hip).
        """
        # Создаем словарь смежности (для каждого сустава - список соединенных с ним)
        adjacency = {}
        for start, end in self.skeleton_model.connections:
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)

        # Создаем иерархию, начиная с корневого сустава
        hierarchy = {}
        visited = set()

        def build_hierarchy(joint_idx):
            visited.add(joint_idx)
            children = []

            for neighbor in adjacency.get(joint_idx, []):
                if neighbor not in visited:
                    children.append(neighbor)
                    build_hierarchy(neighbor)

            hierarchy[joint_idx] = children

        # Строим иерархию, начиная с корневого сустава
        build_hierarchy(self.root_joint_idx)

        # Обновляем иерархию в модели скелета
        self.skeleton_model.joint_hierarchy = hierarchy

        logger.debug(f"Создана иерархия суставов с корнем в {self.root_joint_idx}")

    def calculate_segment_lengths(self, skeleton_data: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Вычисляет длины сегментов на основе данных скелета.

        Args:
            skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)

        Returns:
            Dict[Tuple[int, int], float]: Словарь длин сегментов
        """
        # Используем метод из модели скелета для вычисления длин сегментов
        segment_lengths = self.skeleton_model.calculate_segment_lengths(skeleton_data)

        # Обновляем внутренний словарь длин сегментов
        self.segment_lengths = segment_lengths

        logger.info(f"Вычислены длины {len(segment_lengths)} сегментов")

        return segment_lengths

    def enforce_bone_lengths(
            self,
            skeleton_data: np.ndarray,
            iterations: int = 3,
            reference_lengths: Optional[Dict[Tuple[int, int], float]] = None
    ) -> np.ndarray:
        """
        Применяет ограничения длины костей к данным скелета.

        Args:
            skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
            iterations: Количество итераций алгоритма
            reference_lengths: Эталонные длины сегментов. Если None, используются
                              длины сегментов из self.segment_lengths

        Returns:
            np.ndarray: Скорректированные данные скелета
        """
        # Если эталонные длины не указаны, используем текущие
        lengths = reference_lengths or self.segment_lengths

        # Если нет длин сегментов, вычисляем их
        if not lengths:
            lengths = self.calculate_segment_lengths(skeleton_data)

        # Копируем исходные данные
        corrected_data = skeleton_data.copy()

        # Применяем ограничения несколько раз для лучшей сходимости
        for iteration in range(iterations):
            # Выполняем коррекцию, начиная с корневого сустава и двигаясь к конечностям
            corrected_data = self._enforce_bone_lengths_recursive(
                corrected_data,
                self.root_joint_idx,
                None,  # Нет родительского сустава для корня
                lengths
            )

            logger.debug(f"Завершена итерация {iteration + 1} из {iterations} применения ограничений длины костей")

        return corrected_data

    def _enforce_bone_lengths_recursive(
            self,
            skeleton_data: np.ndarray,
            joint_idx: int,
            parent_idx: Optional[int],
            segment_lengths: Dict[Tuple[int, int], float]
    ) -> np.ndarray:
        """
        Рекурсивно применяет ограничения длины костей, начиная с указанного сустава.

        Args:
            skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
            joint_idx: Индекс текущего сустава
            parent_idx: Индекс родительского сустава (None для корневого)
            segment_lengths: Словарь эталонных длин сегментов

        Returns:
            np.ndarray: Скорректированные данные скелета
        """
        num_frames = skeleton_data.shape[0]
        corrected_data = skeleton_data.copy()

        # Если есть родительский сустав, корректируем расстояние до него
        if parent_idx is not None:
            segment = tuple(sorted([parent_idx, joint_idx]))

            if segment in segment_lengths:
                target_length = segment_lengths[segment]

                for frame in range(num_frames):
                    # Пропускаем кадры с NaN
                    if (np.isnan(skeleton_data[frame, joint_idx]).any() or
                            np.isnan(skeleton_data[frame, parent_idx]).any()):
                        continue

                    # Текущие координаты
                    parent_pos = corrected_data[frame, parent_idx, :3]
                    joint_pos = corrected_data[frame, joint_idx, :3]

                    # Вычисляем вектор и его длину
                    vector = joint_pos - parent_pos
                    current_length = np.linalg.norm(vector)

                    if current_length < 1e-6:  # Избегаем деления на ноль
                        continue

                    # Нормализованный вектор в нужном направлении
                    unit_vector = vector / current_length

                    # Новая позиция сустава
                    new_joint_pos = parent_pos + unit_vector * target_length

                    # Обновляем данные
                    corrected_data[frame, joint_idx, :3] = new_joint_pos

        # Рекурсивно обрабатываем дочерние суставы
        for child_idx in self.skeleton_model.joint_hierarchy.get(joint_idx, []):
            corrected_data = self._enforce_bone_lengths_recursive(
                corrected_data,
                child_idx,
                joint_idx,
                segment_lengths
            )

        return corrected_data

    def enforce_joint_limits(
            self,
            skeleton_data: np.ndarray,
            joint_limits: Optional[Dict[str, Dict[str, float]]] = None
    ) -> np.ndarray:
        """
        Применяет ограничения углов суставов к данным скелета.

        Args:
            skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
            joint_limits: Ограничения углов суставов. Если None, используются
                         ограничения из self.joint_limits

        Returns:
            np.ndarray: Скорректированные данные скелета
        """
        # Если ограничения не указаны, используем текущие
        limits = joint_limits or self.joint_limits

        if not limits:
            logger.warning("Не указаны ограничения углов суставов. Возвращаем исходные данные.")
            return skeleton_data

        # Копируем исходные данные
        corrected_data = skeleton_data.copy()

        # Для каждого сустава с ограничениями
        for joint_name, limit_info in limits.items():
            if joint_name not in self.skeleton_model.landmark_indices:
                logger.warning(f"Сустав {joint_name} не найден в модели скелета. Пропускаем.")
                continue

            joint_idx = self.skeleton_model.landmark_indices[joint_name]

            # Получаем индексы родительского и дочерних суставов
            parent_idx = None
            children_indices = []

            for start, end in self.skeleton_model.connections:
                if start == joint_idx:
                    children_indices.append(end)
                elif end == joint_idx:
                    parent_idx = start

            if parent_idx is None or not children_indices:
                logger.warning(f"Для сустава {joint_name} не найдено соединений. Пропускаем.")
                continue

            # Применяем ограничения для каждого кадра
            min_angle = limit_info.get('min_angle', 0)
            max_angle = limit_info.get('max_angle', 180)

            for frame in range(skeleton_data.shape[0]):
                # Пропускаем кадры с NaN
                if np.isnan(skeleton_data[frame, joint_idx]).any() or np.isnan(skeleton_data[frame, parent_idx]).any():
                    continue

                # Текущие координаты
                parent_pos = corrected_data[frame, parent_idx, :3]
                joint_pos = corrected_data[frame, joint_idx, :3]

                # Для каждого дочернего сустава
                for child_idx in children_indices:
                    if np.isnan(skeleton_data[frame, child_idx]).any():
                        continue

                    child_pos = corrected_data[frame, child_idx, :3]

                    # Вычисляем векторы от сустава к родителю и от сустава к ребенку
                    parent_vector = parent_pos - joint_pos
                    child_vector = child_pos - joint_pos

                    # Вычисляем текущий угол
                    parent_length = np.linalg.norm(parent_vector)
                    child_length = np.linalg.norm(child_vector)

                    if parent_length < 1e-6 or child_length < 1e-6:
                        continue

                    parent_unit = parent_vector / parent_length
                    child_unit = child_vector / child_length

                    cos_angle = np.clip(np.dot(parent_unit, child_unit), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi

                    # Проверяем и корректируем угол
                    if angle < min_angle or angle > max_angle:
                        # Ограничиваем угол
                        target_angle = np.clip(angle, min_angle, max_angle)

                        # Вычисляем новый вектор для дочернего сустава
                        # (это упрощение, на практике нужно использовать кватернионы для 3D-вращений)
                        rotation_axis = np.cross(parent_unit, child_unit)
                        if np.linalg.norm(rotation_axis) < 1e-6:
                            # Векторы коллинеарны, выбираем произвольную ось вращения
                            rotation_axis = np.array([0, 0, 1])
                        else:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                        # Угол в радианах
                        target_angle_rad = target_angle * np.pi / 180

                        # Матрица вращения (формула Родригеса)
                        cos_t = np.cos(target_angle_rad)
                        sin_t = np.sin(target_angle_rad)
                        rx, ry, rz = rotation_axis

                        rotation_matrix = np.array([
                            [cos_t + rx * rx * (1 - cos_t), rx * ry * (1 - cos_t) - rz * sin_t,
                             rx * rz * (1 - cos_t) + ry * sin_t],
                            [ry * rx * (1 - cos_t) + rz * sin_t, cos_t + ry * ry * (1 - cos_t),
                             ry * rz * (1 - cos_t) - rx * sin_t],
                            [rz * rx * (1 - cos_t) - ry * sin_t, rz * ry * (1 - cos_t) + rx * sin_t,
                             cos_t + rz * rz * (1 - cos_t)]
                        ])

                        # Новый вектор и позиция
                        new_child_vector = np.dot(rotation_matrix, parent_unit) * child_length
                        new_child_pos = joint_pos + new_child_vector

                        # Обновляем данные
                        corrected_data[frame, child_idx, :3] = new_child_pos

        return corrected_data

    def enforce_constraints(
            self,
            skeleton_data: np.ndarray,
            enforce_bone_lengths: bool = True,
            enforce_joint_limits: bool = True,
            iterations: int = 3
    ) -> np.ndarray:
        """
        Применяет все ограничения жесткого тела к данным скелета.

        Args:
            skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
            enforce_bone_lengths: Применять ограничения длины костей
            enforce_joint_limits: Применять ограничения углов суставов
            iterations: Количество итераций алгоритма

        Returns:
            np.ndarray: Скорректированные данные скелета
        """
        corrected_data = skeleton_data.copy()

        # Применяем ограничения длины костей
        if enforce_bone_lengths:
            corrected_data = self.enforce_bone_lengths(
                corrected_data,
                iterations=iterations
            )

        # Применяем ограничения углов суставов
        if enforce_joint_limits:
            corrected_data = self.enforce_joint_limits(
                corrected_data
            )

        return corrected_data

    def save_constraints_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Сохраняет ограничения жесткого тела в файл JSON.

        Args:
            file_path: Путь к файлу
        """
        import json

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Преобразуем ключи-кортежи в строки для segment_lengths
        segment_lengths_serializable = {}
        for key, value in self.segment_lengths.items():
            segment_lengths_serializable[str(key)] = value

        data = {
            'segment_lengths': segment_lengths_serializable,
            'joint_limits': self.joint_limits,
            'root_joint_idx': self.root_joint_idx
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Ограничения жесткого тела сохранены в {file_path}")

    @classmethod
    def load_constraints_from_file(
            cls,
            file_path: Union[str, Path],
            skeleton_model: SkeletonModel
    ) -> "RigidBodyConstraints":
        """
        Загружает ограничения жесткого тела из файла JSON.

        Args:
            file_path: Путь к файлу
            skeleton_model: Модель скелета

        Returns:
            RigidBodyConstraints: Объект с загруженными ограничениями
        """
        import json
        import ast

        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Преобразуем строки обратно в кортежи для segment_lengths
        segment_lengths = {}
        for key_str, value in data.get('segment_lengths', {}).items():
            key = ast.literal_eval(key_str)
            segment_lengths[key] = value

        joint_limits = data.get('joint_limits', {})
        root_joint_idx = data.get('root_joint_idx', 23)  # 23 (left_hip) по умолчанию

        return cls(
            skeleton_model=skeleton_model,
            segment_lengths=segment_lengths,
            joint_limits=joint_limits,
            root_joint_idx=root_joint_idx
        )


# Набор стандартных ограничений для углов суставов (в градусах)
DEFAULT_JOINT_LIMITS = {
    # Ограничения для шеи
    'nose': {'min_angle': 30, 'max_angle': 180},

    # Ограничения для плеч
    'left_shoulder': {'min_angle': 0, 'max_angle': 180},
    'right_shoulder': {'min_angle': 0, 'max_angle': 180},

    # Ограничения для локтей
    'left_elbow': {'min_angle': 0, 'max_angle': 160},
    'right_elbow': {'min_angle': 0, 'max_angle': 160},

    # Ограничения для запястий
    'left_wrist': {'min_angle': 0, 'max_angle': 180},
    'right_wrist': {'min_angle': 0, 'max_angle': 180},

    # Ограничения для бедер
    'left_hip': {'min_angle': 0, 'max_angle': 180},
    'right_hip': {'min_angle': 0, 'max_angle': 180},

    # Ограничения для колен
    'left_knee': {'min_angle': 0, 'max_angle': 175},
    'right_knee': {'min_angle': 0, 'max_angle': 175},

    # Ограничения для лодыжек
    'left_ankle': {'min_angle': 45, 'max_angle': 135},
    'right_ankle': {'min_angle': 45, 'max_angle': 135}
}


def enforce_bone_lengths(
        skeleton_data: np.ndarray,
        skeleton_model: SkeletonModel,
        segment_lengths: Optional[Dict[Tuple[int, int], float]] = None,
        iterations: int = 3
) -> np.ndarray:
    """
    Применяет ограничения длины костей к данным скелета.

    Функция-обертка для быстрого использования ограничений длины костей
    без необходимости создавать объект RigidBodyConstraints.

    Args:
        skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
        skeleton_model: Модель скелета с информацией о соединениях
        segment_lengths: Словарь длин сегментов. Если None, вычисляется из данных.
        iterations: Количество итераций алгоритма

    Returns:
        np.ndarray: Скорректированные данные скелета
    """
    constraints = RigidBodyConstraints(
        skeleton_model=skeleton_model,
        segment_lengths=segment_lengths
    )

    if segment_lengths is None:
        constraints.calculate_segment_lengths(skeleton_data)

    return constraints.enforce_bone_lengths(
        skeleton_data,
        iterations=iterations
    )


def enforce_joint_limits(
        skeleton_data: np.ndarray,
        skeleton_model: SkeletonModel,
        joint_limits: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Применяет ограничения углов суставов к данным скелета.

    Функция-обертка для быстрого использования ограничений углов суставов
    без необходимости создавать объект RigidBodyConstraints.

    Args:
        skeleton_data: Данные скелета shape (num_frames, num_landmarks, 3)
        skeleton_model: Модель скелета с информацией о соединениях
        joint_limits: Ограничения углов суставов. Если None, используются DEFAULT_JOINT_LIMITS.

    Returns:
        np.ndarray: Скорректированные данные скелета
    """
    limits = joint_limits or DEFAULT_JOINT_LIMITS

    constraints = RigidBodyConstraints(
        skeleton_model=skeleton_model,
        joint_limits=limits
    )

    return constraints.enforce_joint_limits(skeleton_data)


# Пример использования:
if __name__ == "__main__":
    from openmocap.utils.logger import configure_logging, LogLevel
    from openmocap.tracking.skeleton_model import SkeletonModel

    configure_logging(LogLevel.DEBUG)

    # Создаем простую модель скелета
    landmark_names = ["head", "shoulder_left", "shoulder_right", "elbow_left",
                      "elbow_right", "wrist_left", "wrist_right"]
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

    # Создаем ограничения жесткого тела
    constraints = RigidBodyConstraints(skeleton_model)

    # Вычисляем длины сегментов
    segment_lengths = constraints.calculate_segment_lengths(skeleton_data)
    print("Длины сегментов:")
    for (start, end), length in segment_lengths.items():
        start_name = landmark_names[start]
        end_name = landmark_names[end]
        print(f"  {start_name} -> {end_name}: {length:.3f}")

    # Определяем ограничения углов суставов
    joint_limits = {
        'shoulder_left': {'min_angle': 0, 'max_angle': 180},
        'shoulder_right': {'min_angle': 0, 'max_angle': 180},
        'elbow_left': {'min_angle': 0, 'max_angle': 160},
        'elbow_right': {'min_angle': 0, 'max_angle': 160}
    }

    constraints.joint_limits = joint_limits

    # Применяем ограничения
    corrected_data = constraints.enforce_constraints(
        skeleton_data,
        enforce_bone_lengths=True,
        enforce_joint_limits=True,
        iterations=3
    )

    print(f"\nИсходная форма данных: {skeleton_data.shape}")
    print(f"Форма скорректированных данных: {corrected_data.shape}")

    # Проверяем, что длины костей сохранились
    for (start, end), target_length in segment_lengths.items():
        distances = np.linalg.norm(
            corrected_data[:, end, :3] - corrected_data[:, start, :3],
            axis=1
        )
        mean_distance = np.mean(distances)
        print(f"  Длина {landmark_names[start]} -> {landmark_names[end]}: "
              f"цель = {target_length:.3f}, результат = {mean_distance:.3f}")