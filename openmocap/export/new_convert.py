import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointRotationCalculator:
    """
    Класс для вычисления вращений суставов из их пространственных позиций
    """

    def __init__(self, skeleton_model_path: Union[str, Path]):
        """
        Инициализирует калькулятор вращений

        Args:
            skeleton_model_path: Путь к JSON файлу с моделью скелета
        """
        self.skeleton_model = self._load_skeleton_model(skeleton_model_path)
        self.joint_hierarchy = self._build_joint_hierarchy()
        self.joint_limits = self._define_joint_limits()

    def _load_skeleton_model(self, skeleton_model_path: Union[str, Path]) -> Dict:
        """Загружает модель скелета из JSON файла"""
        try:
            with open(skeleton_model_path, 'r', encoding='utf-8') as f:
                model = json.load(f)

            logger.info(f"Модель скелета успешно загружена из {skeleton_model_path}")
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели скелета: {e}")
            raise

    def _build_joint_hierarchy(self) -> Dict[str, List[str]]:
        """Создает иерархию суставов из данных модели скелета"""
        hierarchy = {}
        connections = self.skeleton_model.get('connections', [])
        landmarks = self.skeleton_model.get('landmarks', [])

        # Создаем словарь для перевода индексов в имена
        idx_to_name = {}
        for landmark in landmarks:
            idx_to_name[landmark.get('index')] = landmark.get('name')

        # Строим иерархию
        for landmark in landmarks:
            name = landmark.get('name')
            hierarchy[name] = []

        # Заполняем дочерние элементы
        for connection in connections:
            start_idx = connection.get('start')
            end_idx = connection.get('end')

            start_name = idx_to_name.get(start_idx, f"point_{start_idx}")
            end_name = idx_to_name.get(end_idx, f"point_{end_idx}")

            if start_name in hierarchy and end_name not in hierarchy[start_name]:
                hierarchy[start_name].append(end_name)

        logger.info(f"Построена иерархия суставов с {len(hierarchy)} узлами")
        return hierarchy

    def _define_joint_limits(self) -> Dict[str, Dict[str, float]]:
        """Определяет биомеханические ограничения для суставов"""
        # Это типичные ограничения для человеческих суставов
        limits = {
            # Шея
            "spine.005": {"min_angle_x": -45, "max_angle_x": 45,
                          "min_angle_y": -45, "max_angle_y": 45,
                          "min_angle_z": -70, "max_angle_z": 70},

            # Голова
            "spine.006": {"min_angle_x": -25, "max_angle_x": 25,
                          "min_angle_y": -25, "max_angle_y": 25,
                          "min_angle_z": -30, "max_angle_z": 30},

            # Плечи
            "shoulder.L": {"min_angle_x": -180, "max_angle_x": 60,
                           "min_angle_y": -60, "max_angle_y": 180,
                           "min_angle_z": -90, "max_angle_z": 90},

            "shoulder.R": {"min_angle_x": -180, "max_angle_x": 60,
                           "min_angle_y": -180, "max_angle_y": 60,
                           "min_angle_z": -90, "max_angle_z": 90},

            # Локти (в основном сгибание по одной оси)
            "forearm.L": {"min_angle_x": -10, "max_angle_x": 10,
                          "min_angle_y": 0, "max_angle_y": 160,
                          "min_angle_z": -10, "max_angle_z": 10},

            "forearm.R": {"min_angle_x": -10, "max_angle_x": 10,
                          "min_angle_y": 0, "max_angle_y": 160,
                          "min_angle_z": -10, "max_angle_z": 10},

            # Бедра
            "thigh.L": {"min_angle_x": -120, "max_angle_x": 45,
                        "min_angle_y": -70, "max_angle_y": 70,
                        "min_angle_z": -45, "max_angle_z": 45},

            "thigh.R": {"min_angle_x": -120, "max_angle_x": 45,
                        "min_angle_y": -70, "max_angle_y": 70,
                        "min_angle_z": -45, "max_angle_z": 45},

            # Колени (в основном сгибание по одной оси)
            "shin.L": {"min_angle_x": 0, "max_angle_x": 160,
                       "min_angle_y": -5, "max_angle_y": 5,
                       "min_angle_z": -5, "max_angle_z": 5},

            "shin.R": {"min_angle_x": 0, "max_angle_x": 160,
                       "min_angle_y": -5, "max_angle_y": 5,
                       "min_angle_z": -5, "max_angle_z": 5},

            # Лодыжки
            "foot.L": {"min_angle_x": -45, "max_angle_x": 45,
                       "min_angle_y": -20, "max_angle_y": 20,
                       "min_angle_z": -30, "max_angle_z": 30},

            "foot.R": {"min_angle_x": -45, "max_angle_x": 45,
                       "min_angle_y": -20, "max_angle_y": 20,
                       "min_angle_z": -30, "max_angle_z": 30},
        }

        return limits

    def calculate_joint_rotations(self, joints_positions: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """
        Вычисляет вращения для суставов на основе их позиций

        Args:
            joints_positions: Словарь с позициями суставов {имя_сустава: [x, y, z]}

        Returns:
            Dict: Словарь с вращениями для каждого сустава
        """
        # Создаем numpy массивы из позиций суставов
        positions = {}
        for joint, pos in joints_positions.items():
            positions[joint] = np.array(pos)

        # Словарь для хранения результатов
        rotations = {}

        # Определяем корневой сустав (обычно таз)
        root_joint = "pelvis"

        # Обработка таза (корень)
        if root_joint in positions:
            # Для корневого сустава определяем глобальное вращение
            rotations[root_joint] = self._calculate_root_rotation(positions)

        # Обходим иерархию суставов и вычисляем вращения
        self._process_joint_hierarchy(root_joint, positions, rotations)

        # Преобразуем вращения в удобный формат для вывода
        formatted_rotations = {}
        for joint, rotation in rotations.items():
            if rotation is not None:
                # Кватернион в формате (w, x, y, z)
                quat = rotation.as_quat()
                # Преобразуем в формат [x, y, z, w] -> [w, x, y, z]
                formatted_rotations[joint] = {
                    "quaternion": {
                        "w": float(quat[3]),
                        "x": float(quat[0]),
                        "y": float(quat[1]),
                        "z": float(quat[2])
                    },
                    "euler": {
                        "x": float(rotation.as_euler("xyz", degrees=True)[0]),
                        "y": float(rotation.as_euler("xyz", degrees=True)[1]),
                        "z": float(rotation.as_euler("xyz", degrees=True)[2])
                    }
                }

        return formatted_rotations

    def _calculate_root_rotation(self, positions: Dict[str, np.ndarray]) -> R:
        """Вычисляет вращение для корневого сустава (таз)"""
        # Для таза можно использовать направление от левого к правому бедру для оси X
        if "thigh.L" in positions and "thigh.R" in positions:
            pelvis_x = self._normalize(positions["thigh.R"] - positions["thigh.L"])

            # Используем направление вверх (например, от таза к позвоночнику) для оси Y
            if "spine" in positions:
                pelvis_y = self._normalize(positions["spine"] - positions["pelvis"])
            else:
                # По умолчанию ось Y направлена вверх
                pelvis_y = np.array([0, 1, 0])

            # Вычисляем ось Z как перпендикуляр к осям X и Y
            pelvis_z = self._normalize(np.cross(pelvis_x, pelvis_y))

            # Переортогонализируем ось Y для обеспечения ортогональности
            pelvis_y = self._normalize(np.cross(pelvis_z, pelvis_x))

            # Создаем матрицу вращения из осей
            rotation_matrix = np.column_stack([pelvis_x, pelvis_y, pelvis_z])

            # Проверяем, что матрица образует правую систему координат
            if np.linalg.det(rotation_matrix) < 0:
                pelvis_x = -pelvis_x
                rotation_matrix = np.column_stack([pelvis_x, pelvis_y, pelvis_z])

            return R.from_matrix(rotation_matrix)

        # Если нет информации о бедрах, возвращаем единичное вращение
        return R.identity()

    def _process_joint_hierarchy(self, joint: str, positions: Dict[str, np.ndarray],
                                 rotations: Dict[str, R], parent_rotation: Optional[R] = None):
        """
        Рекурсивно обрабатывает иерархию суставов и вычисляет их вращения

        Args:
            joint: Текущий сустав
            positions: Словарь с позициями суставов
            rotations: Словарь для хранения результатов вращений
            parent_rotation: Вращение родительского сустава
        """
        # Если у текущего сустава уже вычислено вращение, используем его
        if joint in rotations:
            current_rotation = rotations[joint]
        else:
            # Иначе вычисляем вращение
            if joint in positions and parent_rotation is not None:
                # Вычисляем локальное вращение относительно родителя
                current_rotation = self._calculate_local_rotation(joint, positions, parent_rotation)

                # Применяем ограничения сустава
                if joint in self.joint_limits:
                    current_rotation = self._apply_joint_limits(current_rotation, self.joint_limits[joint])

                rotations[joint] = current_rotation
            else:
                current_rotation = R.identity()
                rotations[joint] = current_rotation

        # Обрабатываем дочерние суставы
        children = self.joint_hierarchy.get(joint, [])
        for child in children:
            self._process_joint_hierarchy(child, positions, rotations, current_rotation)

    def _calculate_local_rotation(self, joint: str, positions: Dict[str, np.ndarray], parent_rotation: R) -> R:
        """
        Вычисляет локальное вращение сустава относительно его родителя

        Args:
            joint: Имя сустава
            positions: Словарь с позициями суставов
            parent_rotation: Вращение родительского сустава

        Returns:
            R: Локальное вращение сустава
        """
        # Находим родительский сустав
        parent = None
        for potential_parent, children in self.joint_hierarchy.items():
            if joint in children:
                parent = potential_parent
                break

        if parent is None or parent not in positions or joint not in positions:
            return R.identity()

        # Вычисляем направление кости в глобальных координатах
        bone_direction = self._normalize(positions[joint] - positions[parent])

        # Инвертируем родительское вращение
        parent_rotation_inv = parent_rotation.inv()

        # Переводим направление кости в локальную систему координат родителя
        local_direction = parent_rotation_inv.apply(bone_direction)

        # Определяем оси локальной системы координат
        # Предполагаем, что в T-позе кость направлена вдоль оси Y (вверх)
        default_direction = np.array([0, 1, 0])

        # Вычисляем вращение, которое переводит ось Y в direction
        return self._rotation_between_vectors(default_direction, local_direction)

    def _apply_joint_limits(self, rotation: R, limits: Dict[str, float]) -> R:
        """
        Применяет биомеханические ограничения к вращению сустава

        Args:
            rotation: Исходное вращение
            limits: Словарь с ограничениями сустава

        Returns:
            R: Вращение с примененными ограничениями
        """
        # Получаем углы Эйлера в градусах
        euler = rotation.as_euler("xyz", degrees=True)

        # Применяем ограничения
        for i, axis in enumerate(["x", "y", "z"]):
            min_key = f"min_angle_{axis}"
            max_key = f"max_angle_{axis}"

            if min_key in limits and max_key in limits:
                euler[i] = np.clip(euler[i], limits[min_key], limits[max_key])

        # Создаем новое вращение из ограниченных углов
        return R.from_euler("xyz", euler, degrees=True)

    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> R:
        """
        Вычисляет вращение, переводящее вектор v1 в вектор v2

        Args:
            v1: Исходный вектор
            v2: Целевой вектор

        Returns:
            R: Вращение
        """
        # Нормализуем векторы
        v1 = self._normalize(v1)
        v2 = self._normalize(v2)

        # Вычисляем ось вращения (перпендикулярную обоим векторам)
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)

        # Если векторы параллельны (или антипараллельны)
        if axis_norm < 1e-6:
            # Проверяем, совпадают ли направления
            if np.dot(v1, v2) > 0.99:
                return R.identity()
            else:
                # Для антипараллельных векторов находим произвольную ось вращения
                if abs(v1[0]) > 0.01:
                    axis = self._normalize(np.cross(v1, [0, 1, 0]))
                else:
                    axis = self._normalize(np.cross(v1, [1, 0, 0]))
                return R.from_rotvec(np.pi * axis)

        # Нормализуем ось вращения
        axis = axis / axis_norm

        # Вычисляем угол между векторами
        cos_angle = np.dot(v1, v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Создаем вращение из оси и угла
        return R.from_rotvec(axis * angle)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Нормализует вектор"""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.zeros_like(vector)
        return vector / norm


# Пример использования
def main():
    # Путь к модели скелета (замените на актуальный путь)
    skeleton_model_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\skeleton_model.json"

    # Создаем калькулятор
    calculator = JointRotationCalculator(skeleton_model_path)

    # Загружаем позиции суставов из JSON
    with open(r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\clean_points_3d.json", "r") as f:
        data = json.load(f)

    # Извлекаем позиции суставов
    joints_positions = data["metadata"]["initial_positions"]

    # Вычисляем вращения
    rotations = calculator.calculate_joint_rotations(joints_positions)

    # Выводим результат
    output = {
        "metadata": data["metadata"],
        "frames": [
            {
                "time": 0.0,
                "frame_index": 0,
                "joints": rotations
            }
        ]
    }

    # Сохраняем результат в файл
    with open("joint_rotations_test.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Вращения суставов успешно вычислены и сохранены")


if __name__ == "__main__":
    main()