"""
Модуль для расчета параметров анимации по данным захвата движения.

Предоставляет функции для расчета углов вращения и других параметров анимации
на основе 3D-координат точек, полученных от MediaPipe.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any


class AnimationCalculator:
    """
    Класс для расчета параметров анимации по данным захвата движения.

    Attributes:
        skeleton_type (str): Тип скелета ('mediapipe', 'custom', etc.)
    """

    def __init__(self, skeleton_type: str = 'mediapipe'):
        """
        Инициализирует калькулятор анимации.

        Args:
            skeleton_type: Тип скелета ('mediapipe', 'custom', etc.)
        """
        self.skeleton_type = skeleton_type

    def calculate_central_points(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисляет центральные точки скелета из ключевых точек MediaPipe.

        Args:
            landmarks: Массив точек формы (num_landmarks, 3)

        Returns:
            Dict: Словарь с вычисленными центральными точками
        """
        central_points = {}

        # Центр бедер (hips/pelvis)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hips = (left_hip + right_hip) / 2
        central_points['hips'] = hips

        # Центр плеч (chest)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        chest = (left_shoulder + right_shoulder) / 2
        central_points['chest'] = chest

        # Позвоночник (spine) - интерполяция между бедрами и плечами
        spine = hips + 0.5 * (chest - hips)  # Точка посередине между бедрами и плечами
        central_points['spine'] = spine

        return central_points

    def calculate_joint_rotations(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисляет углы вращения для суставов скелета.

        Args:
            landmarks: Массив точек формы (num_landmarks, 3)

        Returns:
            Dict: Словарь с углами вращения для каждого сустава
        """
        # Вычисляем центральные точки
        central_points = self.calculate_central_points(landmarks)

        # Создаем словарь для хранения углов
        rotations = {}

        # 1. Туловище

        # Бедра (pelvis/hips) - глобальное вращение относительно осей координат
        # Выбираем направление вперед (обычно ось Z) и направление вверх (обычно ось Y)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hips_forward = np.array([0, 0, 1])  # Предполагаем, что вперед это положительная ось Z
        hips_up = np.array([0, 1, 0])  # Предполагаем, что вверх это положительная ось Y

        # Вычисляем локальные оси для бедер
        hips_right = right_hip - left_hip  # Вектор от левого бедра к правому (ось X)
        hips_right_norm = np.linalg.norm(hips_right)
        if hips_right_norm > 0:
            hips_right = hips_right / hips_right_norm  # Нормализация
            rotations['hips'] = self._create_rotation_matrix_from_vectors(
                [hips_right, hips_up, np.cross(hips_right, hips_up)])

        # Позвоночник (spine) - вращение относительно бедер
        hips = central_points['hips']
        chest = central_points['chest']
        spine_vector = chest - hips
        spine_length = np.linalg.norm(spine_vector)

        if spine_length > 0:
            spine_direction = spine_vector / spine_length
            rotations['spine'] = self._calculate_rotation(hips_up, spine_direction)

        # 2. Руки

        # Левое плечо
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]

        shoulder_vector = left_elbow - left_shoulder
        shoulder_length = np.linalg.norm(shoulder_vector)

        if shoulder_length > 0:
            shoulder_direction = shoulder_vector / shoulder_length
            rotations['left_shoulder'] = self._calculate_rotation(np.array([-1, 0, 0]), shoulder_direction)

        # Левый локоть
        left_wrist = landmarks[15]

        elbow_vector = left_wrist - left_elbow
        elbow_length = np.linalg.norm(elbow_vector)

        if elbow_length > 0 and shoulder_length > 0:
            elbow_direction = elbow_vector / elbow_length
            rotations['left_elbow'] = self._calculate_rotation(shoulder_direction, elbow_direction)

        # Аналогично для правой руки
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]

        shoulder_vector = right_elbow - right_shoulder
        shoulder_length = np.linalg.norm(shoulder_vector)

        if shoulder_length > 0:
            shoulder_direction = shoulder_vector / shoulder_length
            rotations['right_shoulder'] = self._calculate_rotation(np.array([1, 0, 0]), shoulder_direction)

        # Правый локоть
        right_wrist = landmarks[16]

        elbow_vector = right_wrist - right_elbow
        elbow_length = np.linalg.norm(elbow_vector)

        if elbow_length > 0 and shoulder_length > 0:
            elbow_direction = elbow_vector / elbow_length
            rotations['right_elbow'] = self._calculate_rotation(shoulder_direction, elbow_direction)

        # 3. Ноги

        # Левое бедро
        left_hip = landmarks[23]
        left_knee = landmarks[25]

        hip_vector = left_knee - left_hip
        hip_length = np.linalg.norm(hip_vector)

        if hip_length > 0:
            hip_direction = hip_vector / hip_length
            rotations['left_hip'] = self._calculate_rotation(np.array([0, -1, 0]), hip_direction)

        # Левое колено
        left_ankle = landmarks[27]

        knee_vector = left_ankle - left_knee
        knee_length = np.linalg.norm(knee_vector)

        if knee_length > 0 and hip_length > 0:
            knee_direction = knee_vector / knee_length
            rotations['left_knee'] = self._calculate_rotation(hip_direction, knee_direction)

        # Аналогично для правой ноги
        right_hip = landmarks[24]
        right_knee = landmarks[26]

        hip_vector = right_knee - right_hip
        hip_length = np.linalg.norm(hip_vector)

        if hip_length > 0:
            hip_direction = hip_vector / hip_length
            rotations['right_hip'] = self._calculate_rotation(np.array([0, -1, 0]), hip_direction)

        # Правое колено
        right_ankle = landmarks[28]

        knee_vector = right_ankle - right_knee
        knee_length = np.linalg.norm(knee_vector)

        if knee_length > 0 and hip_length > 0:
            knee_direction = knee_vector / knee_length
            rotations['right_knee'] = self._calculate_rotation(hip_direction, knee_direction)

        return rotations

    def _calculate_rotation(self, default_direction: np.ndarray, current_direction: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу вращения от одного вектора к другому.

        Args:
            default_direction: Вектор направления по умолчанию
            current_direction: Текущий вектор направления

        Returns:
            np.ndarray: Матрица вращения 3x3
        """
        # Вычисляем ось вращения как векторное произведение
        rotation_axis = np.cross(default_direction, current_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-6:
            # Векторы параллельны
            if np.dot(default_direction, current_direction) > 0:
                # Совпадающие направления
                return np.eye(3)  # Единичная матрица
            else:
                # Противоположные направления
                # Выбираем произвольную перпендикулярную ось
                perpendicular = np.array([1, 0, 0]) if abs(default_direction[0]) < 0.9 else np.array([0, 1, 0])
                rotation_axis = np.cross(default_direction, perpendicular)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.pi  # 180 градусов
        else:
            # Нормализуем ось вращения
            rotation_axis = rotation_axis / rotation_axis_norm

            # Вычисляем угол между векторами
            cos_angle = np.clip(np.dot(default_direction, current_direction), -1.0, 1.0)
            angle = np.arccos(cos_angle)

        # Создаем матрицу вращения с помощью формулы Родригеса
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return rotation_matrix

    def _create_rotation_matrix_from_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Создает матрицу вращения из трех ортогональных векторов.

        Args:
            vectors: Список из трех векторов [right, up, forward]

        Returns:
            np.ndarray: Матрица вращения 3x3
        """
        # Нормализуем векторы
        normalized_vectors = [v / np.linalg.norm(v) for v in vectors]

        # Создаем матрицу вращения
        rotation_matrix = np.column_stack(normalized_vectors)

        return rotation_matrix

    def rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """
        Преобразует матрицу вращения в углы Эйлера (в градусах).

        Args:
            R: Матрица вращения 3x3

        Returns:
            np.ndarray: Углы Эйлера [roll, pitch, yaw] в градусах
        """
        # Проверка на gimbal lock
        if abs(R[2, 0]) >= 1.0 - 1e-6:
            # Gimbal lock случай
            yaw = np.arctan2(R[1, 2], R[1, 1])
            if R[2, 0] > 0:
                pitch = -np.pi / 2
                roll = 0
            else:
                pitch = np.pi / 2
                roll = 0
        else:
            # Стандартный случай
            pitch = -np.arcsin(R[2, 0])
            roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
            yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

        # Преобразуем радианы в градусы
        return np.array([roll, pitch, yaw]) * 180.0 / np.pi

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Преобразует матрицу вращения в кватернион.

        Args:
            R: Матрица вращения 3x3

        Returns:
            np.ndarray: Кватернион [w, x, y, z]
        """
        trace = np.trace(R)

        if trace > 0:
            S = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

        return np.array([w, x, y, z])

    def process_sequence(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Обрабатывает последовательность 3D-точек и вычисляет параметры анимации.

        Args:
            points_3d: Массив 3D-точек shape (num_frames, num_landmarks, 3)

        Returns:
            Dict: Словарь с параметрами анимации для каждого кадра
        """
        num_frames = points_3d.shape[0]
        animation_params = {}

        for frame_idx in range(num_frames):
            # Получаем данные для текущего кадра
            frame_data = points_3d[frame_idx]

            # Вычисляем углы вращения для суставов
            joint_rotations = self.calculate_joint_rotations(frame_data)

            # Преобразуем матрицы вращения в углы Эйлера или кватернионы
            for joint, rotation in joint_rotations.items():
                if joint not in animation_params:
                    animation_params[joint] = {
                        'euler': np.zeros((num_frames, 3)),
                        'quaternion': np.zeros((num_frames, 4))
                    }

                # Вычисляем углы Эйлера
                euler_angles = self.rotation_matrix_to_euler(rotation)
                animation_params[joint]['euler'][frame_idx] = euler_angles

                # Вычисляем кватернионы
                quaternion = self.rotation_matrix_to_quaternion(rotation)
                animation_params[joint]['quaternion'][frame_idx] = quaternion

        return animation_params

    def export_to_bvh(self, animation_params: Dict[str, Any], output_path: str) -> None:
        """
        Экспортирует параметры анимации в формат BVH.

        Args:
            animation_params: Словарь с параметрами анимации
            output_path: Путь для сохранения файла BVH
        """
        # Заглушка для будущей реализации
        pass

    def export_to_fbx(self, animation_params: Dict[str, Any], output_path: str) -> None:
        """
        Экспортирует параметры анимации в формат FBX.

        Args:
            animation_params: Словарь с параметрами анимации
            output_path: Путь для сохранения файла FBX
        """
        # Заглушка для будущей реализации
        pass


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    test_frames = 10
    test_landmarks = 33
    test_data = np.random.rand(test_frames, test_landmarks, 3)

    # Создаем калькулятор анимации
    calculator = AnimationCalculator()

    # Обрабатываем последовательность
    animation_params = calculator.process_sequence(test_data)

    print(f"Суставы: {list(animation_params.keys())}")
    print(f"Размерность углов Эйлера для бедер: {animation_params['hips']['euler'].shape}")
    print(f"Размерность кватернионов для бедер: {animation_params['hips']['quaternion'].shape}")