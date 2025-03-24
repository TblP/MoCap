import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy.spatial.transform import Rotation as R

from openmocap import logger


class AnimationCalculator:
    """
    Класс для расчета параметров анимации по данным захвата движения.
    Использует scipy.spatial.transform.Rotation для работы с вращениями.
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

        # Проверка на наличие NaN в ключевых точках
        if np.isnan(landmarks[23]).any() or np.isnan(landmarks[24]).any():
            # Если хотя бы одна из точек бедер содержит NaN, используем примерную позицию
            hips = np.array([0.0, 0.0, 0.0])  # Или другая разумная позиция по умолчанию
            central_points['hips'] = hips
        else:
            # Центр бедер (hips/pelvis)
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hips = (left_hip + right_hip) / 2
            central_points['hips'] = hips

        # Проверка на наличие NaN в точках плеч
        if np.isnan(landmarks[11]).any() or np.isnan(landmarks[12]).any():
            # Если хотя бы одна из точек плеч содержит NaN
            # Используем относительную позицию от бедер, если они доступны
            if 'hips' in central_points and not np.isnan(central_points['hips']).any():
                chest = central_points['hips'] + np.array([0.0, 0.5, 0.0])  # Примерно выше бедер
            else:
                chest = np.array([0.0, 0.5, 0.0])  # Или другая позиция по умолчанию
            central_points['chest'] = chest
        else:
            # Центр плеч (chest)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            chest = (left_shoulder + right_shoulder) / 2
            central_points['chest'] = chest

        # Позвоночник (spine) - интерполяция между бедрами и плечами
        if 'hips' in central_points and 'chest' in central_points:
            if not (np.isnan(central_points['hips']).any() or np.isnan(central_points['chest']).any()):
                spine = central_points['hips'] + 0.5 * (central_points['chest'] - central_points['hips'])
                central_points['spine'] = spine
            else:
                central_points['spine'] = np.array([0.0, 0.25, 0.0])  # Примерная позиция
        else:
            central_points['spine'] = np.array([0.0, 0.25, 0.0])  # Примерная позиция

        return central_points

    def calculate_joint_rotations(self, landmarks: np.ndarray) -> Dict[str, R]:
        """
        Вычисляет иерархические вращения для суставов скелета MediaPipe
        с учетом биомеханических ограничений.

        Args:
            landmarks: Массив точек формы (num_landmarks, 3)

        Returns:
            Dict: Словарь с вращениями (scipy.spatial.transform.Rotation) для суставов
        """
        rotations = {}
        local_axes = {}

        # Определяем оси по умолчанию для каждого сустава в T-pose
        default_axes = {
            'hips': np.array([  # Таз (корень иерархии)
                [1, 0, 0],  # X вправо
                [0, 1, 0],  # Y вверх
                [0, 0, 1]  # Z вперед
            ]),
            'spine': np.array([  # Позвоночник
                [1, 0, 0],  # X вправо
                [0, 1, 0],  # Y вверх
                [0, 0, 1]  # Z вперед
            ]),
            'neck': np.array([  # Шея
                [1, 0, 0],  # X вправо
                [0, 1, 0],  # Y вверх
                [0, 0, 1]  # Z вперед
            ]),
            'head': np.array([  # Голова
                [1, 0, 0],  # X вправо
                [0, 1, 0],  # Y вверх
                [0, 0, 1]  # Z вперед
            ]),
            'left_shoulder': np.array([  # Левое плечо
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [-1, 0, 0]  # Z влево (вдоль руки)
            ]),
            'right_shoulder': np.array([  # Правое плечо
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [1, 0, 0]  # Z вправо (вдоль руки)
            ]),
            'left_elbow': np.array([  # Левый локоть
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [-1, 0, 0]  # Z влево (вдоль предплечья)
            ]),
            'right_elbow': np.array([  # Правый локоть
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [1, 0, 0]  # Z вправо (вдоль предплечья)
            ]),
            'left_wrist': np.array([  # Левое запястье
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [-1, 0, 0]  # Z влево (вдоль кисти)
            ]),
            'right_wrist': np.array([  # Правое запястье
                [0, 0, 1],  # X вперед
                [0, 1, 0],  # Y вверх
                [1, 0, 0]  # Z вправо (вдоль кисти)
            ]),
            'left_hip': np.array([  # Левое бедро
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль бедра)
                [0, 0, 1]  # Z вперед
            ]),
            'right_hip': np.array([  # Правое бедро
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль бедра)
                [0, 0, 1]  # Z вперед
            ]),
            'left_knee': np.array([  # Левое колено
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль голени)
                [0, 0, 1]  # Z вперед
            ]),
            'right_knee': np.array([  # Правое колено
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль голени)
                [0, 0, 1]  # Z вперед
            ]),
            'left_ankle': np.array([  # Левая лодыжка
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль стопы)
                [0, 0, 1]  # Z вперед
            ]),
            'right_ankle': np.array([  # Правая лодыжка
                [1, 0, 0],  # X вправо
                [0, -1, 0],  # Y вниз (вдоль стопы)
                [0, 0, 1]  # Z вперед
            ])
        }

        # Определяем иерархию суставов MediaPipe
        hierarchy = {
            'hips': ['spine', 'left_hip', 'right_hip'],
            'spine': ['neck', 'left_shoulder', 'right_shoulder'],
            'neck': ['head'],
            'head': [],
            'left_shoulder': ['left_elbow'],
            'right_shoulder': ['right_elbow'],
            'left_elbow': ['left_wrist'],
            'right_elbow': ['right_wrist'],
            'left_wrist': [],
            'right_wrist': [],
            'left_hip': ['left_knee'],
            'right_hip': ['right_knee'],
            'left_knee': ['left_ankle'],
            'right_knee': ['right_ankle'],
            'left_ankle': [],
            'right_ankle': []
        }

        # Родительские суставы
        parents = {
            'spine': 'hips',
            'neck': 'spine',
            'head': 'neck',
            'left_shoulder': 'spine',
            'right_shoulder': 'spine',
            'left_elbow': 'left_shoulder',
            'right_elbow': 'right_shoulder',
            'left_wrist': 'left_elbow',
            'right_wrist': 'right_elbow',
            'left_hip': 'hips',
            'right_hip': 'hips',
            'left_knee': 'left_hip',
            'right_knee': 'right_hip',
            'left_ankle': 'left_knee',
            'right_ankle': 'right_knee'
        }

        # Инициализация всех вращений как идентичных
        for joint in hierarchy.keys():
            rotations[joint] = R.identity()

        # Маппинг индексов MediaPipe на имена суставов
        index_to_name = {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle'
        }

        # Функция для обеспечения правосторонней системы координат
        def ensure_right_handed(matrix):
            """Проверяет и исправляет матрицу, чтобы гарантировать положительный определитель"""
            if matrix is None:
                return np.eye(3)

            # Проверка на NaN
            if np.isnan(matrix).any():
                return np.eye(3)

            det = np.linalg.det(matrix)
            if abs(det) < 1e-6:  # Проверка на сингулярность
                return np.eye(3)  # Возвращаем единичную матрицу в случае проблем
            if det < 0:
                # Инвертируем ось X для получения правосторонней системы
                matrix = matrix.copy()  # Создаем копию, чтобы не изменять оригинал
                matrix[:, 0] = -matrix[:, 0]
            return matrix

        # Функция для ортогонализации матрицы
        def orthogonalize(matrix):
            """Применяет процесс Грама-Шмидта для ортогонализации матрицы"""
            # Проверка на None
            if matrix is None:
                return np.eye(3)

            # Проверяем на наличие NaN значений
            if np.isnan(matrix).any():
                return np.eye(3)

            try:
                u, s, vh = np.linalg.svd(matrix, full_matrices=False)
                if np.linalg.det(u @ vh) < 0:
                    u[:, -1] = -u[:, -1]
                return u @ vh
            except np.linalg.LinAlgError:
                return np.eye(3)

        # Функция для создания и проверки осей
        def create_validated_axes(x, y, z, default_axes=np.eye(3)):
            # Проверка на None
            if x is None or y is None or z is None:
                return default_axes

            # Проверка на NaN или нулевые векторы
            if (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any() or
                    np.allclose(x, 0) or np.allclose(y, 0) or np.allclose(z, 0)):
                return default_axes

            # Проверка на малые значения векторов
            if np.linalg.norm(x) < 1e-6 or np.linalg.norm(y) < 1e-6 or np.linalg.norm(z) < 1e-6:
                return default_axes

            try:
                # Создаем матрицу из осей и ортогонализируем
                axes = np.column_stack([x, y, z])
                orthogonal_axes = orthogonalize(axes)

                # Проверяем определитель
                return ensure_right_handed(orthogonal_axes)
            except Exception as e:
                logger.warning(f"Ошибка при создании осей: {e}")
                return default_axes

        # Проверка наличия NaN в ключевых точках скелета
        has_nan = False
        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if np.isnan(landmarks[i]).any():
                has_nan = True
                break

        if has_nan:
            # В случае наличия NaN в ключевых точках, возвращаем идентичные вращения
            # Это позволит избежать ошибок при расчетах
            for joint in hierarchy.keys():
                rotations[joint] = R.identity()
            return rotations

        try:
            # Вычисляем центральные точки
            central_points = self.calculate_central_points(landmarks)

            # Проверка наличия НАН в центральных точках
            for key, point in central_points.items():
                if np.isnan(point).any():
                    # Если в центральных точках есть NaN, возвращаем идентичные вращения
                    return {joint: R.identity() for joint in hierarchy.keys()}

            # 1. Начинаем с таза (бедер) - корень иерархии
            if np.isnan(landmarks[23]).any() or np.isnan(landmarks[24]).any():
                # Если одно из бедер содержит NaN, используем идентичное вращение
                rotations['hips'] = R.identity()
                local_axes['hips'] = default_axes['hips']
            else:
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                hips = central_points['hips']

                # Получаем точки для позвоночника и плеч
                left_shoulder = landmarks[11] if not np.isnan(landmarks[11]).any() else None
                right_shoulder = landmarks[12] if not np.isnan(landmarks[12]).any() else None

                if left_shoulder is not None and right_shoulder is not None:
                    chest = central_points.get('chest', (left_shoulder + right_shoulder) / 2)
                else:
                    chest = hips + np.array([0, 0.5, 0])  # Примерная позиция

                spine = central_points.get('spine', (hips + chest) / 2)

                # Создаем локальные оси для таза
                pelvis_x = right_hip - left_hip  # Вектор от левого к правому бедру
                norm_x = np.linalg.norm(pelvis_x)

                if norm_x > 1e-6:
                    pelvis_x = pelvis_x / norm_x

                    # Y-ось вверх
                    pelvis_y = chest - hips
                    norm_y = np.linalg.norm(pelvis_y)

                    if norm_y > 1e-6:
                        pelvis_y = pelvis_y / norm_y

                        # Z-ось вперед (перпендикулярно X и Y)
                        pelvis_z = np.cross(pelvis_x, pelvis_y)
                        norm_z = np.linalg.norm(pelvis_z)

                        if norm_z > 1e-6:
                            pelvis_z = pelvis_z / norm_z

                            # Создаем и проверяем матрицу локальных осей
                            try:
                                local_axes['hips'] = create_validated_axes(pelvis_x, pelvis_y, pelvis_z,
                                                                           default_axes['hips'])

                                # Вычисляем вращение от осей по умолчанию
                                default = default_axes['hips']
                                rotation_matrix = local_axes['hips'] @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    rotations['hips'] = R.from_matrix(rotation_matrix)
                                else:
                                    rotations['hips'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для hips: {e}")
                                rotations['hips'] = R.identity()
                                local_axes['hips'] = default_axes['hips']
                        else:
                            rotations['hips'] = R.identity()
                            local_axes['hips'] = default_axes['hips']
                    else:
                        rotations['hips'] = R.identity()
                        local_axes['hips'] = default_axes['hips']
                else:
                    rotations['hips'] = R.identity()
                    local_axes['hips'] = default_axes['hips']

            # 2. Вычисление позвоночника
            if 'spine' not in central_points or 'chest' not in central_points or 'hips' not in central_points:
                rotations['spine'] = R.identity()
                local_axes['spine'] = default_axes['spine']
            else:
                spine_vector = central_points['chest'] - central_points['hips']
                norm_spine = np.linalg.norm(spine_vector)

                if norm_spine > 1e-6:
                    spine_vector = spine_vector / norm_spine

                    # Используем оси X и Z от таза
                    spine_y = spine_vector  # Ось позвоночника (вверх)

                    if 'hips' in local_axes:
                        spine_x = local_axes['hips'][:, 0]  # X от таза
                    else:
                        spine_x = default_axes['hips'][:, 0]

                    # Z перпендикулярно X и Y
                    spine_z = np.cross(spine_x, spine_y)
                    norm_z = np.linalg.norm(spine_z)

                    if norm_z > 1e-6:
                        spine_z = spine_z / norm_z

                        # Переортогонализируем X
                        spine_x = np.cross(spine_y, spine_z)
                        norm_x = np.linalg.norm(spine_x)

                        if norm_x > 1e-6:
                            spine_x = spine_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['spine'] = create_validated_axes(spine_x, spine_y, spine_z,
                                                                            default_axes['spine'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('hips', default_axes['hips'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['spine']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['spine']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    rotations['spine'] = R.from_matrix(rotation_matrix)
                                else:
                                    rotations['spine'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для spine: {e}")
                                rotations['spine'] = R.identity()
                        else:
                            rotations['spine'] = R.identity()
                    else:
                        rotations['spine'] = R.identity()
                else:
                    rotations['spine'] = R.identity()

            # Инициализируем шею и голову
            rotations['neck'] = R.identity()
            rotations['head'] = R.identity()

            # 3. Левое плечо
            if np.isnan(landmarks[11]).any() or np.isnan(landmarks[13]).any():
                rotations['left_shoulder'] = R.identity()
                local_axes['left_shoulder'] = default_axes['left_shoulder']
            else:
                left_arm_vector = landmarks[13] - landmarks[11]  # Левый локоть - левое плечо
                norm_arm = np.linalg.norm(left_arm_vector)

                if norm_arm > 1e-6:
                    left_arm_vector = left_arm_vector / norm_arm

                    # Z - основная ось руки (влево)
                    shoulder_z = -left_arm_vector  # Направление к локтю

                    # Y берем от позвоночника
                    if 'spine' in local_axes:
                        shoulder_y = local_axes['spine'][:, 1]
                    else:
                        shoulder_y = default_axes['spine'][:, 1]

                    # X перпендикулярно Y и Z
                    shoulder_x = np.cross(shoulder_y, shoulder_z)
                    norm_x = np.linalg.norm(shoulder_x)

                    if norm_x > 1e-6:
                        shoulder_x = shoulder_x / norm_x

                        # Переортогонализируем Y
                        shoulder_y = np.cross(shoulder_z, shoulder_x)
                        norm_y = np.linalg.norm(shoulder_y)

                        if norm_y > 1e-6:
                            shoulder_y = shoulder_y / norm_y

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['left_shoulder'] = create_validated_axes(shoulder_x, shoulder_y, shoulder_z,
                                                                                    default_axes['left_shoulder'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('spine', default_axes['spine'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['left_shoulder']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['left_shoulder']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -80, 130)  # Сгибание вперед/назад
                                    euler[1] = np.clip(euler[1], -30, 170)  # Отведение в сторону
                                    euler[2] = np.clip(euler[2], -90, 90)  # Вращение

                                    rotations['left_shoulder'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['left_shoulder'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для left_shoulder: {e}")
                                rotations['left_shoulder'] = R.identity()
                        else:
                            rotations['left_shoulder'] = R.identity()
                    else:
                        rotations['left_shoulder'] = R.identity()
                else:
                    rotations['left_shoulder'] = R.identity()

            # 4. Правое плечо
            if np.isnan(landmarks[12]).any() or np.isnan(landmarks[14]).any():
                rotations['right_shoulder'] = R.identity()
                local_axes['right_shoulder'] = default_axes['right_shoulder']
            else:
                right_arm_vector = landmarks[14] - landmarks[12]  # Правый локоть - правое плечо
                norm_arm = np.linalg.norm(right_arm_vector)

                if norm_arm > 1e-6:
                    right_arm_vector = right_arm_vector / norm_arm

                    # Z - основная ось руки (вправо)
                    shoulder_z = right_arm_vector  # Направление к локтю

                    # Y берем от позвоночника
                    if 'spine' in local_axes:
                        shoulder_y = local_axes['spine'][:, 1]
                    else:
                        shoulder_y = default_axes['spine'][:, 1]

                    # X перпендикулярно Y и Z
                    shoulder_x = np.cross(shoulder_y, shoulder_z)
                    norm_x = np.linalg.norm(shoulder_x)

                    if norm_x > 1e-6:
                        shoulder_x = shoulder_x / norm_x

                        # Переортогонализируем Y
                        shoulder_y = np.cross(shoulder_z, shoulder_x)
                        norm_y = np.linalg.norm(shoulder_y)

                        if norm_y > 1e-6:
                            shoulder_y = shoulder_y / norm_y

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['right_shoulder'] = create_validated_axes(shoulder_x, shoulder_y, shoulder_z,
                                                                                     default_axes['right_shoulder'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('spine', default_axes['spine'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['right_shoulder']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['right_shoulder']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -80, 130)  # Сгибание вперед/назад
                                    euler[1] = np.clip(euler[1], -170, 30)  # Отведение в сторону
                                    euler[2] = np.clip(euler[2], -90, 90)  # Вращение

                                    rotations['right_shoulder'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['right_shoulder'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для right_shoulder: {e}")
                                rotations['right_shoulder'] = R.identity()
                        else:
                            rotations['right_shoulder'] = R.identity()
                    else:
                        rotations['right_shoulder'] = R.identity()
                else:
                    rotations['right_shoulder'] = R.identity()

            # 5. Левый локоть
            if np.isnan(landmarks[13]).any() or np.isnan(landmarks[15]).any():
                rotations['left_elbow'] = R.identity()
                local_axes['left_elbow'] = default_axes['left_elbow']
            else:
                left_forearm_vector = landmarks[15] - landmarks[13]  # Левое запястье - левый локоть
                norm_forearm = np.linalg.norm(left_forearm_vector)

                if norm_forearm > 1e-6 and 'left_shoulder' in local_axes:
                    left_forearm_vector = left_forearm_vector / norm_forearm

                    # Z - основная ось предплечья (влево)
                    elbow_z = -left_forearm_vector  # Направление к запястью

                    # Используем X от плеча
                    if 'left_shoulder' in local_axes:
                        elbow_x = local_axes['left_shoulder'][:, 0]
                    else:
                        elbow_x = default_axes['left_shoulder'][:, 0]

                    # Y перпендикулярно Z и X
                    elbow_y = np.cross(elbow_z, elbow_x)
                    norm_y = np.linalg.norm(elbow_y)

                    if norm_y > 1e-6:
                        elbow_y = elbow_y / norm_y

                        # Переортогонализируем X
                        elbow_x = np.cross(elbow_y, elbow_z)
                        norm_x = np.linalg.norm(elbow_x)

                        if norm_x > 1e-6:
                            elbow_x = elbow_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['left_elbow'] = create_validated_axes(elbow_x, elbow_y, elbow_z,
                                                                                 default_axes['left_elbow'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('left_shoulder', default_axes['left_shoulder'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['left_elbow']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['left_elbow']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -10, 10)  # Очень мало вращения по X
                                    euler[1] = np.clip(euler[1], -150, 0)  # Сгибание локтя (только в одну сторону)
                                    euler[2] = np.clip(euler[2], -10, 10)  # Очень мало вращения по Z

                                    rotations['left_elbow'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['left_elbow'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для left_elbow: {e}")
                                rotations['left_elbow'] = R.identity()
                        else:
                            rotations['left_elbow'] = R.identity()
                    else:
                        rotations['left_elbow'] = R.identity()
                else:
                    rotations['left_elbow'] = R.identity()

            # Инициализация left_wrist
            rotations['left_wrist'] = R.identity()

            # 6. Правый локоть
            if np.isnan(landmarks[14]).any() or np.isnan(landmarks[16]).any():
                rotations['right_elbow'] = R.identity()
                local_axes['right_elbow'] = default_axes['right_elbow']
            else:
                right_forearm_vector = landmarks[16] - landmarks[14]  # Правое запястье - правый локоть
                norm_forearm = np.linalg.norm(right_forearm_vector)

                if norm_forearm > 1e-6 and 'right_shoulder' in local_axes:
                    right_forearm_vector = right_forearm_vector / norm_forearm

                    # Z - основная ось предплечья (вправо)
                    elbow_z = right_forearm_vector  # Направление к запястью

                    # Используем X от плеча
                    if 'right_shoulder' in local_axes:
                        elbow_x = local_axes['right_shoulder'][:, 0]
                    else:
                        elbow_x = default_axes['right_shoulder'][:, 0]

                    # Y перпендикулярно Z и X
                    elbow_y = np.cross(elbow_z, elbow_x)
                    norm_y = np.linalg.norm(elbow_y)

                    if norm_y > 1e-6:
                        elbow_y = elbow_y / norm_y

                        # Переортогонализируем X
                        elbow_x = np.cross(elbow_y, elbow_z)
                        norm_x = np.linalg.norm(elbow_x)

                        if norm_x > 1e-6:
                            elbow_x = elbow_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['right_elbow'] = create_validated_axes(elbow_x, elbow_y, elbow_z,
                                                                                  default_axes['right_elbow'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('right_shoulder', default_axes['right_shoulder'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['right_elbow']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['right_elbow']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -10, 10)  # Очень мало вращения по X
                                    euler[1] = np.clip(euler[1], -150, 0)  # Сгибание локтя (только в одну сторону)
                                    euler[2] = np.clip(euler[2], -10, 10)  # Очень мало вращения по Z

                                    rotations['right_elbow'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['right_elbow'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для right_elbow: {e}")
                                rotations['right_elbow'] = R.identity()
                        else:
                            rotations['right_elbow'] = R.identity()
                    else:
                        rotations['right_elbow'] = R.identity()
                else:
                    rotations['right_elbow'] = R.identity()

            # Инициализация right_wrist
            rotations['right_wrist'] = R.identity()

            # 7. Левое бедро
            if np.isnan(landmarks[23]).any() or np.isnan(landmarks[25]).any():
                rotations['left_hip'] = R.identity()
                local_axes['left_hip'] = default_axes['left_hip']
            else:
                left_thigh_vector = landmarks[25] - landmarks[23]  # Левое колено - левое бедро
                norm_thigh = np.linalg.norm(left_thigh_vector)

                if norm_thigh > 1e-6:
                    left_thigh_vector = left_thigh_vector / norm_thigh

                    # Y - основная ось ноги (вниз)
                    hip_y = -left_thigh_vector  # Направление к колену

                    # X берем от таза
                    if 'hips' in local_axes:
                        hip_x = local_axes['hips'][:, 0]
                    else:
                        hip_x = default_axes['hips'][:, 0]

                    # Z перпендикулярно X и Y
                    hip_z = np.cross(hip_x, hip_y)
                    norm_z = np.linalg.norm(hip_z)

                    if norm_z > 1e-6:
                        hip_z = hip_z / norm_z

                        # Переортогонализируем X
                        hip_x = np.cross(hip_y, hip_z)
                        norm_x = np.linalg.norm(hip_x)

                        if norm_x > 1e-6:
                            hip_x = hip_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['left_hip'] = create_validated_axes(hip_x, hip_y, hip_z,
                                                                               default_axes['left_hip'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('hips', default_axes['hips'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['left_hip']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['left_hip']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -120, 45)  # Сгибание вперед/назад
                                    euler[1] = np.clip(euler[1], -70, 70)  # Отведение в стороны
                                    euler[2] = np.clip(euler[2], -45, 45)  # Вращение

                                    rotations['left_hip'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['left_hip'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для left_hip: {e}")
                                rotations['left_hip'] = R.identity()
                        else:
                            rotations['left_hip'] = R.identity()
                    else:
                        rotations['left_hip'] = R.identity()
                else:
                    rotations['left_hip'] = R.identity()

            # 8. Правое бедро
            if np.isnan(landmarks[24]).any() or np.isnan(landmarks[26]).any():
                rotations['right_hip'] = R.identity()
                local_axes['right_hip'] = default_axes['right_hip']
            else:
                right_thigh_vector = landmarks[26] - landmarks[24]  # Правое колено - правое бедро
                norm_thigh = np.linalg.norm(right_thigh_vector)

                if norm_thigh > 1e-6:
                    right_thigh_vector = right_thigh_vector / norm_thigh

                    # Y - основная ось ноги (вниз)
                    hip_y = -right_thigh_vector  # Направление к колену

                    # X берем от таза
                    if 'hips' in local_axes:
                        hip_x = local_axes['hips'][:, 0]
                    else:
                        hip_x = default_axes['hips'][:, 0]

                    # Z перпендикулярно X и Y
                    hip_z = np.cross(hip_x, hip_y)
                    norm_z = np.linalg.norm(hip_z)

                    if norm_z > 1e-6:
                        hip_z = hip_z / norm_z

                        # Переортогонализируем X
                        hip_x = np.cross(hip_y, hip_z)
                        norm_x = np.linalg.norm(hip_x)

                        if norm_x > 1e-6:
                            hip_x = hip_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['right_hip'] = create_validated_axes(hip_x, hip_y, hip_z,
                                                                                default_axes['right_hip'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('hips', default_axes['hips'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['right_hip']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['right_hip']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], -120, 45)  # Сгибание вперед/назад
                                    euler[1] = np.clip(euler[1], -70, 70)  # Отведение в стороны
                                    euler[2] = np.clip(euler[2], -45, 45)  # Вращение

                                    rotations['right_hip'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['right_hip'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для right_hip: {e}")
                                rotations['right_hip'] = R.identity()
                        else:
                            rotations['right_hip'] = R.identity()
                    else:
                        rotations['right_hip'] = R.identity()
                else:
                    rotations['right_hip'] = R.identity()

            # 9. Левое колено
            if np.isnan(landmarks[25]).any() or np.isnan(landmarks[27]).any():
                rotations['left_knee'] = R.identity()
                local_axes['left_knee'] = default_axes['left_knee']
            else:
                left_shin_vector = landmarks[27] - landmarks[25]  # Левая лодыжка - левое колено
                norm_shin = np.linalg.norm(left_shin_vector)

                if norm_shin > 1e-6:
                    left_shin_vector = left_shin_vector / norm_shin

                    # Y - основная ось голени (вниз)
                    knee_y = -left_shin_vector  # Направление к лодыжке

                    # Используем X от бедра
                    if 'left_hip' in local_axes:
                        knee_x = local_axes['left_hip'][:, 0]
                    else:
                        knee_x = default_axes['left_hip'][:, 0]

                    # Z перпендикулярно X и Y
                    knee_z = np.cross(knee_x, knee_y)
                    norm_z = np.linalg.norm(knee_z)

                    if norm_z > 1e-6:
                        knee_z = knee_z / norm_z

                        # Переортогонализируем X
                        knee_x = np.cross(knee_y, knee_z)
                        norm_x = np.linalg.norm(knee_x)

                        if norm_x > 1e-6:
                            knee_x = knee_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['left_knee'] = create_validated_axes(knee_x, knee_y, knee_z,
                                                                                default_axes['left_knee'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('left_hip', default_axes['left_hip'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['left_knee']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['left_knee']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], 0, 160)  # Сгибание колена (в основном в этой оси)
                                    euler[1] = np.clip(euler[1], -5, 5)  # Минимальное отклонение в стороны
                                    euler[2] = np.clip(euler[2], -5, 5)  # Минимальное вращение

                                    rotations['left_knee'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['left_knee'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для left_knee: {e}")
                                rotations['left_knee'] = R.identity()
                        else:
                            rotations['left_knee'] = R.identity()
                    else:
                        rotations['left_knee'] = R.identity()
                else:
                    rotations['left_knee'] = R.identity()

            # 10. Правое колено
            if np.isnan(landmarks[26]).any() or np.isnan(landmarks[28]).any():
                rotations['right_knee'] = R.identity()
                local_axes['right_knee'] = default_axes['right_knee']
            else:
                right_shin_vector = landmarks[28] - landmarks[26]  # Правая лодыжка - правое колено
                norm_shin = np.linalg.norm(right_shin_vector)

                if norm_shin > 1e-6:
                    right_shin_vector = right_shin_vector / norm_shin

                    # Y - основная ось голени (вниз)
                    knee_y = -right_shin_vector  # Направление к лодыжке

                    # Используем X от бедра
                    if 'right_hip' in local_axes:
                        knee_x = local_axes['right_hip'][:, 0]
                    else:
                        knee_x = default_axes['right_hip'][:, 0]

                    # Z перпендикулярно X и Y
                    knee_z = np.cross(knee_x, knee_y)
                    norm_z = np.linalg.norm(knee_z)

                    if norm_z > 1e-6:
                        knee_z = knee_z / norm_z

                        # Переортогонализируем X
                        knee_x = np.cross(knee_y, knee_z)
                        norm_x = np.linalg.norm(knee_x)

                        if norm_x > 1e-6:
                            knee_x = knee_x / norm_x

                            # Создаем матрицу локальных осей
                            try:
                                local_axes['right_knee'] = create_validated_axes(knee_x, knee_y, knee_z,
                                                                                 default_axes['right_knee'])

                                # Получаем вращение в локальной системе координат родителя
                                parent_axes = local_axes.get('right_hip', default_axes['right_hip'])

                                parent_inverse = np.linalg.inv(parent_axes)
                                local_to_parent = parent_inverse @ local_axes['right_knee']

                                # Вычисляем вращение относительно осей по умолчанию
                                default = default_axes['right_knee']
                                rotation_matrix = local_to_parent @ default.T

                                if not np.isnan(rotation_matrix).any() and abs(
                                        np.linalg.det(rotation_matrix) - 1.0) < 0.1:
                                    # Сначала получаем вращение без ограничений
                                    rot = R.from_matrix(rotation_matrix)

                                    # Применяем ограничения к углам Эйлера
                                    euler = rot.as_euler('xyz', degrees=True)
                                    euler[0] = np.clip(euler[0], 0, 160)  # Сгибание колена (в основном в этой оси)
                                    euler[1] = np.clip(euler[1], -5, 5)  # Минимальное отклонение в стороны
                                    euler[2] = np.clip(euler[2], -5, 5)  # Минимальное вращение

                                    rotations['right_knee'] = R.from_euler('xyz', euler, degrees=True)
                                else:
                                    rotations['right_knee'] = R.identity()
                            except Exception as e:
                                logger.warning(f"Ошибка при расчете вращения для right_knee: {e}")
                                rotations['right_knee'] = R.identity()
                        else:
                            rotations['right_knee'] = R.identity()
                    else:
                        rotations['right_knee'] = R.identity()
                else:
                    rotations['right_knee'] = R.identity()

            # 11. Инициализируем лодыжки с идентичными вращениями
            rotations['left_ankle'] = R.identity()
            rotations['right_ankle'] = R.identity()

        except Exception as e:
            logger.error(f"Ошибка при вычислении вращений суставов: {str(e)}")
            # В случае ошибки возвращаем пустые вращения для всех суставов
            for joint in hierarchy.keys():
                rotations[joint] = R.identity()

        return rotations

    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> R:
        """
        Вычисляет вращение от одного вектора к другому.

        Args:
            v1: Исходный вектор
            v2: Целевой вектор

        Returns:
            scipy.spatial.transform.Rotation: Объект вращения
        """
        # Проверка на None или NaN
        if v1 is None or v2 is None or np.isnan(v1).any() or np.isnan(v2).any():
            return R.identity()

        # Проверка на нулевые векторы
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return R.identity()

        # Нормализуем векторы
        v1 = v1 / norm1
        v2 = v2 / norm2

        # Если векторы почти совпадают
        if np.allclose(v1, v2, rtol=1e-4, atol=1e-4):
            return R.identity()

        # Если векторы противоположны
        if np.allclose(v1, -v2, rtol=1e-4, atol=1e-4):
            # Выбираем произвольную ось вращения, перпендикулярную v1
            if np.abs(v1[0]) < np.abs(v1[1]):
                axis = np.cross(v1, [1, 0, 0])
            else:
                axis = np.cross(v1, [0, 1, 0])

            # Проверка на нулевую ось вращения
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                return R.identity()

            axis = axis / axis_norm
            return R.from_rotvec(np.pi * axis)

        try:
            # Общий случай: используем кватернион вращения
            # Вычисляем ось вращения (перпендикулярную обоим векторам)
            axis = np.cross(v1, v2)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-6:
                # Если ось вращения очень мала, векторы почти сонаправлены или противонаправлены
                return R.identity()

            axis = axis / axis_norm

            # Вычисляем угол между векторами
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Создаем вращение из оси и угла
            return R.from_rotvec(axis * angle)
        except Exception as e:
            logger.warning(f"Ошибка при вычислении вращения между векторами: {e}")
            return R.identity()

    def process_sequence(self, points_3d: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Обрабатывает последовательность 3D-точек и вычисляет параметры анимации.

        Args:
            points_3d: Массив 3D-точек shape (num_frames, num_landmarks, 3)

        Returns:
            Dict: Словарь с параметрами анимации для каждого кадра
        """
        num_frames, num_landmarks, coords_dim = points_3d.shape
        animation_params = {}

        # Определение суставов, для которых будем вычислять вращения
        joint_names = [
            'hips', 'spine', 'neck', 'head',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # Инициализируем массивы для вращений и позиций
        for joint in joint_names:
            animation_params[joint] = {
                'euler': np.zeros((num_frames, 3)),  # Углы Эйлера в градусах
                'quaternion': np.zeros((num_frames, 4)),  # [w, x, y, z]
                'rotvec': np.zeros((num_frames, 3))  # Вектор вращения
            }

            # Заполняем значениями по умолчанию (идентичное вращение)
            identity_rot = R.identity()
            default_euler = identity_rot.as_euler('xyz', degrees=True)
            default_quat = np.array([1, 0, 0, 0])  # w, x, y, z
            default_rotvec = np.array([0, 0, 0])

            animation_params[joint]['euler'][:] = default_euler
            animation_params[joint]['quaternion'][:] = default_quat
            animation_params[joint]['rotvec'][:] = default_rotvec

        # Ключевые индексы для MediaPipe Pose
        key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        key_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                     "right_knee", "left_ankle", "right_ankle"]

        # Проверяем размерность данных
        if num_landmarks <= max(key_indices):
            logger.warning(f"Недостаточно точек в данных: {num_landmarks}, требуются индексы до {max(key_indices)}")
            # Возвращаем анимационные параметры со значениями по умолчанию
            return animation_params

        # Обрабатываем каждый кадр
        for frame_idx in range(num_frames):
            frame_data = points_3d[frame_idx]

            # Расширенная диагностика NaN
            nan_points = []

            # Проверяем наличие NaN в ключевых точках и собираем информацию
            for i, idx in enumerate(key_indices):
                if idx < frame_data.shape[0] and np.isnan(frame_data[idx]).any():
                    nan_coords = []
                    for j in range(coords_dim):
                        if np.isnan(frame_data[idx, j]):
                            nan_coords.append(j)
                    nan_points.append((idx, key_names[i], nan_coords))

            # Если есть NaN в ключевых точках
            if nan_points:
                missing_info = ', '.join([f"{name}({idx}): {coords}" for idx, name, coords in nan_points])
                logger.debug(f"Кадр {frame_idx} содержит NaN в ключевых точках: {missing_info}")
                # Пропускаем кадр - для него уже установлены значения по умолчанию
                continue

            # Основная обработка, если все ключевые точки валидны
            try:
                # Вычисляем вращения для суставов
                joint_rotations = self.calculate_joint_rotations(frame_data)

                # Сохраняем данные для каждого сустава
                for joint, rotation in joint_rotations.items():
                    if joint not in animation_params:
                        continue

                    try:
                        # Сохраняем различные представления вращения
                        animation_params[joint]['euler'][frame_idx] = rotation.as_euler('xyz', degrees=True)

                        # scipy вращения хранят кватернионы в формате xyzw, переводим в wxyz
                        quat_xyzw = rotation.as_quat()
                        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                        animation_params[joint]['quaternion'][frame_idx] = quat_wxyz

                        animation_params[joint]['rotvec'][frame_idx] = rotation.as_rotvec()
                    except Exception as e:
                        logger.warning(f"Ошибка при сохранении вращения для сустава {joint} в кадре {frame_idx}: {e}")
                        # При ошибке оставляем значения по умолчанию для этого сустава в текущем кадре

            except Exception as e:
                logger.warning(f"Ошибка при обработке кадра {frame_idx}: {e}")
                # При ошибке оставляем значения по умолчанию для всех суставов в текущем кадре

        # Постобработка: интерполируем пропущенные значения
        for joint in joint_names:
            for param_type in ['euler', 'quaternion', 'rotvec']:
                param_data = animation_params[joint][param_type]

                # Интерполяция для euler и rotvec (линейная)
                if param_type in ['euler', 'rotvec']:
                    # Пропускаем - для этих параметров мы уже установили значения по умолчанию,
                    # а не NaN, поэтому интерполяция не нужна
                    pass

                # Специальная обработка для кватернионов (сферическая интерполяция)
                # Если бы мы использовали NaN для кватернионов, их нужно было бы интерполировать
                # с учетом их специфики, но поскольку мы используем идентичные вращения,
                # то дополнительная интерполяция не требуется

        return animation_params
