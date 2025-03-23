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
        spine = hips + 0.5 * (chest - hips)  # Точка посередине
        central_points['spine'] = spine

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

        # Функция для обеспечения правосторонней системы координат
        def ensure_right_handed(matrix):
            """Проверяет и исправляет матрицу, чтобы гарантировать положительный определитель"""
            det = np.linalg.det(matrix)
            if abs(det) < 1e-6:  # Проверка на сингулярность
                return np.eye(3)  # Возвращаем единичную матрицу в случае проблем
            if det < 0:
                # Инвертируем ось X для получения правосторонней системы
                matrix = matrix.copy()  # Создаем копию, чтобы не изменять оригинал
                matrix[:, 0] = -matrix[:, 0]
                return matrix
            return matrix

        # Функция для ортогонализации матрицы
        def orthogonalize(matrix):
            """Применяет процесс Грама-Шмидта для ортогонализации матрицы"""
            # Проверяем на наличие NaN значений
            if np.isnan(matrix).any():
                return np.eye(3)

            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            if np.linalg.det(u @ vh) < 0:
                u[:, -1] = -u[:, -1]
            return u @ vh

        # Функция для создания и проверки осей
        def create_validated_axes(x, y, z, default_axes=np.eye(3)):
            # Проверка на NaN или нулевые векторы
            if (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any() or
                    np.allclose(x, 0) or np.allclose(y, 0) or np.allclose(z, 0)):
                return default_axes

            # Создаем матрицу из осей и ортогонализируем
            axes = np.column_stack([x, y, z])
            orthogonal_axes = orthogonalize(axes)

            # Проверяем определитель
            return ensure_right_handed(orthogonal_axes)

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

        try:
            # Вычисляем центральные точки
            central_points = self.calculate_central_points(landmarks)

            # 1. Начинаем с таза (бедер) - корень иерархии
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hips = central_points['hips']

            # Получаем точки для позвоночника и плеч
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            chest = central_points.get('chest', (left_shoulder + right_shoulder) / 2)
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
                        local_axes['hips'] = create_validated_axes(pelvis_x, pelvis_y, pelvis_z)

                        # Вычисляем вращение от осей по умолчанию
                        default = default_axes['hips']
                        rotation_matrix = local_axes['hips'] @ default.T

                        try:
                            rotations['hips'] = R.from_matrix(rotation_matrix)
                        except ValueError:
                            # Если матрица имеет проблемы, используем единичное вращение
                            logger.warning(f"Невозможно создать вращение для hips, используем единичное вращение")
                            rotations['hips'] = R.identity()

            # Если не удалось создать вращение для таза, используем единичное
            if 'hips' not in rotations:
                rotations['hips'] = R.identity()
                local_axes['hips'] = np.eye(3)

            # 2. Вычисление позвоночника
            spine_vector = chest - hips
            norm_spine = np.linalg.norm(spine_vector)

            if norm_spine > 1e-6:
                spine_vector = spine_vector / norm_spine

                # Используем оси X и Z от таза
                spine_y = spine_vector  # Ось позвоночника (вверх)
                spine_x = local_axes['hips'][:, 0]  # X от таза

                # Z перпендикулярно X и Y
                spine_z = np.cross(spine_x, spine_y)
                norm_z = np.linalg.norm(spine_z)

                if norm_z > 1e-6:
                    spine_z = spine_z / norm_z

                    # Переортогонализируем X
                    spine_x = np.cross(spine_y, spine_z)
                    spine_x = spine_x / np.linalg.norm(spine_x)

                    # Создаем матрицу локальных осей
                    local_axes['spine'] = create_validated_axes(spine_x, spine_y, spine_z)

                    # Получаем вращение в локальной системе координат родителя
                    parent_axes = local_axes['hips']
                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['spine']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['spine']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            rotations['spine'] = R.from_matrix(rotation_matrix)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для spine, используем единичное вращение")
                            rotations['spine'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для spine")
                        rotations['spine'] = R.identity()
            else:
                rotations['spine'] = R.identity()

            # 3. Шея и голова
            # Приближаем положение шеи как точку между плечами
            neck_position = (left_shoulder + right_shoulder) / 2
            head_position = landmarks[0]  # Нос

            neck_to_head = head_position - neck_position
            norm_neck = np.linalg.norm(neck_to_head)

            if norm_neck > 1e-6:
                neck_to_head = neck_to_head / norm_neck

                # Используем Y как основное направление для шеи (вверх)
                neck_y = neck_to_head

                # X берем от позвоночника
                if 'spine' in local_axes:
                    neck_x = local_axes['spine'][:, 0]
                else:
                    neck_x = np.array([1, 0, 0])

                # Z перпендикулярно X и Y
                neck_z = np.cross(neck_x, neck_y)
                norm_z = np.linalg.norm(neck_z)

                if norm_z > 1e-6:
                    neck_z = neck_z / norm_z

                    # Переортогонализируем X
                    neck_x = np.cross(neck_y, neck_z)
                    neck_x = neck_x / np.linalg.norm(neck_x)

                    # Создаем матрицу локальных осей
                    local_axes['neck'] = create_validated_axes(neck_x, neck_y, neck_z)

                    # Получаем вращение в локальной системе координат родителя
                    if 'spine' in local_axes:
                        parent_axes = local_axes['spine']
                    else:
                        parent_axes = np.eye(3)

                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['neck']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['neck']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            rotations['neck'] = R.from_matrix(rotation_matrix)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для neck, используем единичное вращение")
                            rotations['neck'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для neck")
                        rotations['neck'] = R.identity()

                    # Добавляем голову (для простоты без дополнительных поворотов)
                    rotations['head'] = R.identity()
                else:
                    rotations['neck'] = R.identity()
                    rotations['head'] = R.identity()
            else:
                rotations['neck'] = R.identity()
                rotations['head'] = R.identity()

            # 4. Левое плечо
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
                    shoulder_y = np.array([0, 1, 0])

                # X перпендикулярно Y и Z
                shoulder_x = np.cross(shoulder_y, shoulder_z)
                norm_x = np.linalg.norm(shoulder_x)

                if norm_x > 1e-6:
                    shoulder_x = shoulder_x / norm_x

                    # Переортогонализируем Y
                    shoulder_y = np.cross(shoulder_z, shoulder_x)
                    shoulder_y = shoulder_y / np.linalg.norm(shoulder_y)

                    # Создаем матрицу локальных осей
                    local_axes['left_shoulder'] = create_validated_axes(shoulder_x, shoulder_y, shoulder_z)

                    # Получаем вращение в локальной системе координат родителя
                    if 'spine' in local_axes:
                        parent_axes = local_axes['spine']
                    else:
                        parent_axes = np.eye(3)

                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['left_shoulder']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['left_shoulder']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -80, 130)  # Сгибание вперед/назад
                            euler[1] = np.clip(euler[1], -30, 170)  # Отведение в сторону
                            euler[2] = np.clip(euler[2], -90, 90)  # Вращение

                            rotations['left_shoulder'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(
                                f"Невозможно создать вращение для left_shoulder, используем единичное вращение")
                            rotations['left_shoulder'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для left_shoulder")
                        rotations['left_shoulder'] = R.identity()
                else:
                    rotations['left_shoulder'] = R.identity()
            else:
                rotations['left_shoulder'] = R.identity()

            # 5. Правое плечо (аналогично левому)
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
                    shoulder_y = np.array([0, 1, 0])

                # X перпендикулярно Y и Z
                shoulder_x = np.cross(shoulder_y, shoulder_z)
                norm_x = np.linalg.norm(shoulder_x)

                if norm_x > 1e-6:
                    shoulder_x = shoulder_x / norm_x

                    # Переортогонализируем Y
                    shoulder_y = np.cross(shoulder_z, shoulder_x)
                    shoulder_y = shoulder_y / np.linalg.norm(shoulder_y)

                    # Создаем матрицу локальных осей
                    local_axes['right_shoulder'] = create_validated_axes(shoulder_x, shoulder_y, shoulder_z)

                    # Получаем вращение в локальной системе координат родителя
                    if 'spine' in local_axes:
                        parent_axes = local_axes['spine']
                    else:
                        parent_axes = np.eye(3)

                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['right_shoulder']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['right_shoulder']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -80, 130)  # Сгибание вперед/назад
                            euler[1] = np.clip(euler[1], -170, 30)  # Отведение в сторону
                            euler[2] = np.clip(euler[2], -90, 90)  # Вращение

                            rotations['right_shoulder'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(
                                f"Невозможно создать вращение для right_shoulder, используем единичное вращение")
                            rotations['right_shoulder'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для right_shoulder")
                        rotations['right_shoulder'] = R.identity()
                else:
                    rotations['right_shoulder'] = R.identity()
            else:
                rotations['right_shoulder'] = R.identity()

            # 6. Левый локоть
            left_forearm_vector = landmarks[15] - landmarks[13]  # Левое запястье - левый локоть
            norm_forearm = np.linalg.norm(left_forearm_vector)

            if norm_forearm > 1e-6 and 'left_shoulder' in local_axes:
                left_forearm_vector = left_forearm_vector / norm_forearm

                # Z - основная ось предплечья (влево)
                elbow_z = -left_forearm_vector  # Направление к запястью

                # Используем X от плеча
                elbow_x = local_axes['left_shoulder'][:, 0]

                # Y перпендикулярно Z и X
                elbow_y = np.cross(elbow_z, elbow_x)
                norm_y = np.linalg.norm(elbow_y)

                if norm_y > 1e-6:
                    elbow_y = elbow_y / norm_y

                    # Переортогонализируем X
                    elbow_x = np.cross(elbow_y, elbow_z)
                    elbow_x = elbow_x / np.linalg.norm(elbow_x)

                    # Создаем матрицу локальных осей
                    local_axes['left_elbow'] = create_validated_axes(elbow_x, elbow_y, elbow_z)

                    # Получаем вращение в локальной системе координат родителя
                    try:
                        parent_inverse = np.linalg.inv(local_axes['left_shoulder'])
                        local_to_parent = parent_inverse @ local_axes['left_elbow']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['left_elbow']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -10, 10)  # Очень мало вращения по X
                            euler[1] = np.clip(euler[1], -150, 0)  # Сгибание локтя (только в одну сторону)
                            euler[2] = np.clip(euler[2], -10, 10)  # Очень мало вращения по Z

                            rotations['left_elbow'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для left_elbow, используем единичное вращение")
                            rotations['left_elbow'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для left_elbow")
                        rotations['left_elbow'] = R.identity()

                    # Добавляем левое запястье (без дополнительного вращения)
                    rotations['left_wrist'] = R.identity()
                else:
                    rotations['left_elbow'] = R.identity()
                    rotations['left_wrist'] = R.identity()
            else:
                rotations['left_elbow'] = R.identity()
                rotations['left_wrist'] = R.identity()

            # 7. Правый локоть (аналогично левому)
            right_forearm_vector = landmarks[16] - landmarks[14]  # Правое запястье - правый локоть
            norm_forearm = np.linalg.norm(right_forearm_vector)

            if norm_forearm > 1e-6 and 'right_shoulder' in local_axes:
                right_forearm_vector = right_forearm_vector / norm_forearm

                # Z - основная ось предплечья (вправо)
                elbow_z = right_forearm_vector  # Направление к запястью

                # Используем X от плеча
                elbow_x = local_axes['right_shoulder'][:, 0]

                # Y перпендикулярно Z и X
                elbow_y = np.cross(elbow_z, elbow_x)
                norm_y = np.linalg.norm(elbow_y)

                if norm_y > 1e-6:
                    elbow_y = elbow_y / norm_y

                    # Переортогонализируем X
                    elbow_x = np.cross(elbow_y, elbow_z)
                    elbow_x = elbow_x / np.linalg.norm(elbow_x)

                    # Создаем матрицу локальных осей
                    local_axes['right_elbow'] = create_validated_axes(elbow_x, elbow_y, elbow_z)

                    # Получаем вращение в локальной системе координат родителя

                    try:
                        parent_inverse = np.linalg.inv(local_axes['right_shoulder'])
                        local_to_parent = parent_inverse @ local_axes['right_elbow']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['right_elbow']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -10, 10)  # Очень мало вращения по X
                            euler[1] = np.clip(euler[1], -150, 0)  # Сгибание локтя (только в одну сторону)
                            euler[2] = np.clip(euler[2], -10, 10)  # Очень мало вращения по Z

                            rotations['right_elbow'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(
                                f"Невозможно создать вращение для right_elbow, используем единичное вращение")
                            rotations['right_elbow'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для right_elbow")
                        rotations['right_elbow'] = R.identity()

                    # Добавляем правое запястье (без дополнительного вращения)
                    rotations['right_wrist'] = R.identity()
                else:
                    rotations['right_elbow'] = R.identity()
                    rotations['right_wrist'] = R.identity()

                # 8. Левое бедро
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
                    hip_x = np.array([1, 0, 0])

                # Z перпендикулярно X и Y
                hip_z = np.cross(hip_x, hip_y)
                norm_z = np.linalg.norm(hip_z)

                if norm_z > 1e-6:
                    hip_z = hip_z / norm_z

                    # Переортогонализируем X
                    hip_x = np.cross(hip_y, hip_z)
                    hip_x = hip_x / np.linalg.norm(hip_x)

                    # Создаем матрицу локальных осей
                    local_axes['left_hip'] = create_validated_axes(hip_x, hip_y, hip_z)

                    # Получаем вращение в локальной системе координат родителя
                    if 'hips' in local_axes:
                        parent_axes = local_axes['hips']
                    else:
                        parent_axes = np.eye(3)

                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['left_hip']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['left_hip']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -120, 45)  # Сгибание вперед/назад
                            euler[1] = np.clip(euler[1], -70, 70)  # Отведение в стороны
                            euler[2] = np.clip(euler[2], -45, 45)  # Вращение

                            rotations['left_hip'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для left_hip, используем единичное вращение")
                            rotations['left_hip'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для left_hip")
                        rotations['left_hip'] = R.identity()
                else:
                    rotations['left_hip'] = R.identity()
            else:
                rotations['left_hip'] = R.identity()

            # 9. Правое бедро (аналогично левому)
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
                    hip_x = np.array([1, 0, 0])

                # Z перпендикулярно X и Y
                hip_z = np.cross(hip_x, hip_y)
                norm_z = np.linalg.norm(hip_z)

                if norm_z > 1e-6:
                    hip_z = hip_z / norm_z

                    # Переортогонализируем X
                    hip_x = np.cross(hip_y, hip_z)
                    hip_x = hip_x / np.linalg.norm(hip_x)

                    # Создаем матрицу локальных осей
                    local_axes['right_hip'] = create_validated_axes(hip_x, hip_y, hip_z)

                    # Получаем вращение в локальной системе координат родителя
                    if 'hips' in local_axes:
                        parent_axes = local_axes['hips']
                    else:
                        parent_axes = np.eye(3)

                    try:
                        parent_inverse = np.linalg.inv(parent_axes)
                        local_to_parent = parent_inverse @ local_axes['right_hip']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['right_hip']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], -120, 45)  # Сгибание вперед/назад
                            euler[1] = np.clip(euler[1], -70, 70)  # Отведение в стороны
                            euler[2] = np.clip(euler[2], -45, 45)  # Вращение

                            rotations['right_hip'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для right_hip, используем единичное вращение")
                            rotations['right_hip'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для right_hip")
                        rotations['right_hip'] = R.identity()
                else:
                    rotations['right_hip'] = R.identity()
            else:
                rotations['right_hip'] = R.identity()

            # 10. Левое колено
            left_shin_vector = landmarks[27] - landmarks[25]  # Левая лодыжка - левое колено
            norm_shin = np.linalg.norm(left_shin_vector)

            if norm_shin > 1e-6 and 'left_hip' in local_axes:
                left_shin_vector = left_shin_vector / norm_shin

                # Y - основная ось голени (вниз)
                knee_y = -left_shin_vector  # Направление к лодыжке

                # Используем X от бедра
                knee_x = local_axes['left_hip'][:, 0]

                # Z перпендикулярно X и Y
                knee_z = np.cross(knee_x, knee_y)
                norm_z = np.linalg.norm(knee_z)

                if norm_z > 1e-6:
                    knee_z = knee_z / norm_z

                    # Переортогонализируем X
                    knee_x = np.cross(knee_y, knee_z)
                    knee_x = knee_x / np.linalg.norm(knee_x)

                    # Создаем матрицу локальных осей
                    local_axes['left_knee'] = create_validated_axes(knee_x, knee_y, knee_z)

                    # Получаем вращение в локальной системе координат родителя
                    try:
                        parent_inverse = np.linalg.inv(local_axes['left_hip'])
                        local_to_parent = parent_inverse @ local_axes['left_knee']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['left_knee']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], 0, 160)  # Сгибание колена (в основном в этой оси)
                            euler[1] = np.clip(euler[1], -5, 5)  # Минимальное отклонение в стороны
                            euler[2] = np.clip(euler[2], -5, 5)  # Минимальное вращение

                            rotations['left_knee'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для left_knee, используем единичное вращение")
                            rotations['left_knee'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для left_knee")
                        rotations['left_knee'] = R.identity()
                else:
                    rotations['left_knee'] = R.identity()
            else:
                rotations['left_knee'] = R.identity()

            # 11. Правое колено (аналогично левому)
            right_shin_vector = landmarks[28] - landmarks[26]  # Правая лодыжка - правое колено
            norm_shin = np.linalg.norm(right_shin_vector)

            if norm_shin > 1e-6 and 'right_hip' in local_axes:
                right_shin_vector = right_shin_vector / norm_shin

                # Y - основная ось голени (вниз)
                knee_y = -right_shin_vector  # Направление к лодыжке

                # Используем X от бедра
                knee_x = local_axes['right_hip'][:, 0]

                # Z перпендикулярно X и Y
                knee_z = np.cross(knee_x, knee_y)
                norm_z = np.linalg.norm(knee_z)

                if norm_z > 1e-6:
                    knee_z = knee_z / norm_z

                    # Переортогонализируем X
                    knee_x = np.cross(knee_y, knee_z)
                    knee_x = knee_x / np.linalg.norm(knee_x)

                    # Создаем матрицу локальных осей
                    local_axes['right_knee'] = create_validated_axes(knee_x, knee_y, knee_z)

                    # Получаем вращение в локальной системе координат родителя
                    try:
                        parent_inverse = np.linalg.inv(local_axes['right_hip'])
                        local_to_parent = parent_inverse @ local_axes['right_knee']

                        # Вычисляем вращение относительно осей по умолчанию
                        default = default_axes['right_knee']
                        rotation_matrix = local_to_parent @ default.T

                        try:
                            # Сначала получаем вращение без ограничений
                            rot = R.from_matrix(rotation_matrix)

                            # Применяем ограничения к углам Эйлера
                            euler = rot.as_euler('xyz', degrees=True)
                            euler[0] = np.clip(euler[0], 0, 160)  # Сгибание колена (в основном в этой оси)
                            euler[1] = np.clip(euler[1], -5, 5)  # Минимальное отклонение в стороны
                            euler[2] = np.clip(euler[2], -5, 5)  # Минимальное вращение

                            rotations['right_knee'] = R.from_euler('xyz', euler, degrees=True)
                        except ValueError:
                            logger.warning(f"Невозможно создать вращение для right_knee, используем единичное вращение")
                            rotations['right_knee'] = R.identity()
                    except np.linalg.LinAlgError:
                        logger.warning("Ошибка при инвертировании матрицы для right_knee")
                        rotations['right_knee'] = R.identity()
                else:
                    rotations['right_knee'] = R.identity()
            else:
                rotations['right_knee'] = R.identity()

            # 12. Левая лодыжка
            if 'left_knee' in local_axes:
                # Для простоты используем единичное вращение
                rotations['left_ankle'] = R.identity()
            else:
                rotations['left_ankle'] = R.identity()

            # 13. Правая лодыжка
            if 'right_knee' in local_axes:
                # Для простоты используем единичное вращение
                rotations['right_ankle'] = R.identity()
            else:
                rotations['right_ankle'] = R.identity()

            # 14. Добавляем идентичные вращения для всех суставов, для которых не удалось вычислить корректные вращения
            for joint in hierarchy.keys():
                if joint not in rotations:
                    rotations[joint] = R.identity()

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
        # Нормализуем векторы
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
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
            axis = axis / np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis)
        
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

    def process_sequence(self, points_3d: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
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
            # Пропускаем кадр, если в нем есть NaN-значения
            if np.isnan(points_3d[frame_idx]).any():
                continue
                
            # Получаем данные для текущего кадра
            frame_data = points_3d[frame_idx]

            # Вычисляем вращения для суставов
            joint_rotations = self.calculate_joint_rotations(frame_data)

            # Сохраняем данные для каждого сустава
            for joint, rotation in joint_rotations.items():
                if joint not in animation_params:
                    animation_params[joint] = {
                        'euler': np.zeros((num_frames, 3)),  # Углы Эйлера в градусах
                        'quaternion': np.zeros((num_frames, 4)),  # [w, x, y, z]
                        'rotvec': np.zeros((num_frames, 3))  # Вектор вращения
                    }

                # Получаем различные представления вращения
                animation_params[joint]['euler'][frame_idx] = rotation.as_euler('xyz', degrees=True)
                animation_params[joint]['quaternion'][frame_idx] = rotation.as_quat()  # xyzw формат
                animation_params[joint]['rotvec'][frame_idx] = rotation.as_rotvec()

        # Постобработка: меняем порядок элементов кватерниона с xyzw на wxyz
        for joint in animation_params:
            for frame_idx in range(num_frames):
                xyzw = animation_params[joint]['quaternion'][frame_idx]
                wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # xyzw -> wxyz
                animation_params[joint]['quaternion'][frame_idx] = wxyz

        return animation_params

    def export_to_bvh(self, animation_params: Dict[str, Any], output_path: str) -> None:
        """
        Экспортирует параметры анимации в формат BVH.
        """
        # Заглушка для будущей реализации
        pass