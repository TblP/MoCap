"""
Утилиты для геометрических преобразований и вычислений.
"""

import logging
import numpy as np
from typing import Tuple, List, Union, Optional

logger = logging.getLogger(__name__)


def project_points_to_z_plane(points: np.ndarray) -> np.ndarray:
    """
    Проецирует 3D-точки на плоскость z=0.

    Args:
        points: Массив 3D-точек shape (..., 3)

    Returns:
        np.ndarray: Массив 3D-точек с z=0
    """
    # Копируем массив, чтобы не изменять оригинал
    projected_points = np.copy(points)

    # Устанавливаем z-координату равной 0
    projected_points[..., 2] = 0

    return projected_points


def rotate_around_x_axis(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Поворачивает точки вокруг оси X на заданный угол.

    Args:
        points: Массив 3D-точек shape (..., 3)
        angle_degrees: Угол поворота в градусах

    Returns:
        np.ndarray: Повернутые точки
    """
    # Преобразуем градусы в радианы
    angle_rad = np.radians(angle_degrees)

    # Создаем матрицу поворота вокруг оси X
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Получаем исходную форму массива
    original_shape = points.shape

    # Преобразуем массив точек в 2D (N, 3) для умножения на матрицу
    points_2d = points.reshape(-1, 3)

    # Применяем поворот
    rotated_points = np.dot(points_2d, rotation_matrix.T)

    # Возвращаем исходную форму массива
    return rotated_points.reshape(original_shape)


def rotate_around_y_axis(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Поворачивает точки вокруг оси Y на заданный угол.

    Args:
        points: Массив 3D-точек shape (..., 3)
        angle_degrees: Угол поворота в градусах

    Returns:
        np.ndarray: Повернутые точки
    """
    # Преобразуем градусы в радианы
    angle_rad = np.radians(angle_degrees)

    # Создаем матрицу поворота вокруг оси Y
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

    # Получаем исходную форму массива
    original_shape = points.shape

    # Преобразуем массив точек в 2D (N, 3) для умножения на матрицу
    points_2d = points.reshape(-1, 3)

    # Применяем поворот
    rotated_points = np.dot(points_2d, rotation_matrix.T)

    # Возвращаем исходную форму массива
    return rotated_points.reshape(original_shape)


def rotate_around_z_axis(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Поворачивает точки вокруг оси Z на заданный угол.

    Args:
        points: Массив 3D-точек shape (..., 3)
        angle_degrees: Угол поворота в градусах

    Returns:
        np.ndarray: Повернутые точки
    """
    # Преобразуем градусы в радианы
    angle_rad = np.radians(angle_degrees)

    # Создаем матрицу поворота вокруг оси Z
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    # Получаем исходную форму массива
    original_shape = points.shape

    # Преобразуем массив точек в 2D (N, 3) для умножения на матрицу
    points_2d = points.reshape(-1, 3)

    # Применяем поворот
    rotated_points = np.dot(points_2d, rotation_matrix.T)

    # Возвращаем исходную форму массива
    return rotated_points.reshape(original_shape)


def create_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Создает матрицу вращения из углов Эйлера (в радианах).

    Args:
        rx: Угол поворота вокруг оси X в радианах
        ry: Угол поворота вокруг оси Y в радианах
        rz: Угол поворота вокруг оси Z в радианах

    Returns:
        np.ndarray: Матрица поворота 3x3
    """
    # Матрицы поворота вокруг каждой оси
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Объединяем матрицы поворота (применяем их в порядке Z -> Y -> X)
    return Rx @ Ry @ Rz


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Преобразует вектор Родригеса в матрицу поворота.

    Args:
        rvec: Вектор Родригеса (3,)

    Returns:
        np.ndarray: Матрица поворота 3x3
    """
    theta = np.linalg.norm(rvec)

    if theta < 1e-8:
        return np.identity(3)

    # Нормализованный вектор поворота
    k = rvec / theta

    # Матрица поворота
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def rotation_matrix_to_rodrigues(rmat: np.ndarray) -> np.ndarray:
    """
    Преобразует матрицу поворота в вектор Родригеса.

    Args:
        rmat: Матрица поворота 3x3

    Returns:
        np.ndarray: Вектор Родригеса (3,)
    """
    # Проверяем, что матрица имеет правильный размер
    assert rmat.shape == (3, 3), "Матрица поворота должна иметь размер 3x3"

    # Вычисляем угол поворота
    trace = np.trace(rmat)
    theta = np.arccos((trace - 1) / 2)

    if np.abs(theta) < 1e-8:  # Если угол близок к нулю
        return np.zeros(3)

    # Вычисляем ось поворота
    axis = np.array([
        rmat[2, 1] - rmat[1, 2],
        rmat[0, 2] - rmat[2, 0],
        rmat[1, 0] - rmat[0, 1]
    ])

    # Нормализуем ось
    axis = axis / (2 * np.sin(theta))

    # Создаем вектор Родригеса
    rvec = axis * theta

    return rvec


def create_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Создает матрицу преобразования 4x4 из матрицы поворота и вектора перемещения.

    Args:
        rotation: Матрица поворота 3x3
        translation: Вектор перемещения (3,)

    Returns:
        np.ndarray: Матрица преобразования 4x4
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.flatten()

    return transform


def apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Применяет матрицу преобразования к набору точек.

    Args:
        points: Массив 3D-точек shape (..., 3)
        transformation: Матрица преобразования 4x4

    Returns:
        np.ndarray: Преобразованные точки
    """
    # Получаем исходную форму массива
    original_shape = points.shape

    # Преобразуем массив точек в 2D (N, 3) для умножения на матрицу
    points_2d = points.reshape(-1, 3)

    # Добавляем однородные координаты
    points_homogeneous = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))

    # Применяем преобразование
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation.T)

    # Конвертируем обратно в неоднородные координаты
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3:]

    # Возвращаем исходную форму массива
    return transformed_points.reshape(original_shape)


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Вычисляет евклидово расстояние между двумя точками.

    Args:
        point1: Первая точка (x, y, z)
        point2: Вторая точка (x, y, z)

    Returns:
        float: Расстояние между точками
    """
    return np.linalg.norm(point1 - point2)


def calculate_angle(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
    """
    Вычисляет угол между двумя векторами.

    Args:
        v1: Первый вектор
        v2: Второй вектор
        degrees: Если True, возвращает угол в градусах, иначе в радианах

    Returns:
        float: Угол между векторами
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)

    if degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad


def triangulate_points(
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        min_cameras: int = 2
) -> np.ndarray:
    """
    Триангулирует 3D-точки из набора 2D-точек и матриц проекции камер.

    Args:
        points_2d: Массив 2D-точек shape (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции shape (n_cameras, 3, 4)
        min_cameras: Минимальное количество камер для триангуляции

    Returns:
        np.ndarray: Массив 3D-точек shape (n_points, 3)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Создаем массив для хранения результатов
    points_3d = np.zeros((n_points, 3))

    for i in range(n_points):
        # Собираем 2D-точки для данной точки со всех камер
        point_2d_all = points_2d[:, i, :]

        # Проверяем, сколько камер видят эту точку
        valid_cameras = ~np.any(np.isnan(point_2d_all), axis=1)
        n_valid_cameras = np.sum(valid_cameras)

        if n_valid_cameras < min_cameras:
            # Недостаточно камер для триангуляции
            points_3d[i] = np.nan
            continue

        # Собираем валидные 2D-точки и матрицы проекции
        valid_points_2d = point_2d_all[valid_cameras]
        valid_proj_matrices = projection_matrices[valid_cameras]

        # Формируем матрицу уравнений DLT (Direct Linear Transform)
        A = np.zeros((n_valid_cameras * 2, 4))

        for j, (point, proj_mat) in enumerate(zip(valid_points_2d, valid_proj_matrices)):
            x, y = point

            # A формируется из (x*P[2,:] - P[0,:]) и (y*P[2,:] - P[1,:])
            A[j * 2] = x * proj_mat[2] - proj_mat[0]
            A[j * 2 + 1] = y * proj_mat[2] - proj_mat[1]

        # Решаем задачу наименьших квадратов
        _, _, Vh = np.linalg.svd(A)
        point_3d_homogeneous = Vh[-1]

        # Переходим от однородных к евклидовым координатам
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        # Сохраняем результат
        points_3d[i] = point_3d

    return points_3d


def calculate_reprojection_error(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        mean: bool = True
) -> np.ndarray:
    """
    Вычисляет ошибку репроекции 3D-точек на изображения.

    Args:
        points_3d: Массив 3D-точек shape (n_points, 3)
        points_2d: Массив 2D-точек shape (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции shape (n_cameras, 3, 4)
        mean: Если True, возвращает среднюю ошибку для каждой точки

    Returns:
        np.ndarray: Ошибка репроекции
            Если mean=True: shape (n_points,)
            Если mean=False: shape (n_cameras, n_points, 2)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Проверяем размеры
    assert points_3d.shape[0] == n_points, "Количество 3D-точек не соответствует количеству 2D-точек"
    assert projection_matrices.shape[0] == n_cameras, "Количество матриц проекции не соответствует количеству камер"

    # Преобразуем 3D-точки в однородные координаты
    points_3d_homogeneous = np.hstack((points_3d, np.ones((n_points, 1))))

    # Массив для хранения ошибок репроекции
    errors = np.full((n_cameras, n_points, 2), np.nan)

    # Для каждой камеры
    for cam_idx in range(n_cameras):
        # Проецируем 3D-точки на плоскость камеры
        projected_points_homogeneous = np.dot(points_3d_homogeneous, projection_matrices[cam_idx].T)

        # Переходим к неоднородным координатам
        projected_points = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2:]

        # Вычисляем ошибку репроекции
        valid_mask = ~np.any(np.isnan(points_2d[cam_idx]), axis=1)
        errors[cam_idx, valid_mask] = points_2d[cam_idx, valid_mask] - projected_points[valid_mask]

    if mean:
        # Вычисляем евклидову норму ошибки
        errors_norm = np.linalg.norm(errors, axis=2)

        # Учитываем только валидные камеры
        valid_mask = ~np.isnan(errors_norm)
        errors_sum = np.nansum(errors_norm, axis=0)
        valid_counts = np.sum(valid_mask, axis=0)

        # Средняя ошибка для каждой точки
        mean_errors = np.zeros(n_points)
        valid_points = valid_counts > 0
        mean_errors[valid_points] = errors_sum[valid_points] / valid_counts[valid_points]

        return mean_errors
    else:
        return errors


def rigid_transform_3d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Находит оптимальное жесткое преобразование между двумя наборами точек.
    Использует алгоритм Кабша.

    Args:
        A: Первый набор 3D-точек (N, 3)
        B: Второй набор 3D-точек (N, 3)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (R, t) - матрица поворота и вектор перемещения
    """
    assert A.shape == B.shape, "Наборы точек должны иметь одинаковую форму"

    # Находим центры масс
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Центрируем точки
    AA = A - centroid_A
    BB = B - centroid_B

    # Матрица ковариации
    H = AA.T @ BB

    # SVD разложение
    U, _, Vt = np.linalg.svd(H)

    # Корректируем матрицу поворота для обеспечения правильной ориентации
    R = Vt.T @ U.T

    # Проверяем, не является ли поворот отражением
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Находим вектор перемещения
    t = centroid_B - R @ centroid_A

    return R, t


def filter_points_ransac(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        threshold: float = 2.0,
        max_iterations: int = 100,
        min_inliers_ratio: float = 0.5
) -> np.ndarray:
    """
    Фильтрует 3D-точки с использованием алгоритма RANSAC для устранения выбросов.

    Args:
        points_3d: Массив 3D-точек shape (n_points, 3)
        points_2d: Массив 2D-точек shape (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции shape (n_cameras, 3, 4)
        threshold: Порог ошибки репроекции для определения выбросов
        max_iterations: Максимальное число итераций RANSAC
        min_inliers_ratio: Минимальное соотношение инлаеров для принятия модели

    Returns:
        np.ndarray: Маска инлаеров shape (n_points,)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Рассчитываем ошибки репроекции
    reprojection_errors = calculate_reprojection_error(points_3d, points_2d, projection_matrices, mean=True)

    # Определяем инлаеры (точки с ошибкой репроекции ниже порога)
    inliers_mask = reprojection_errors < threshold

    # Если достаточно инлаеров, возвращаем результат
    if np.mean(inliers_mask) >= min_inliers_ratio:
        return inliers_mask

    # Иначе применяем RANSAC
    best_inliers_mask = inliers_mask.copy()
    best_inliers_count = np.sum(inliers_mask)

    for _ in range(max_iterations):
        # Выбираем случайное подмножество точек
        sample_size = min(max(3, int(n_points * 0.2)), n_points // 2)
        sample_indices = np.random.choice(n_points, sample_size, replace=False)

        # Рассчитываем центроид и векторы от центроида к каждой точке
        centroid = np.mean(points_3d[sample_indices], axis=0)
        vectors = points_3d - centroid

        # Рассчитываем расстояния от каждой точки до центроида
        distances = np.linalg.norm(vectors, axis=1)

        # Нормализуем векторы
        normalized_vectors = vectors / distances[:, np.newaxis]

        # Находим косинусные расстояния между векторами
        cosine_dist = np.abs(normalized_vectors @ normalized_vectors.T)

        # Определяем инлаеры на основе косинусного расстояния и расстояния до центроида
        current_inliers = np.all(cosine_dist > 0.9, axis=1) & (np.abs(distances - np.median(distances)) < threshold)

        # Проверяем, улучшилась ли модель
        current_inliers_count = np.sum(current_inliers)
        if current_inliers_count > best_inliers_count:
            best_inliers_mask = current_inliers
            best_inliers_count = current_inliers_count

            # Если достаточно инлаеров, завершаем
            if current_inliers_count >= n_points * min_inliers_ratio:
                break

    return best_inliers_mask


def interpolate_missing_points(points: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """
    Интерполирует отсутствующие точки в траектории.

    Args:
        points: Массив точек (frames, ...) с возможными NaN
        max_gap: Максимальный размер пропуска для интерполяции

    Returns:
        np.ndarray: Массив с интерполированными значениями
    """
    # Создаем копию массива
    interpolated = points.copy()

    # Получаем размеры массива
    n_frames = points.shape[0]
    original_shape = points.shape

    # Преобразуем массив для обработки
    flat_shape = (n_frames, -1)
    points_flat = points.reshape(flat_shape)
    interpolated_flat = interpolated.reshape(flat_shape)

    n_channels = points_flat.shape[1]

    # Для каждого канала
    for channel in range(n_channels):
        # Получаем маску пропущенных значений
        missing_mask = np.isnan(points_flat[:, channel])

        if not np.any(missing_mask):
            continue

        # Находим индексы не-NaN значений
        valid_indices = np.where(~missing_mask)[0]

        if len(valid_indices) < 2:
            continue

        # Находим пропуски
        missing_ranges = []
        start_idx = None

        for i in range(1, len(valid_indices)):
            if valid_indices[i] - valid_indices[i - 1] > 1:
                start_idx = valid_indices[i - 1] + 1
                end_idx = valid_indices[i] - 1
                gap_size = end_idx - start_idx + 1
                missing_ranges.append((start_idx, end_idx, gap_size))

        # Интерполируем пропуски не больше max_gap
        for start_idx, end_idx, gap_size in missing_ranges:
            if gap_size <= max_gap:
                # Линейная интерполяция
                left_val = points_flat[start_idx - 1, channel]
                right_val = points_flat[end_idx + 1, channel]

                for i in range(start_idx, end_idx + 1):
                    t = (i - (start_idx - 1)) / (gap_size + 1)
                    interpolated_flat[i, channel] = left_val * (1 - t) + right_val * t

    # Возвращаем массив к исходной форме
    return interpolated_flat.reshape(original_shape)


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])

    # Пример поворота
    rotated_points = rotate_around_z_axis(points, 90)
    logger.info(f"Исходные точки:\n{points}")
    logger.info(f"Точки после поворота на 90° вокруг оси Z:\n{rotated_points}")

    # Пример создания и применения матрицы преобразования
    rotation = create_rotation_matrix(0, 0, np.radians(90))
    translation = np.array([1, 2, 3])
    transform = create_transformation_matrix(rotation, translation)

    transformed_points = apply_transformation(points, transform)
    logger.info(f"Точки после преобразования:\n{transformed_points}")

    # Пример интерполяции
    trajectory = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [np.nan, np.nan, np.nan],  # Пропуск
        [np.nan, np.nan, np.nan],  # Пропуск
        [5, 5, 5]
    ])

    interpolated = interpolate_missing_points(trajectory)
    logger.info(f"Исходная траектория:\n{trajectory}")
    logger.info(f"Интерполированная траектория:\n{interpolated}")