"""
Модуль для триангуляции 2D-точек в 3D-пространство.

Содержит функции и классы для преобразования 2D-координат с нескольких камер
в 3D-координаты с использованием различных методов триангуляции.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.utils.geometry import triangulate_points, calculate_reprojection_error, filter_points_ransac

logger = logging.getLogger(__name__)


class Triangulator:
    """
    Класс для триангуляции 2D-точек с нескольких камер в 3D-пространство.

    Attributes:
        camera_matrices (List[np.ndarray]): Матрицы проекции для каждой камеры (3x4)
        camera_names (List[str]): Имена камер
        use_ransac (bool): Использовать RANSAC для фильтрации выбросов
        ransac_threshold (float): Порог ошибки репроекции для RANSAC
        min_cameras (int): Минимальное количество камер для триангуляции
    """

    def __init__(
            self,
            camera_matrices: List[np.ndarray],
            camera_names: Optional[List[str]] = None,
            use_ransac: bool = False,
            ransac_threshold: float = 2.0,
            min_cameras: int = 2
    ):
        """
        Инициализирует объект для триангуляции.

        Args:
            camera_matrices: Список матриц проекции камер (каждая размера 3x4)
            camera_names: Список имен камер (если None, используются номера)
            use_ransac: Использовать RANSAC для фильтрации выбросов
            ransac_threshold: Порог ошибки репроекции для RANSAC
            min_cameras: Минимальное количество камер для триангуляции
        """
        if len(camera_matrices) < 2:
            raise ValueError("Для триангуляции требуется как минимум две камеры")

        self.camera_matrices = np.array(camera_matrices)

        # Проверка размерности матриц камер
        for i, mat in enumerate(camera_matrices):
            if mat.shape != (3, 4):
                raise ValueError(f"Матрица камеры {i} имеет неверный размер: {mat.shape}, ожидается (3, 4)")

        # Назначение имен камер
        if camera_names is None:
            self.camera_names = [f"camera_{i}" for i in range(len(camera_matrices))]
        else:
            if len(camera_names) != len(camera_matrices):
                raise ValueError("Количество имен камер не соответствует количеству матриц камер")
            self.camera_names = camera_names

        self.use_ransac = use_ransac
        self.ransac_threshold = ransac_threshold
        self.min_cameras = min_cameras

        logger.info(f"Инициализирован триангулятор с {len(camera_matrices)} камерами")
        if use_ransac:
            logger.info(f"Включена фильтрация выбросов RANSAC с порогом {ransac_threshold}")

    def triangulate_points(
            self,
            points_2d: np.ndarray,
            visibility: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Выполняет триангуляцию набора 2D-точек в 3D-пространство.

        Args:
            points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
            visibility: Маска видимости точек размера (n_cameras, n_points)
                        Если None, точки считаются видимыми, если не NaN

        Returns:
            np.ndarray: Массив 3D-точек размера (n_points, 3)
        """
        n_cameras, n_points, _ = points_2d.shape

        # Если маска видимости не предоставлена, создаем её на основе NaN
        if visibility is None:
            visibility = ~np.isnan(points_2d[:, :, 0])

        # Проверки входных данных
        if n_cameras != len(self.camera_matrices):
            raise ValueError(f"Количество камер в точках ({n_cameras}) не соответствует "
                             f"количеству матриц камер ({len(self.camera_matrices)})")

        # Массив для хранения результатов
        points_3d = np.full((n_points, 3), np.nan)

        # Триангулируем каждую точку
        for point_idx in range(n_points):
            # Получаем видимость точки для каждой камеры
            point_visibility = visibility[:, point_idx]

            # Количество камер, видящих точку
            n_visible_cameras = np.sum(point_visibility)

            # Если точка видна менее чем min_cameras камерам, не триангулируем
            if n_visible_cameras < self.min_cameras:
                continue

            # Получаем 2D координаты и матрицы проекции для камер, видящих точку
            valid_cameras = np.where(point_visibility)[0]
            valid_points = points_2d[valid_cameras, point_idx, :]
            valid_matrices = self.camera_matrices[valid_cameras]

            # Выполняем триангуляцию
            point_3d = triangulate_points(
                valid_points.reshape(n_visible_cameras, 1, 2),
                valid_matrices,
                min_cameras=self.min_cameras
            )[0]  # Берем первую точку, так как мы реконструируем только одну

            # Если включен RANSAC, фильтруем выбросы
            if self.use_ransac and n_visible_cameras > self.min_cameras + 1:
                # Проецируем 3D-точку обратно на изображения камер
                projected_2d = project_3d_to_2d(point_3d.reshape(1, 3), valid_matrices)

                # Вычисляем ошибку репроекции
                reproj_errors = np.linalg.norm(projected_2d - valid_points, axis=1)

                # Отбрасываем камеры с большой ошибкой репроекции
                good_cameras = reproj_errors < self.ransac_threshold

                # Если осталось достаточно камер, повторно триангулируем
                if np.sum(good_cameras) >= self.min_cameras:
                    point_3d = triangulate_points(
                        valid_points[good_cameras].reshape(-1, 1, 2),
                        valid_matrices[good_cameras],
                        min_cameras=self.min_cameras
                    )[0]

            # Сохраняем результат
            points_3d[point_idx] = point_3d

        return points_3d

    def triangulate_points_batch(
            self,
            points_2d_batch: np.ndarray,
            visibility_batch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Выполняет триангуляцию для пакета кадров с 2D-точками.

        Args:
            points_2d_batch: Массив 2D-точек размера (n_frames, n_cameras, n_points, 2)
            visibility_batch: Маска видимости точек размера (n_frames, n_cameras, n_points)
                              Если None, точки считаются видимыми, если не NaN

        Returns:
            np.ndarray: Массив 3D-точек размера (n_frames, n_points, 3)
        """
        n_frames, n_cameras, n_points, _ = points_2d_batch.shape

        # Если маска видимости не предоставлена, создаем её на основе NaN
        if visibility_batch is None:
            visibility_batch = ~np.isnan(points_2d_batch[:, :, :, 0])

        # Массив для хранения результатов
        points_3d_batch = np.full((n_frames, n_points, 3), np.nan)

        # Обрабатываем каждый кадр отдельно
        for frame_idx in range(n_frames):
            points_2d = points_2d_batch[frame_idx]
            visibility = visibility_batch[frame_idx] if visibility_batch is not None else None

            points_3d_batch[frame_idx] = self.triangulate_points(points_2d, visibility)

        return points_3d_batch

    def calculate_reprojection_errors(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            mean: bool = True
    ) -> np.ndarray:
        """
        Вычисляет ошибки репроекции для 3D-точек.

        Args:
            points_3d: Массив 3D-точек размера (n_points, 3)
            points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
            mean: Если True, возвращает среднюю ошибку для каждой точки

        Returns:
            np.ndarray: Ошибки репроекции
                Если mean=True: размера (n_points,)
                Если mean=False: размера (n_cameras, n_points, 2)
        """
        return calculate_reprojection_error(
            points_3d,
            points_2d,
            self.camera_matrices,
            mean=mean
        )

    def filter_outliers(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            max_error: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Фильтрует выбросы на основе ошибки репроекции.

        Args:
            points_3d: Массив 3D-точек размера (n_points, 3)
            points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
            max_error: Максимально допустимая ошибка репроекции
                       Если None, используется self.ransac_threshold

        Returns:
            Tuple[np.ndarray, np.ndarray]: Отфильтрованные 3D-точки и маска хороших точек
        """
        if max_error is None:
            max_error = self.ransac_threshold

        # Вычисляем ошибки репроекции
        errors = self.calculate_reprojection_errors(points_3d, points_2d, mean=True)

        # Создаем маску для точек с приемлемой ошибкой репроекции
        good_points_mask = errors < max_error

        # Копируем 3D-точки и заменяем выбросы на NaN
        filtered_points_3d = points_3d.copy()
        filtered_points_3d[~good_points_mask] = np.nan

        return filtered_points_3d, good_points_mask


def project_3d_to_2d(points_3d: np.ndarray, projection_matrices: np.ndarray) -> np.ndarray:
    """
    Проецирует 3D-точки на плоскость изображения с использованием матриц проекции.

    Args:
        points_3d: Массив 3D-точек размера (n_points, 3)
        projection_matrices: Матрицы проекции размера (n_cameras, 3, 4)

    Returns:
        np.ndarray: Массив спроецированных 2D-точек размера (n_cameras, n_points, 2)
    """
    n_cameras = projection_matrices.shape[0]
    n_points = points_3d.shape[0]

    # Преобразуем 3D-точки в однородные координаты
    points_3d_homogeneous = np.hstack((points_3d, np.ones((n_points, 1))))

    # Массив для хранения результатов
    points_2d = np.zeros((n_cameras, n_points, 2))

    # Для каждой камеры
    for cam_idx in range(n_cameras):
        # Проецируем 3D-точки на плоскость камеры
        projected_points_homogeneous = np.dot(points_3d_homogeneous, projection_matrices[cam_idx].T)

        # Переходим к неоднородным координатам
        points_2d[cam_idx] = projected_points_homogeneous[:, :2] / projected_points_homogeneous[:, 2:]

    return points_2d


def linear_triangulation(
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        min_cameras: int = 2
) -> np.ndarray:
    """
    Выполняет линейную триангуляцию для набора 2D-точек.

    Реализация использует метод DLT (Direct Linear Transform).

    Args:
        points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции размера (n_cameras, 3, 4)
        min_cameras: Минимальное количество камер для триангуляции

    Returns:
        np.ndarray: Массив 3D-точек размера (n_points, 3)
    """
    return triangulate_points(points_2d, projection_matrices, min_cameras=min_cameras)


def robust_triangulation(
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        threshold: float = 2.0,
        min_cameras: int = 2,
        max_iterations: int = 100
) -> np.ndarray:
    """
    Выполняет устойчивую триангуляцию с использованием RANSAC.

    Args:
        points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции размера (n_cameras, 3, 4)
        threshold: Порог ошибки репроекции для RANSAC
        min_cameras: Минимальное количество камер для триангуляции
        max_iterations: Максимальное число итераций RANSAC

    Returns:
        np.ndarray: Массив 3D-точек размера (n_points, 3)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Создаем массив для хранения результатов
    points_3d = np.full((n_points, 3), np.nan)

    # Выполняем исходную триангуляцию
    initial_points_3d = linear_triangulation(points_2d, projection_matrices, min_cameras=min_cameras)

    # Применяем RANSAC для каждой точки
    for point_idx in range(n_points):
        # Проверяем, что точка видна хотя бы на min_cameras камерах
        valid_cameras = ~np.isnan(points_2d[:, point_idx, 0])
        n_valid_cameras = np.sum(valid_cameras)

        if n_valid_cameras < min_cameras:
            continue

        # Если точка видна только на min_cameras камерах, используем обычную триангуляцию
        if n_valid_cameras == min_cameras:
            points_3d[point_idx] = initial_points_3d[point_idx]
            continue

        # Применяем RANSAC
        # 1. Исходная 3D-точка
        best_point_3d = initial_points_3d[point_idx]

        # 2. Вычисляем ошибку репроекции для всех камер
        valid_points_2d = points_2d[valid_cameras, point_idx]
        valid_matrices = projection_matrices[valid_cameras]

        # Проецируем 3D-точку обратно на изображения
        projected_2d = project_3d_to_2d(
            best_point_3d.reshape(1, 3),
            valid_matrices
        )[0]  # Учитываем, что у нас одна точка

        # Вычисляем ошибки репроекции
        errors = np.linalg.norm(projected_2d - valid_points_2d, axis=1)

        # 3. Итеративно улучшаем решение
        for _ in range(max_iterations):
            # Находим камеры с ошибкой репроекции ниже порога
            inliers = errors < threshold

            # Если мало инлаеров, завершаем
            if np.sum(inliers) < min_cameras:
                break

            # Выполняем триангуляцию только с инлаерами
            point_3d = linear_triangulation(
                valid_points_2d[inliers].reshape(-1, 1, 2),
                valid_matrices[inliers],
                min_cameras=min_cameras
            )[0]

            # Проецируем новую 3D-точку обратно
            new_projected_2d = project_3d_to_2d(
                point_3d.reshape(1, 3),
                valid_matrices
            )[0]

            # Вычисляем новые ошибки
            new_errors = np.linalg.norm(new_projected_2d - valid_points_2d, axis=1)

            # Если среднее ошибок уменьшилось, обновляем результат
            if np.mean(new_errors) < np.mean(errors):
                best_point_3d = point_3d
                errors = new_errors
            else:
                # Решение не улучшается, завершаем
                break

        # Сохраняем наилучшее решение
        points_3d[point_idx] = best_point_3d

    return points_3d


def midpoint_triangulation(
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        min_cameras: int = 2
) -> np.ndarray:
    """
    Выполняет триангуляцию методом средней точки.

    Находит 3D-точку, которая минимизирует сумму расстояний до лучей,
    проведенных из центров камер через соответствующие 2D-точки.

    Args:
        points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции размера (n_cameras, 3, 4)
        min_cameras: Минимальное количество камер для триангуляции

    Returns:
        np.ndarray: Массив 3D-точек размера (n_points, 3)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Получаем центры камер
    camera_centers = []
    for P in projection_matrices:
        # Находим центр камеры как решение системы P*C = 0
        # Для калиброванной камеры центр находится как -R^T * t
        # Где R - матрица поворота, t - вектор перемещения
        # Используем SVD для поиска решения
        _, _, Vh = np.linalg.svd(P)
        C = Vh[-1, :] / Vh[-1, -1]  # Последняя строка V^T, нормализованная
        camera_centers.append(C[:3])  # Отбрасываем однородную координату

    camera_centers = np.array(camera_centers)

    # Массив для хранения результатов
    points_3d = np.full((n_points, 3), np.nan)

    # Обрабатываем каждую точку
    for point_idx in range(n_points):
        # Проверяем, что точка видна хотя бы на min_cameras камерах
        valid_cameras = ~np.isnan(points_2d[:, point_idx, 0])
        n_valid_cameras = np.sum(valid_cameras)

        if n_valid_cameras < min_cameras:
            continue

        # Получаем индексы валидных камер
        valid_cam_indices = np.where(valid_cameras)[0]

        # Создаем матрицу уравнений и вектор правой части
        A = np.zeros((3 * n_valid_cameras, 3 + n_valid_cameras))
        b = np.zeros(3 * n_valid_cameras)

        for i, cam_idx in enumerate(valid_cam_indices):
            # Получаем матрицу проекции и центр камеры
            P = projection_matrices[cam_idx][:3, :3]  # Матрица поворота
            C = camera_centers[cam_idx]

            # Получаем луч из центра камеры через 2D-точку
            x, y = points_2d[cam_idx, point_idx]

            # Вычисляем направление луча
            ray_direction = np.linalg.inv(P) @ np.array([x, y, 1])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Заполняем матрицу уравнений для этой камеры
            A[3 * i:3 * (i + 1), :3] = np.eye(3)
            A[3 * i:3 * (i + 1), 3 + i] = -ray_direction

            # Заполняем правую часть
            b[3 * i:3 * (i + 1)] = C

        # Решаем систему методом наименьших квадратов
        try:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            points_3d[point_idx] = x[:3]  # Первые три элемента - это 3D-точка
        except np.linalg.LinAlgError:
            # Если не удается решить систему, оставляем NaN
            pass

    return points_3d


def triangulate_from_camera_calibrations(
        points_2d: np.ndarray,
        camera_calibrations: List[Dict[str, Any]],
        camera_names: Optional[List[str]] = None,
        method: str = "linear",
        **kwargs
) -> np.ndarray:
    """
    Выполняет триангуляцию на основе калибровок камер.

    Args:
        points_2d: Массив 2D-точек размера (n_cameras, n_points, 2)
        camera_calibrations: Список словарей с параметрами калибровки камер
                             Каждый словарь должен содержать 'camera_matrix', 'dist_coeffs',
                             'rvecs' и 'tvecs'
        camera_names: Список имен камер
        method: Метод триангуляции ('linear', 'robust', 'midpoint')
        **kwargs: Дополнительные параметры для выбранного метода

    Returns:
        np.ndarray: Массив 3D-точек размера (n_points, 3)
    """
    if len(camera_calibrations) < 2:
        raise ValueError("Для триангуляции требуется как минимум две камеры")

    if len(camera_calibrations) != points_2d.shape[0]:
        raise ValueError("Количество калибровок камер не соответствует количеству камер в точках")

    # Создаем матрицы проекции для всех камер
    projection_matrices = []

    for calib in camera_calibrations:
        # Извлекаем параметры калибровки
        camera_matrix = np.array(calib['camera_matrix'])

        # Используем первый вектор поворота и перемещения, если их несколько
        rvec = np.array(calib['rvecs'][0]) if isinstance(calib['rvecs'], list) else np.array(calib['rvecs'])
        tvec = np.array(calib['tvecs'][0]) if isinstance(calib['tvecs'], list) else np.array(calib['tvecs'])

        # Создаем матрицу поворота из вектора Родригеса
        R, _ = cv2.Rodrigues(rvec)

        # Создаем матрицу внешних параметров [R|t]
        RT = np.hstack((R, tvec.reshape(3, 1)))

        # Создаем матрицу проекции P = K[R|t]
        P = camera_matrix @ RT

        projection_matrices.append(P)

    projection_matrices = np.array(projection_matrices)

    # Выбираем метод триангуляции
    if method == "linear":
        return linear_triangulation(points_2d, projection_matrices, **kwargs)
    elif method == "robust":
        return robust_triangulation(points_2d, projection_matrices, **kwargs)
    elif method == "midpoint":
        return midpoint_triangulation(points_2d, projection_matrices, **kwargs)
    else:
        raise ValueError(f"Неизвестный метод триангуляции: {method}")


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    # Две камеры, наблюдающие три точки
    camera_matrices = [
        np.array([[1000, 0, 320, 0],
                  [0, 1000, 240, 0],
                  [0, 0, 1, 0]]),
        np.array([[1000, 0, 320, 0],
                  [0, 1000, 240, 0],
                  [0, 0, 1, 0]]) @ np.array([[1, 0, 0, 100],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])[:3]
    ]

    # Создаем 3D-точки для тестирования
    true_points_3d = np.array([
        [0, 0, 100],
        [50, 0, 120],
        [0, 50, 80]
    ])

    # Проецируем 3D-точки на плоскость камер
    points_2d = project_3d_to_2d(true_points_3d, np.array(camera_matrices))

    # Создаем триангулятор
    triangulator = Triangulator(camera_matrices)

    # Выполняем триангуляцию
    reconstructed_points_3d = triangulator.triangulate_points(points_2d)

    # Выводим результаты
    logger.info("Истинные 3D-точки:")
    for point in true_points_3d:
        logger.info(f"  {point}")

    logger.info("Реконструированные 3D-точки:")
    for point in reconstructed_points_3d:
        logger.info(f"  {point}")

    # Вычисляем ошибку
    error = np.linalg.norm(true_points_3d - reconstructed_points_3d, axis=1)
    logger.info("Ошибка реконструкции:")
    for e in error:
        logger.info(f"  {e}")