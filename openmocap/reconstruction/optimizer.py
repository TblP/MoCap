"""
Модуль для оптимизации 3D-реконструкций.

Содержит классы и функции для улучшения качества 3D-данных,
полученных триангуляцией из 2D-точек камер, путем минимизации ошибок
репроекции и применения различных ограничений.
"""

import logging
import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.utils.geometry import project_3d_to_2d
#from openmocap.reconstruction.reprojection import calculate_reprojection_error

logger = logging.getLogger(__name__)


class PointCloudOptimizer:
    """
    Класс для оптимизации облака 3D-точек.

    Выполняет оптимизацию 3D-точек путем минимизации ошибок репроекции
    и применения различных ограничений, таких как постоянство длин сегментов.

    Attributes:
        projection_matrices (np.ndarray): Матрицы проекции камер
        optimization_method (str): Метод оптимизации
        max_iterations (int): Максимальное число итераций
        convergence_tolerance (float): Порог сходимости
        verbose (bool): Подробный вывод информации
    """

    def __init__(
            self,
            projection_matrices: np.ndarray,
            optimization_method: str = 'levenberg_marquardt',
            max_iterations: int = 100,
            convergence_tolerance: float = 1e-6,
            verbose: bool = False
    ):
        """
        Инициализирует оптимизатор облака точек.

        Args:
            projection_matrices: Матрицы проекции камер размера (n_cameras, 3, 4)
            optimization_method: Метод оптимизации ('levenberg_marquardt', 'gauss_newton', 'lbfgs')
            max_iterations: Максимальное число итераций
            convergence_tolerance: Порог сходимости
            verbose: Подробный вывод информации
        """
        self.projection_matrices = projection_matrices
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.verbose = verbose

        # Проверка формы матриц проекции
        if projection_matrices.ndim != 3 or projection_matrices.shape[1:] != (3, 4):
            raise ValueError(
                f"Неверная форма projection_matrices: {projection_matrices.shape}, "
                f"ожидается (n_cameras, 3, 4)"
            )

        self.n_cameras = projection_matrices.shape[0]

        logger.info(f"Инициализирован оптимизатор с {self.n_cameras} камерами")

    def optimize_points(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            visibility: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
            segment_lengths: Optional[Dict[Tuple[int, int], float]] = None
    ) -> np.ndarray:
        """
        Оптимизирует 3D-точки для минимизации ошибки репроекции.

        Args:
            points_3d: Исходные 3D-точки размера (n_points, 3)
            points_2d: 2D-точки размера (n_cameras, n_points, 2)
            visibility: Маска видимости точек размера (n_cameras, n_points)
            weights: Веса для каждой точки размера (n_cameras, n_points)
            segment_lengths: Словарь с длинами сегментов скелета {(start_idx, end_idx): length}

        Returns:
            np.ndarray: Оптимизированные 3D-точки размера (n_points, 3)
        """
        n_points = points_3d.shape[0]

        # Проверка размерности входных данных
        if points_2d.shape[0] != self.n_cameras or points_2d.shape[1] != n_points:
            raise ValueError(
                f"Неверная форма points_2d: {points_2d.shape}, "
                f"ожидается ({self.n_cameras}, {n_points}, 2)"
            )

        # Если маска видимости не предоставлена, создаем её из points_2d
        if visibility is None:
            visibility = ~np.isnan(points_2d[:, :, 0])

        # Если веса не предоставлены, устанавливаем единичные веса для всех видимых точек
        if weights is None:
            weights = np.ones((self.n_cameras, n_points))
            weights[~visibility] = 0.0

        # Функция потерь для оптимизации
        def objective_function(x):
            # Преобразуем плоский вектор обратно в массив 3D-точек
            p3d = x.reshape(-1, 3)

            # Проецируем 3D-точки на 2D
            projected_points = project_3d_to_2d(p3d, self.projection_matrices)

            # Вычисляем ошибки репроекции
            errors = points_2d - projected_points

            # Применяем маску видимости и веса
            weighted_errors = errors * weights[:, :, np.newaxis] * visibility[:, :, np.newaxis]

            # Добавляем ограничения на длины сегментов, если они предоставлены
            bone_errors = []
            if segment_lengths:
                for (start_idx, end_idx), length in segment_lengths.items():
                    if start_idx < n_points and end_idx < n_points:
                        start_point = p3d[start_idx]
                        end_point = p3d[end_idx]
                        current_length = np.linalg.norm(end_point - start_point)
                        bone_errors.append((current_length - length) ** 2)

            # Объединяем все ошибки
            all_errors = np.concatenate([
                weighted_errors.reshape(-1),
                np.array(bone_errors) * 10.0  # Увеличиваем вес для ограничений длин
            ])

            return all_errors

        # Начальное приближение - плоский вектор из points_3d
        x0 = points_3d.reshape(-1)

        # Выбор метода оптимизации
        if self.optimization_method == 'levenberg_marquardt':
            # Метод Левенберга-Марквардта (для нелинейных наименьших квадратов)
            result = optimize.least_squares(
                objective_function,
                x0,
                method='lm',
                ftol=self.convergence_tolerance,
                max_nfev=self.max_iterations,
                verbose=2 if self.verbose else 0
            )
        elif self.optimization_method == 'gauss_newton':
            # Метод Гаусса-Ньютона (для нелинейных наименьших квадратов)
            result = optimize.least_squares(
                objective_function,
                x0,
                method='trf',  # Trust Region Reflective
                ftol=self.convergence_tolerance,
                max_nfev=self.max_iterations,
                verbose=2 if self.verbose else 0
            )
        elif self.optimization_method == 'lbfgs':
            # Метод L-BFGS (для общей оптимизации)
            def squared_objective(x):
                errors = objective_function(x)
                return np.sum(errors ** 2)

            result = optimize.minimize(
                squared_objective,
                x0,
                method='L-BFGS-B',
                tol=self.convergence_tolerance,
                options={'maxiter': self.max_iterations, 'disp': self.verbose}
            )
        else:
            raise ValueError(f"Неподдерживаемый метод оптимизации: {self.optimization_method}")

        # Преобразуем оптимизированные параметры обратно в массив 3D-точек
        optimized_points_3d = result.x.reshape(-1, 3)

        # Вычисляем итоговую ошибку репроекции
        projected_points = project_3d_to_2d(optimized_points_3d, self.projection_matrices)
        initial_error = np.mean(
            np.linalg.norm(points_2d - project_3d_to_2d(points_3d, self.projection_matrices), axis=2)[visibility]
        )
        final_error = np.mean(np.linalg.norm(points_2d - projected_points, axis=2)[visibility])

        logger.info(f"Оптимизация завершена. Начальная ошибка: {initial_error:.4f}, "
                    f"конечная ошибка: {final_error:.4f}")

        return optimized_points_3d

    def optimize_sequence(
            self,
            points_3d_sequence: np.ndarray,
            points_2d_sequence: np.ndarray,
            visibility_sequence: Optional[np.ndarray] = None,
            weights_sequence: Optional[np.ndarray] = None,
            segment_lengths: Optional[Dict[Tuple[int, int], float]] = None,
            temporal_smoothness_weight: float = 0.1
    ) -> np.ndarray:
        """
        Оптимизирует последовательность 3D-точек с учетом временной согласованности.

        Args:
            points_3d_sequence: Исходные 3D-точки размера (n_frames, n_points, 3)
            points_2d_sequence: 2D-точки размера (n_frames, n_cameras, n_points, 2)
            visibility_sequence: Маска видимости точек размера (n_frames, n_cameras, n_points)
            weights_sequence: Веса для каждой точки размера (n_frames, n_cameras, n_points)
            segment_lengths: Словарь с длинами сегментов скелета {(start_idx, end_idx): length}
            temporal_smoothness_weight: Вес для ограничения временной гладкости

        Returns:
            np.ndarray: Оптимизированные 3D-точки размера (n_frames, n_points, 3)
        """
        n_frames, n_points, _ = points_3d_sequence.shape

        # Проверка размерности входных данных
        if points_2d_sequence.shape[0] != n_frames or points_2d_sequence.shape[1] != self.n_cameras or \
                points_2d_sequence.shape[2] != n_points:
            raise ValueError(
                f"Неверная форма points_2d_sequence: {points_2d_sequence.shape}, "
                f"ожидается ({n_frames}, {self.n_cameras}, {n_points}, 2)"
            )

        # Если маска видимости не предоставлена, создаем её из points_2d_sequence
        if visibility_sequence is None:
            visibility_sequence = ~np.isnan(points_2d_sequence[:, :, :, 0])

        # Если веса не предоставлены, устанавливаем единичные веса для всех видимых точек
        if weights_sequence is None:
            weights_sequence = np.ones((n_frames, self.n_cameras, n_points))
            weights_sequence[~visibility_sequence] = 0.0

        # Функция потерь для оптимизации
        def objective_function(x):
            # Преобразуем плоский вектор обратно в последовательность 3D-точек
            p3d_sequence = x.reshape(n_frames, n_points, 3)

            # Ошибки репроекции
            reprojection_errors = []
            for frame_idx in range(n_frames):
                p3d = p3d_sequence[frame_idx]
                points_2d = points_2d_sequence[frame_idx]
                visibility = visibility_sequence[frame_idx]
                weights = weights_sequence[frame_idx]

                # Проецируем 3D-точки на 2D
                projected_points = project_3d_to_2d(p3d, self.projection_matrices)

                # Вычисляем ошибки репроекции
                errors = points_2d - projected_points

                # Применяем маску видимости и веса
                weighted_errors = errors * weights[:, :, np.newaxis] * visibility[:, :, np.newaxis]
                reprojection_errors.append(weighted_errors.reshape(-1))

            # Ограничения на длины сегментов
            bone_errors = []
            if segment_lengths:
                for frame_idx in range(n_frames):
                    p3d = p3d_sequence[frame_idx]
                    for (start_idx, end_idx), length in segment_lengths.items():
                        if start_idx < n_points and end_idx < n_points:
                            start_point = p3d[start_idx]
                            end_point = p3d[end_idx]
                            current_length = np.linalg.norm(end_point - start_point)
                            bone_errors.append((current_length - length) ** 2)

            # Ограничения временной гладкости
            smoothness_errors = []
            if n_frames > 2:
                for frame_idx in range(1, n_frames - 1):
                    p3d_prev = p3d_sequence[frame_idx - 1]
                    p3d_curr = p3d_sequence[frame_idx]
                    p3d_next = p3d_sequence[frame_idx + 1]

                    # Ошибка гладкости: насколько текущий кадр отклоняется от среднего соседних
                    expected_p3d = (p3d_prev + p3d_next) / 2.0
                    smoothness_error = (p3d_curr - expected_p3d) * temporal_smoothness_weight
                    smoothness_errors.append(smoothness_error.reshape(-1))

            # Объединяем все ошибки
            all_errors = np.concatenate(
                reprojection_errors +
                [np.array(bone_errors) * 10.0] +  # Увеличиваем вес для ограничений длин
                smoothness_errors
            )

            return all_errors

        # Начальное приближение - плоский вектор из points_3d_sequence
        x0 = points_3d_sequence.reshape(-1)

        # Оптимизация
        if self.optimization_method == 'levenberg_marquardt':
            # Метод Левенберга-Марквардта (для нелинейных наименьших квадратов)
            result = optimize.least_squares(
                objective_function,
                x0,
                method='lm',
                ftol=self.convergence_tolerance,
                max_nfev=self.max_iterations,
                verbose=2 if self.verbose else 0
            )
        elif self.optimization_method == 'gauss_newton':
            # Метод Гаусса-Ньютона (для нелинейных наименьших квадратов)
            result = optimize.least_squares(
                objective_function,
                x0,
                method='trf',  # Trust Region Reflective
                ftol=self.convergence_tolerance,
                max_nfev=self.max_iterations,
                verbose=2 if self.verbose else 0
            )
        elif self.optimization_method == 'lbfgs':
            # Метод L-BFGS (для общей оптимизации)
            def squared_objective(x):
                errors = objective_function(x)
                return np.sum(errors ** 2)

            result = optimize.minimize(
                squared_objective,
                x0,
                method='L-BFGS-B',
                tol=self.convergence_tolerance,
                options={'maxiter': self.max_iterations, 'disp': self.verbose}
            )
        else:
            raise ValueError(f"Неподдерживаемый метод оптимизации: {self.optimization_method}")

        # Преобразуем оптимизированные параметры обратно в последовательность 3D-точек
        optimized_points_3d_sequence = result.x.reshape(n_frames, n_points, 3)

        # Вычисляем итоговую ошибку репроекции
        initial_error = 0.0
        final_error = 0.0
        for frame_idx in range(n_frames):
            p3d_initial = points_3d_sequence[frame_idx]
            p3d_final = optimized_points_3d_sequence[frame_idx]
            points_2d = points_2d_sequence[frame_idx]
            visibility = visibility_sequence[frame_idx]

            initial_error += np.mean(
                np.linalg.norm(points_2d - project_3d_to_2d(p3d_initial, self.projection_matrices), axis=2)[visibility]
            )
            final_error += np.mean(
                np.linalg.norm(points_2d - project_3d_to_2d(p3d_final, self.projection_matrices), axis=2)[visibility]
            )

        initial_error /= n_frames
        final_error /= n_frames

        logger.info(f"Оптимизация последовательности завершена. Начальная ошибка: {initial_error:.4f}, "
                    f"конечная ошибка: {final_error:.4f}")

        return optimized_points_3d_sequence


class BundleAdjustment:
    """
    Реализация bundle adjustment для одновременной оптимизации 3D-точек и параметров камер.

    Bundle adjustment - это процесс совместной оптимизации положения 3D-точек
    и параметров камер (внутренних и внешних) для минимизации ошибки репроекции.

    Attributes:
        camera_matrices (np.ndarray): Начальные матрицы камер
        optimize_intrinsics (bool): Оптимизировать внутренние параметры камер
        optimize_extrinsics (bool): Оптимизировать внешние параметры камер
        max_iterations (int): Максимальное число итераций
        convergence_tolerance (float): Порог сходимости
        verbose (bool): Подробный вывод информации
    """

    def __init__(
            self,
            camera_matrices: np.ndarray,
            optimize_intrinsics: bool = False,
            optimize_extrinsics: bool = True,
            max_iterations: int = 100,
            convergence_tolerance: float = 1e-6,
            verbose: bool = False
    ):
        """
        Инициализирует bundle adjustment.

        Args:
            camera_matrices: Матрицы проекции камер размера (n_cameras, 3, 4)
            optimize_intrinsics: Оптимизировать внутренние параметры камер
            optimize_extrinsics: Оптимизировать внешние параметры камер
            max_iterations: Максимальное число итераций
            convergence_tolerance: Порог сходимости
            verbose: Подробный вывод информации
        """
        self.camera_matrices = camera_matrices
        self.optimize_intrinsics = optimize_intrinsics
        self.optimize_extrinsics = optimize_extrinsics
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.verbose = verbose

        # Проверка формы матриц проекции
        if camera_matrices.ndim != 3 or camera_matrices.shape[1:] != (3, 4):
            raise ValueError(
                f"Неверная форма camera_matrices: {camera_matrices.shape}, "
                f"ожидается (n_cameras, 3, 4)"
            )

        self.n_cameras = camera_matrices.shape[0]

        # Разделяем матрицы проекции на внутренние и внешние параметры
        self.camera_intrinsics = []  # (fx, fy, cx, cy)
        self.camera_extrinsics = []  # (rx, ry, rz, tx, ty, tz)

        for cam_idx in range(self.n_cameras):
            # Извлекаем внутренние параметры (упрощенно)
            # Предполагаем, что матрица проекции имеет вид P = K[R|t]
            # где K - матрица внутренних параметров, [R|t] - матрица внешних параметров

            # Предположим P = [M | p4], где M - подматрица 3x3, p4 - четвертый столбец
            P = camera_matrices[cam_idx]
            M = P[:, :3]
            p4 = P[:, 3]

            # K в верхнетреугольной форме, R - ортогональная
            K, R = self._rq_decomposition_3x3(M)

            # Для положительных диагональных элементов K
            T = np.diag(np.sign(np.diag(K)))
            K = K @ T
            R = T @ R

            # Получаем внутренние параметры
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            self.camera_intrinsics.append([fx, fy, cx, cy])

            # Получаем внешние параметры
            # Вектор перемещения t = -R^T * p4, если M = KR
            t = -R.T @ np.linalg.inv(K) @ p4

            # Преобразуем матрицу поворота в вектор поворота (ось-угол)
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(R).as_rotvec()

            self.camera_extrinsics.append([r[0], r[1], r[2], t[0], t[1], t[2]])

        self.camera_intrinsics = np.array(self.camera_intrinsics)
        self.camera_extrinsics = np.array(self.camera_extrinsics)

        logger.info(f"Инициализирован bundle adjustment с {self.n_cameras} камерами")

    def _rq_decomposition_3x3(self, M):
        """
        Выполняет RQ-разложение матрицы 3x3.

        Args:
            M: Матрица 3x3 для разложения

        Returns:
            tuple: (R, Q), где R - верхнетреугольная матрица, Q - ортогональная матрица
        """
        # Используем QR-разложение для транспонированной матрицы M
        Q, R = np.linalg.qr(M.T)
        R = R.T
        Q = Q.T

        # Хотим R с положительными диагональными элементами
        D = np.diag(np.sign(np.diag(R)))

        # Если определитель R отрицательный, отражаем последний столбец Q
        if np.linalg.det(D) < 0:
            D[-1, -1] = -D[-1, -1]
            Q[:, -1] = -Q[:, -1]

        R = D @ R
        Q = Q @ D

        return R, Q

    def _reconstruct_camera_matrix(self, intrinsics, extrinsics):
        """
        Восстанавливает матрицу проекции из внутренних и внешних параметров.

        Args:
            intrinsics: Внутренние параметры [fx, fy, cx, cy]
            extrinsics: Внешние параметры [rx, ry, rz, tx, ty, tz]

        Returns:
            np.ndarray: Матрица проекции 3x4
        """
        fx, fy, cx, cy = intrinsics
        rx, ry, rz, tx, ty, tz = extrinsics

        # Внутренняя матрица K
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Внешняя матрица [R|t]
        # Создаем матрицу поворота из вектора поворота
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
        t = np.array([tx, ty, tz])

        # Матрица RT [R|t]
        RT = np.zeros((3, 4))
        RT[:3, :3] = R
        RT[:3, 3] = t

        # Матрица проекции P = K[R|t]
        P = K @ RT

        return P

    def optimize(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            visibility: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оптимизирует 3D-точки и параметры камер одновременно.

        Args:
            points_3d: Исходные 3D-точки размера (n_points, 3)
            points_2d: 2D-точки размера (n_cameras, n_points, 2)
            visibility: Маска видимости точек размера (n_cameras, n_points)
            weights: Веса для каждой точки размера (n_cameras, n_points)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Оптимизированные 3D-точки размера (n_points, 3)
                - Оптимизированные матрицы проекции размера (n_cameras, 3, 4)
        """
        n_points = points_3d.shape[0]

        # Проверка размерности входных данных
        if points_2d.shape[0] != self.n_cameras or points_2d.shape[1] != n_points:
            raise ValueError(
                f"Неверная форма points_2d: {points_2d.shape}, "
                f"ожидается ({self.n_cameras}, {n_points}, 2)"
            )

        # Если маска видимости не предоставлена, создаем её из points_2d
        if visibility is None:
            visibility = ~np.isnan(points_2d[:, :, 0])

        # Если веса не предоставлены, устанавливаем единичные веса для всех видимых точек
        if weights is None:
            weights = np.ones((self.n_cameras, n_points))
            weights[~visibility] = 0.0

        # Создаем начальное приближение для оптимизации
        params = []

        # Добавляем параметры камер
        for cam_idx in range(self.n_cameras):
            # Если оптимизируем внутренние параметры
            if self.optimize_intrinsics:
                params.extend(self.camera_intrinsics[cam_idx])

            # Если оптимизируем внешние параметры
            if self.optimize_extrinsics:
                params.extend(self.camera_extrinsics[cam_idx])

        # Добавляем параметры 3D-точек
        params.extend(points_3d.reshape(-1))

        # Преобразуем в numpy массив
        x0 = np.array(params)

        # Функция потерь для оптимизации
        def objective_function(x):
            params_idx = 0

            # Восстанавливаем параметры камер
            camera_matrices_new = []
            for cam_idx in range(self.n_cameras):
                intrinsics_new = self.camera_intrinsics[cam_idx]
                extrinsics_new = self.camera_extrinsics[cam_idx]

                # Если оптимизируем внутренние параметры
                if self.optimize_intrinsics:
                    intrinsics_new = x[params_idx:params_idx + 4]
                    params_idx += 4

                # Если оптимизируем внешние параметры
                if self.optimize_extrinsics:
                    extrinsics_new = x[params_idx:params_idx + 6]
                    params_idx += 6

                # Восстанавливаем матрицу проекции
                camera_matrices_new.append(self._reconstruct_camera_matrix(intrinsics_new, extrinsics_new))

            camera_matrices_new = np.array(camera_matrices_new)
            # Восстанавливаем 3D-точки
            points_3d_new = x[params_idx:params_idx + n_points * 3].reshape(n_points, 3)

            # Проецируем 3D-точки на 2D
            projected_points = project_3d_to_2d(points_3d_new, camera_matrices_new)

            # Вычисляем ошибки репроекции
            errors = points_2d - projected_points

            # Применяем маску видимости и веса
            weighted_errors = errors * weights[:, :, np.newaxis] * visibility[:, :, np.newaxis]

            # Возвращаем плоский массив ошибок
            return weighted_errors.reshape(-1)

        # Выполняем оптимизацию
        result = optimize.least_squares(
            objective_function,
            x0,
            method='trf',  # Trust Region Reflective
            loss='huber',  # Устойчивая к выбросам функция потерь
            ftol=self.convergence_tolerance,
            max_nfev=self.max_iterations,
            verbose=2 if self.verbose else 0
        )

        # Восстанавливаем оптимизированные параметры
        params_idx = 0
        camera_matrices_optimized = []

        for cam_idx in range(self.n_cameras):
            intrinsics_new = self.camera_intrinsics[cam_idx]
            extrinsics_new = self.camera_extrinsics[cam_idx]

            # Если оптимизируем внутренние параметры
            if self.optimize_intrinsics:
                intrinsics_new = result.x[params_idx:params_idx + 4]
                params_idx += 4

            # Если оптимизируем внешние параметры
            if self.optimize_extrinsics:
                extrinsics_new = result.x[params_idx:params_idx + 6]
                params_idx += 6

            # Обновляем параметры камер
            self.camera_intrinsics[cam_idx] = intrinsics_new
            self.camera_extrinsics[cam_idx] = extrinsics_new

            # Восстанавливаем матрицу проекции
            camera_matrices_optimized.append(self._reconstruct_camera_matrix(intrinsics_new, extrinsics_new))

        camera_matrices_optimized = np.array(camera_matrices_optimized)

        # Восстанавливаем оптимизированные 3D-точки
        points_3d_optimized = result.x[params_idx:params_idx + n_points * 3].reshape(n_points, 3)

        # Вычисляем итоговую ошибку репроекции
        initial_error = np.mean(
            np.linalg.norm(points_2d - project_3d_to_2d(points_3d, self.camera_matrices), axis=2)[visibility]
        )
        final_error = np.mean(
            np.linalg.norm(points_2d - project_3d_to_2d(points_3d_optimized, camera_matrices_optimized), axis=2)[
                visibility]
        )

        logger.info(f"Bundle adjustment завершен. Начальная ошибка: {initial_error:.4f}, "
                    f"конечная ошибка: {final_error:.4f}")

        return points_3d_optimized, camera_matrices_optimized


def refine_triangulation(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        max_iterations: int = 10,
        convergence_tolerance: float = 1e-6
) -> np.ndarray:
    """
    Уточняет триангуляцию 3D-точек.

    Более простая альтернатива полной оптимизации, уточняющая каждую 3D-точку
    независимо путем минимизации ошибки репроекции.

    Args:
        points_3d: Исходные 3D-точки размера (n_points, 3)
        points_2d: 2D-точки размера (n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции камер размера (n_cameras, 3, 4)
        visibility: Маска видимости точек размера (n_cameras, n_points)
        max_iterations: Максимальное число итераций
        convergence_tolerance: Порог сходимости

    Returns:
        np.ndarray: Уточненные 3D-точки размера (n_points, 3)
    """
    n_cameras, n_points, _ = points_2d.shape

    # Если маска видимости не предоставлена, создаем её из points_2d
    if visibility is None:
        visibility = ~np.isnan(points_2d[:, :, 0])

    # Копируем исходные точки
    refined_points_3d = np.copy(points_3d)

    # Для каждой точки
    for point_idx in range(n_points):
        # Получаем список камер, на которых видна эта точка
        visible_cameras = np.where(visibility[:, point_idx])[0]

        # Если точка видна менее чем на двух камерах, пропускаем её
        if len(visible_cameras) < 2:
            continue

        # Получаем 2D-координаты для этой точки
        point_2d = points_2d[visible_cameras, point_idx]

        # Получаем матрицы проекции для этих камер
        point_projection_matrices = projection_matrices[visible_cameras]

        # Функция для минимизации ошибки репроекции
        def reprojection_error(x):
            # x - это 3D-координаты точки
            x = x.reshape(1, 3)

            # Проецируем на все камеры
            projected = project_3d_to_2d(x, point_projection_matrices)

            # Вычисляем ошибку
            error = point_2d - projected

            # Возвращаем плоский массив ошибок
            return error.reshape(-1)

        # Начальное приближение - исходная 3D-точка
        x0 = points_3d[point_idx]

        # Минимизируем ошибку репроекции
        result = optimize.least_squares(
            reprojection_error,
            x0,
            method='trf',
            ftol=convergence_tolerance,
            max_nfev=max_iterations,
            loss='huber'
        )

        # Сохраняем уточненную точку
        refined_points_3d[point_idx] = result.x

    return refined_points_3d


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    n_cameras = 3
    n_points = 10

    # Случайные 3D-точки
    points_3d = np.random.rand(n_points, 3) * 10

    # Случайные матрицы проекции
    projection_matrices = np.random.rand(n_cameras, 3, 4)

    # Проецируем 3D-точки на 2D
    points_2d = project_3d_to_2d(points_3d, projection_matrices)

    # Добавляем немного шума
    points_2d += np.random.normal(0, 0.1, points_2d.shape)

    # Создаем оптимизатор
    optimizer = PointCloudOptimizer(
        projection_matrices=projection_matrices,
        optimization_method='gauss_newton',
        verbose=True
    )

    # Оптимизируем 3D-точки
    optimized_points_3d = optimizer.optimize_points(points_3d, points_2d)

    # Или используем более простое уточнение
    refined_points_3d = refine_triangulation(
        points_3d=points_3d,
        points_2d=points_2d,
        projection_matrices=projection_matrices
    )

    # Вычисляем ошибку репроекции до и после оптимизации
    initial_error = np.mean(np.linalg.norm(
        points_2d - project_3d_to_2d(points_3d, projection_matrices),
        axis=2
    ))

    optimized_error = np.mean(np.linalg.norm(
        points_2d - project_3d_to_2d(optimized_points_3d, projection_matrices),
        axis=2
    ))

    refined_error = np.mean(np.linalg.norm(
        points_2d - project_3d_to_2d(refined_points_3d, projection_matrices),
        axis=2
    ))

    print(f"Начальная ошибка: {initial_error:.6f}")
    print(f"Ошибка после оптимизации: {optimized_error:.6f}")
    print(f"Ошибка после уточнения: {refined_error:.6f}")
