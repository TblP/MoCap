"""
Модуль для работы с ошибками репроекции.

Содержит функции и классы для анализа, визуализации и обработки ошибок репроекции
при 3D-реконструкции. Ошибка репроекции - это расстояние между наблюдаемыми 2D-координатами
точки и проекцией соответствующей реконструированной 3D-точки на изображение.
"""

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.utils.geometry import project_3d_to_2d

logger = logging.getLogger(__name__)


class ReprojectionErrorAnalyzer:
    """
    Класс для анализа и визуализации ошибок репроекции.

    Attributes:
        camera_matrices (np.ndarray): Матрицы проекции для камер
        camera_names (List[str]): Имена камер
        image_sizes (List[Tuple[int, int]]): Размеры изображений камер (ширина, высота)
    """

    def __init__(
            self,
            camera_matrices: np.ndarray,
            camera_names: Optional[List[str]] = None,
            image_sizes: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Инициализирует объект анализа ошибок репроекции.

        Args:
            camera_matrices: Матрицы проекции для камер, размер (n_cameras, 3, 4)
            camera_names: Имена камер (если None, используются индексы)
            image_sizes: Размеры изображений камер (если None, не используются)
        """
        self.camera_matrices = np.array(camera_matrices)
        n_cameras = len(camera_matrices)

        # Проверка размерности матриц
        for i, mat in enumerate(camera_matrices):
            if mat.shape != (3, 4):
                raise ValueError(f"Матрица камеры {i} имеет неверный размер: {mat.shape}, ожидается (3, 4)")

        # Установка имен камер
        if camera_names is None:
            self.camera_names = [f"camera_{i}" for i in range(n_cameras)]
        else:
            if len(camera_names) != n_cameras:
                raise ValueError("Количество имен камер не соответствует количеству матриц камер")
            self.camera_names = camera_names

        # Установка размеров изображений
        if image_sizes is None:
            self.image_sizes = [(1920, 1080) for _ in range(n_cameras)]  # Стандартный размер по умолчанию
        else:
            if len(image_sizes) != n_cameras:
                raise ValueError("Количество размеров изображений не соответствует количеству камер")
            self.image_sizes = image_sizes

        logger.info(f"Инициализирован анализатор ошибок репроекции для {n_cameras} камер")

    def calculate_reprojection_errors(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            visibility: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет ошибки репроекции для всех камер и точек.

        Args:
            points_3d: 3D-точки размера (n_points, 3) или (n_frames, n_points, 3)
            points_2d: 2D-точки размера (n_cameras, n_points, 2) или (n_frames, n_cameras, n_points, 2)
            visibility: Маска видимости точек размера (n_cameras, n_points) или (n_frames, n_cameras, n_points)
                        Если None, точки считаются видимыми, если не NaN в points_2d

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Ошибки репроекции (евклидово расстояние) размера (n_cameras, n_points) или (n_frames, n_cameras, n_points)
                - Векторы ошибок репроекции размера (n_cameras, n_points, 2) или (n_frames, n_cameras, n_points, 2)
        """
        is_sequence = len(points_3d.shape) == 3

        if is_sequence:
            # Обработка последовательности кадров
            n_frames, n_points, _ = points_3d.shape
            n_cameras = len(self.camera_matrices)

            # Инициализация массивов для результатов
            error_distances = np.full((n_frames, n_cameras, n_points), np.nan)
            error_vectors = np.full((n_frames, n_cameras, n_points, 2), np.nan)

            # Если маска видимости не предоставлена, создаем ее из points_2d
            if visibility is None:
                visibility = ~np.isnan(points_2d[:, :, :, 0])

            # Для каждого кадра
            for frame_idx in range(n_frames):
                # Проецируем 3D-точки на 2D
                projected_points = project_3d_to_2d(points_3d[frame_idx], self.camera_matrices)

                # Вычисляем ошибки репроекции
                errors = points_2d[frame_idx] - projected_points

                # Применяем маску видимости
                frame_visibility = visibility[frame_idx]
                errors[~frame_visibility] = np.nan

                # Вычисляем евклидово расстояние
                error_distances[frame_idx] = np.linalg.norm(errors, axis=2)
                error_vectors[frame_idx] = errors
        else:
            # Обработка одного кадра
            n_points = points_3d.shape[0]
            n_cameras = len(self.camera_matrices)

            # Инициализация массивов для результатов
            error_distances = np.full((n_cameras, n_points), np.nan)
            error_vectors = np.full((n_cameras, n_points, 2), np.nan)

            # Если маска видимости не предоставлена, создаем ее из points_2d
            if visibility is None:
                visibility = ~np.isnan(points_2d[:, :, 0])

            # Проецируем 3D-точки на 2D
            projected_points = project_3d_to_2d(points_3d, self.camera_matrices)

            # Вычисляем ошибки репроекции
            errors = points_2d - projected_points

            # Применяем маску видимости
            errors[~visibility] = np.nan

            # Вычисляем евклидово расстояние
            error_distances = np.linalg.norm(errors, axis=2)
            error_vectors = errors

        return error_distances, error_vectors

    def analyze_errors(
            self,
            error_distances: np.ndarray
    ) -> Dict[str, Any]:
        """
        Анализирует ошибки репроекции и возвращает статистику.

        Args:
            error_distances: Ошибки репроекции размера (n_cameras, n_points) или (n_frames, n_cameras, n_points)

        Returns:
            Dict[str, Any]: Словарь со статистикой ошибок репроекции
        """
        is_sequence = len(error_distances.shape) == 3

        if is_sequence:
            n_frames, n_cameras, n_points = error_distances.shape

            # Статистика по всем данным
            all_errors = error_distances.reshape(-1)
            all_errors = all_errors[~np.isnan(all_errors)]

            # Статистика по камерам
            camera_stats = []
            for cam_idx in range(n_cameras):
                cam_errors = error_distances[:, cam_idx, :].reshape(-1)
                cam_errors = cam_errors[~np.isnan(cam_errors)]

                if len(cam_errors) > 0:
                    camera_stats.append({
                        'camera': self.camera_names[cam_idx],
                        'mean': float(np.mean(cam_errors)),
                        'median': float(np.median(cam_errors)),
                        'std': float(np.std(cam_errors)),
                        'min': float(np.min(cam_errors)),
                        'max': float(np.max(cam_errors)),
                        'percentile_90': float(np.percentile(cam_errors, 90)),
                        'percentile_95': float(np.percentile(cam_errors, 95)),
                        'percentile_99': float(np.percentile(cam_errors, 99)),
                        'num_points': len(cam_errors)
                    })
                else:
                    camera_stats.append({
                        'camera': self.camera_names[cam_idx],
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'percentile_90': np.nan,
                        'percentile_95': np.nan,
                        'percentile_99': np.nan,
                        'num_points': 0
                    })

            # Статистика по кадрам
            frame_stats = []
            for frame_idx in range(n_frames):
                frame_errors = error_distances[frame_idx].reshape(-1)
                frame_errors = frame_errors[~np.isnan(frame_errors)]

                if len(frame_errors) > 0:
                    frame_stats.append({
                        'frame': frame_idx,
                        'mean': float(np.mean(frame_errors)),
                        'median': float(np.median(frame_errors)),
                        'std': float(np.std(frame_errors)),
                        'min': float(np.min(frame_errors)),
                        'max': float(np.max(frame_errors)),
                        'num_points': len(frame_errors)
                    })

            # Статистика по точкам (маркерам)
            point_stats = []
            for point_idx in range(n_points):
                point_errors = error_distances[:, :, point_idx].reshape(-1)
                point_errors = point_errors[~np.isnan(point_errors)]

                if len(point_errors) > 0:
                    point_stats.append({
                        'point': point_idx,
                        'mean': float(np.mean(point_errors)),
                        'median': float(np.median(point_errors)),
                        'std': float(np.std(point_errors)),
                        'min': float(np.min(point_errors)),
                        'max': float(np.max(point_errors)),
                        'num_observations': len(point_errors)
                    })
                else:
                    point_stats.append({
                        'point': point_idx,
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'num_observations': 0
                    })
        else:
            n_cameras, n_points = error_distances.shape

            # Статистика по всем данным
            all_errors = error_distances.reshape(-1)
            all_errors = all_errors[~np.isnan(all_errors)]

            # Статистика по камерам
            camera_stats = []
            for cam_idx in range(n_cameras):
                cam_errors = error_distances[cam_idx, :].reshape(-1)
                cam_errors = cam_errors[~np.isnan(cam_errors)]

                if len(cam_errors) > 0:
                    camera_stats.append({
                        'camera': self.camera_names[cam_idx],
                        'mean': float(np.mean(cam_errors)),
                        'median': float(np.median(cam_errors)),
                        'std': float(np.std(cam_errors)),
                        'min': float(np.min(cam_errors)),
                        'max': float(np.max(cam_errors)),
                        'percentile_90': float(np.percentile(cam_errors, 90)),
                        'percentile_95': float(np.percentile(cam_errors, 95)),
                        'percentile_99': float(np.percentile(cam_errors, 99)),
                        'num_points': len(cam_errors)
                    })
                else:
                    camera_stats.append({
                        'camera': self.camera_names[cam_idx],
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'percentile_90': np.nan,
                        'percentile_95': np.nan,
                        'percentile_99': np.nan,
                        'num_points': 0
                    })

            # Статистика по точкам (маркерам)
            point_stats = []
            for point_idx in range(n_points):
                point_errors = error_distances[:, point_idx].reshape(-1)
                point_errors = point_errors[~np.isnan(point_errors)]

                if len(point_errors) > 0:
                    point_stats.append({
                        'point': point_idx,
                        'mean': float(np.mean(point_errors)),
                        'median': float(np.median(point_errors)),
                        'std': float(np.std(point_errors)),
                        'min': float(np.min(point_errors)),
                        'max': float(np.max(point_errors)),
                        'num_observations': len(point_errors)
                    })
                else:
                    point_stats.append({
                        'point': point_idx,
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'num_observations': 0
                    })

            frame_stats = []  # Нет данных о кадрах для одного изображения

        # Общая статистика
        if len(all_errors) > 0:
            overall_stats = {
                'mean': float(np.mean(all_errors)),
                'median': float(np.median(all_errors)),
                'std': float(np.std(all_errors)),
                'min': float(np.min(all_errors)),
                'max': float(np.max(all_errors)),
                'percentile_90': float(np.percentile(all_errors, 90)),
                'percentile_95': float(np.percentile(all_errors, 95)),
                'percentile_99': float(np.percentile(all_errors, 99)),
                'total_points': len(all_errors)
            }
        else:
            overall_stats = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'percentile_90': np.nan,
                'percentile_95': np.nan,
                'percentile_99': np.nan,
                'total_points': 0
            }

        return {
            'overall': overall_stats,
            'cameras': camera_stats,
            'points': point_stats,
            'frames': frame_stats if is_sequence else [],
            'is_sequence': is_sequence
        }

    def visualize_error_heatmap(
            self,
            error_distances: np.ndarray,
            frame_idx: Optional[int] = None,
            output_path: Optional[Union[str, Path]] = None,
            figsize: Tuple[int, int] = (12, 8),
            vmax: Optional[float] = None
    ) -> Any:
        """
        Визуализирует тепловую карту ошибок репроекции.

        Args:
            error_distances: Ошибки репроекции размера (n_cameras, n_points) или (n_frames, n_cameras, n_points)
            frame_idx: Индекс кадра для визуализации (только для последовательности)
            output_path: Путь для сохранения визуализации (если None, не сохраняет)
            figsize: Размер фигуры
            vmax: Максимальное значение для цветовой шкалы (если None, использует максимум данных)

        Returns:
            matplotlib.figure.Figure: Объект фигуры
        """
        is_sequence = len(error_distances.shape) == 3

        if is_sequence:
            if frame_idx is None:
                # Усредняем по всем кадрам
                error_distances_to_plot = np.nanmean(error_distances, axis=0)
                title = "Средняя ошибка репроекции по всем кадрам"
            else:
                error_distances_to_plot = error_distances[frame_idx]
                title = f"Ошибка репроекции для кадра {frame_idx}"
        else:
            error_distances_to_plot = error_distances
            title = "Ошибка репроекции"

        n_cameras, n_points = error_distances_to_plot.shape

        fig, ax = plt.subplots(figsize=figsize)

        # Устанавливаем максимальное значение для цветовой шкалы
        if vmax is None:
            vmax = np.nanmax(error_distances_to_plot)
            if np.isnan(vmax) or vmax == 0:
                vmax = 1.0

        # Создаем тепловую карту
        im = ax.imshow(error_distances_to_plot, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)

        # Настраиваем оси
        if n_cameras <= 20:  # Если камер немного, показываем все метки
            ax.set_yticks(np.arange(n_cameras))
            ax.set_yticklabels(self.camera_names)
        else:  # Иначе показываем только часть
            step = max(1, n_cameras // 20)
            ax.set_yticks(np.arange(0, n_cameras, step))
            ax.set_yticklabels([self.camera_names[i] for i in range(0, n_cameras, step)])

        if n_points <= 50:  # Если точек немного, показываем все метки
            ax.set_xticks(np.arange(n_points))
            ax.set_xticklabels(np.arange(n_points))
        else:  # Иначе показываем только часть
            step = max(1, n_points // 20)
            ax.set_xticks(np.arange(0, n_points, step))
            ax.set_xticklabels(np.arange(0, n_points, step))

        # Устанавливаем заголовок и подписи осей
        ax.set_title(title)
        ax.set_xlabel('Точка')
        ax.set_ylabel('Камера')

        # Добавляем цветовую шкалу
        cbar = plt.colorbar(im)
        cbar.set_label('Ошибка репроекции (пиксели)')

        plt.tight_layout()

        # Сохраняем, если указан путь
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Тепловая карта ошибок репроекции сохранена в {output_path}")

        return fig

    def visualize_errors_on_image(
            self,
            image: np.ndarray,
            points_2d: np.ndarray,
            error_vectors: np.ndarray,
            camera_idx: int,
            point_labels: Optional[List[str]] = None,
            point_marker_size: int = 5,
            error_scale: float = 1.0,
            max_error_to_show: float = 10.0,
            min_error_to_show: float = 0.0,
            output_path: Optional[Union[str, Path]] = None,
            show_image: bool = True
    ) -> np.ndarray:
        """
        Визуализирует ошибки репроекции на изображении.

        Args:
            image: Изображение для визуализации
            points_2d: 2D-точки размера (n_cameras, n_points, 2) или (n_points, 2) для одной камеры
            error_vectors: Векторы ошибок размера (n_cameras, n_points, 2) или (n_points, 2) для одной камеры
            camera_idx: Индекс камеры для визуализации (игнорируется если points_2d имеет размер (n_points, 2))
            point_labels: Метки для точек (если None, используются индексы)
            point_marker_size: Размер маркера точки
            error_scale: Масштаб для визуализации векторов ошибок
            max_error_to_show: Максимальная ошибка для отображения
            min_error_to_show: Минимальная ошибка для отображения
            output_path: Путь для сохранения визуализации (если None, не сохраняет)
            show_image: Показывать ли исходное изображение (если False, создается пустое изображение)

        Returns:
            np.ndarray: Изображение с визуализацией
        """
        # Проверяем форму points_2d
        is_multi_camera = len(points_2d.shape) == 3
        if is_multi_camera:
            if camera_idx >= points_2d.shape[0]:
                raise ValueError(f"Индекс камеры {camera_idx} выходит за пределы {points_2d.shape[0]} камер")
            points = points_2d[camera_idx]
            errors = error_vectors[camera_idx]
        else:
            points = points_2d
            errors = error_vectors

        n_points = points.shape[0]

        # Проверка размеров изображения
        if not show_image:
            # Если не показываем исходное изображение, создаем пустое
            image_height, image_width = self.image_sizes[camera_idx] if is_multi_camera else (1080, 1920)
            image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        else:
            if image.ndim == 2:
                # Конвертируем черно-белое изображение в цветное
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if image.dtype != np.uint8:
                # Нормализуем изображение
                image = (image * 255).astype(np.uint8)

        image_height, image_width = image.shape[:2]

        # Создаем метки для точек
        if point_labels is None:
            point_labels = [str(i) for i in range(n_points)]

        # Вычисляем нормы ошибок для цветового кодирования
        error_norms = np.linalg.norm(errors, axis=1)

        # Копируем изображение для визуализации
        vis_image = image.copy()

        # Рисуем каждую точку и вектор ошибки
        for i in range(n_points):
            if np.any(np.isnan(points[i])) or np.any(np.isnan(errors[i])):
                continue

            error_norm = error_norms[i]

            # Пропускаем точки с ошибкой вне диапазона
            if error_norm < min_error_to_show or error_norm > max_error_to_show:
                continue

            # Координаты точки и конца вектора ошибки
            x, y = points[i].astype(int)
            dx, dy = errors[i] * error_scale
            end_x, end_y = int(x + dx), int(y + dy)

            # Ограничиваем координаты границами изображения
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            end_x = max(0, min(end_x, image_width - 1))
            end_y = max(0, min(end_y, image_height - 1))

            # Цвет зависит от величины ошибки
            # От зеленого (маленькая ошибка) до красного (большая ошибка)
            if error_norm <= max_error_to_show / 3:
                color = (0, 255, 0)  # Зеленый
            elif error_norm <= 2 * max_error_to_show / 3:
                color = (0, 255, 255)  # Желтый
            else:
                color = (0, 0, 255)  # Красный

            # Рисуем точку
            cv2.circle(vis_image, (x, y), point_marker_size, color, -1)

            # Рисуем вектор ошибки
            cv2.line(vis_image, (x, y), (end_x, end_y), color, 2)

            # Добавляем метку точки
            cv2.putText(
                vis_image,
                point_labels[i],
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Добавляем информацию о камере и масштабе
        camera_name = self.camera_names[camera_idx] if is_multi_camera else "Camera"
        cv2.putText(
            vis_image,
            f"{camera_name} (scale: {error_scale}x)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Сохраняем, если указан путь
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"Визуализация ошибок репроекции сохранена в {output_path}")

        return vis_image

    def create_reprojection_error_report(
            self,
            error_stats: Dict[str, Any],
            output_path: Union[str, Path],
            include_heatmap: bool = True,
            error_distances: Optional[np.ndarray] = None
    ) -> None:
        """
        Создает отчет об ошибках репроекции.

        Args:
            error_stats: Статистика ошибок репроекции (результат метода analyze_errors)
            output_path: Путь для сохранения отчета
            include_heatmap: Включать ли тепловую карту ошибок
            error_distances: Ошибки репроекции для тепловой карты (необходимо, если include_heatmap=True)

        Returns:
            None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Отчет об ошибках репроекции\n\n")

            # Общая статистика
            f.write("## Общая статистика\n\n")
            f.write(f"- Средняя ошибка: {error_stats['overall']['mean']:.2f} пикселей\n")
            f.write(f"- Медианная ошибка: {error_stats['overall']['median']:.2f} пикселей\n")
            f.write(f"- Стандартное отклонение: {error_stats['overall']['std']:.2f} пикселей\n")
            f.write(f"- Минимальная ошибка: {error_stats['overall']['min']:.2f} пикселей\n")
            f.write(f"- Максимальная ошибка: {error_stats['overall']['max']:.2f} пикселей\n")
            f.write(f"- 90-й процентиль: {error_stats['overall']['percentile_90']:.2f} пикселей\n")
            f.write(f"- 95-й процентиль: {error_stats['overall']['percentile_95']:.2f} пикселей\n")
            f.write(f"- 99-й процентиль: {error_stats['overall']['percentile_99']:.2f} пикселей\n")
            f.write(f"- Общее количество точек: {error_stats['overall']['total_points']}\n\n")

            # Статистика по камерам
            f.write("## Статистика по камерам\n\n")
            f.write(
                "| Камера | Средняя ошибка | Медианная ошибка | Стд. отклонение | Мин. | Макс. | 95% | Кол-во точек |\n")
            f.write(
                "|--------|---------------|-----------------|-----------------|------|------|-----|-------------|\n")
            for cam_stat in error_stats['cameras']:
                f.write(
                    f"| {cam_stat['camera']} | {cam_stat['mean']:.2f} | {cam_stat['median']:.2f} | {cam_stat['std']:.2f} | {cam_stat['min']:.2f} | {cam_stat['max']:.2f} | {cam_stat['percentile_95']:.2f} | {cam_stat['num_points']} |\n")
            f.write("\n")

            # Статистика по точкам (топ 10 с наибольшей ошибкой)
            f.write("## Статистика по точкам (Топ 10 с наибольшей ошибкой)\n\n")
            f.write(
                "| Точка | Средняя ошибка | Медианная ошибка | Стд. отклонение | Мин. | Макс. | Кол-во наблюдений |\n")
            f.write("|-------|---------------|-----------------|-----------------|------|------|-------------------|\n")

            # Сортируем точки по средней ошибке (по убыванию)
            sorted_points = sorted(error_stats['points'], key=lambda x: x['mean'] if not np.isnan(x['mean']) else -1,
                                   reverse=True)
            for point_stat in sorted_points[:10]:
                f.write(
                    f"| {point_stat['point']} | {point_stat['mean']:.2f} | {point_stat['median']:.2f} | {point_stat['std']:.2f} | {point_stat['min']:.2f} | {point_stat['max']:.2f} | {point_stat['num_observations']} |\n")
            f.write("\n")

            # Статистика по кадрам (если это последовательность)
            if error_stats['is_sequence'] and error_stats['frames']:
                f.write("## Статистика по кадрам\n\n")

                # Извлекаем средние ошибки для каждого кадра
                frame_means = [frame_stat['mean'] for frame_stat in error_stats['frames']]

                # Находим кадры с наибольшей ошибкой
                worst_frames_indices = np.argsort(frame_means)[-10:][::-1]
                worst_frames = [error_stats['frames'][i] for i in worst_frames_indices]

                f.write("### Топ 10 кадров с наибольшей ошибкой\n\n")
                f.write(
                    "| Кадр | Средняя ошибка | Медианная ошибка | Стд. отклонение | Мин. | Макс. | Кол-во точек |\n")
                f.write("|------|---------------|-----------------|-----------------|------|------|-------------|\n")
                for frame_stat in worst_frames:
                    f.write(
                        f"| {frame_stat['frame']} | {frame_stat['mean']:.2f} | {frame_stat['median']:.2f} | {frame_stat['std']:.2f} | {frame_stat['min']:.2f} | {frame_stat['max']:.2f} | {frame_stat['num_points']} |\n")
                f.write("\n")

            # Добавляем тепловую карту, если требуется
            if include_heatmap and error_distances is not None:
                heatmap_path = output_path.with_name(f"{output_path.stem}_heatmap.png")
                f.write("## Тепловая карта ошибок репроекции\n\n")

                # Создаем и сохраняем тепловую карту
                self.visualize_error_heatmap(error_distances, output_path=heatmap_path)

                # Добавляем ссылку на изображение в отчет
                f.write(f"![Тепловая карта ошибок репроекции]({heatmap_path.name})\n\n")

            f.write("## Заключение\n\n")

            # Даем оценку качества реконструкции на основе ошибок
            mean_error = error_stats['overall']['mean']
            if mean_error < 1.0:
                f.write("Качество реконструкции: **Отличное**. Средняя ошибка репроекции меньше 1 пикселя.\n")
            elif mean_error < 2.0:
                f.write("Качество реконструкции: **Хорошее**. Средняя ошибка репроекции меньше 2 пикселей.\n")
            elif mean_error < 5.0:
                f.write(
                    "Качество реконструкции: **Удовлетворительное**. Средняя ошибка репроекции меньше 5 пикселей.\n")
            else:
                f.write(
                    "Качество реконструкции: **Требует улучшения**. Средняя ошибка репроекции превышает 5 пикселей.\n")

                # Рекомендации по улучшению
                f.write("\nРекомендации по улучшению качества реконструкции:\n\n")
                f.write("1. Проверьте калибровку камер, особенно для камер с высокой ошибкой репроекции.\n")
                f.write("2. Убедитесь, что точки корректно идентифицированы на всех изображениях.\n")
                f.write("3. Рассмотрите возможность фильтрации выбросов при триангуляции.\n")
                f.write("4. При необходимости примените методы оптимизации для уточнения 3D-позиций.\n")

        logger.info(f"Отчет об ошибках репроекции сохранен в {output_path}")


def identify_outliers(
        error_distances: np.ndarray,
        threshold_method: str = 'percentile',
        threshold_value: float = 95.0,
        return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Идентифицирует выбросы в ошибках репроекции.

    Args:
        error_distances: Ошибки репроекции размера (n_cameras, n_points) или (n_frames, n_cameras, n_points)
        threshold_method: Метод определения порога ('percentile', 'std', 'absolute')
        threshold_value: Значение порога (процентиль, количество стандартных отклонений или абсолютное значение)
        return_mask: Возвращать ли маску выбросов

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - Массив индексов выбросов формы (n_outliers, 2) или (n_outliers, 3) для последовательности
            - Если return_mask=True, также возвращает булеву маску выбросов
    """
    is_sequence = len(error_distances.shape) == 3

    # Создаем копию для безопасности
    errors = np.copy(error_distances)

    # Определяем порог в зависимости от метода
    if threshold_method == 'percentile':
        all_errors = errors.reshape(-1)
        all_errors = all_errors[~np.isnan(all_errors)]
        threshold = np.percentile(all_errors, threshold_value)
    elif threshold_method == 'std':
        all_errors = errors.reshape(-1)
        all_errors = all_errors[~np.isnan(all_errors)]
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        threshold = mean_error + threshold_value * std_error
    elif threshold_method == 'absolute':
        threshold = threshold_value
    else:
        raise ValueError(f"Неизвестный метод определения порога: {threshold_method}")

    # Идентифицируем выбросы
    if is_sequence:
        n_frames, n_cameras, n_points = errors.shape
        outlier_mask = errors > threshold

        # Получаем индексы выбросов
        frame_idx, cam_idx, point_idx = np.where(outlier_mask)
        outliers = np.column_stack((frame_idx, cam_idx, point_idx))
    else:
        n_cameras, n_points = errors.shape
        outlier_mask = errors > threshold

        # Получаем индексы выбросов
        cam_idx, point_idx = np.where(outlier_mask)
        outliers = np.column_stack((cam_idx, point_idx))

    if return_mask:
        return outliers, outlier_mask
    else:
        return outliers


def filter_outliers(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        threshold_method: str = 'percentile',
        threshold_value: float = 95.0,
        return_errors: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Фильтрует выбросы в реконструированных 3D-точках на основе ошибок репроекции.

    Args:
        points_3d: 3D-точки размера (n_points, 3) или (n_frames, n_points, 3)
        points_2d: 2D-точки размера (n_cameras, n_points, 2) или (n_frames, n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции размера (n_cameras, 3, 4)
        visibility: Маска видимости точек (если None, определяется из points_2d)
        threshold_method: Метод определения порога ('percentile', 'std', 'absolute')
        threshold_value: Значение порога
        return_errors: Возвращать ли ошибки репроекции

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - Отфильтрованные 3D-точки
            - Если return_errors=True, также возвращает ошибки репроекции
    """
    is_sequence = len(points_3d.shape) == 3

    # Проецируем 3D-точки на 2D
    if is_sequence:
        n_frames, n_points, _ = points_3d.shape
        n_cameras = projection_matrices.shape[0]

        # Если маска видимости не предоставлена, создаем ее
        if visibility is None:
            visibility = ~np.isnan(points_2d[:, :, :, 0])

        # Инициализируем массивы для результатов
        filtered_points_3d = np.copy(points_3d)
        error_distances = np.full((n_frames, n_cameras, n_points), np.nan)

        # Для каждого кадра
        for frame_idx in range(n_frames):
            # Проецируем 3D-точки текущего кадра на 2D
            projected_points = project_3d_to_2d(points_3d[frame_idx], projection_matrices)

            # Вычисляем ошибки репроекции
            frame_errors = points_2d[frame_idx] - projected_points

            # Применяем маску видимости
            if visibility is not None:
                frame_visibility = visibility[frame_idx]
                frame_errors[~frame_visibility] = np.nan

            # Вычисляем евклидово расстояние
            error_distances[frame_idx] = np.linalg.norm(frame_errors, axis=2)

        # Идентифицируем выбросы
        _, outlier_mask = identify_outliers(
            error_distances,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            return_mask=True
        )

        # Заменяем выбросы на NaN
        for frame_idx in range(n_frames):
            for point_idx in range(n_points):
                # Если точка является выбросом хотя бы для одной камеры
                if np.any(outlier_mask[frame_idx, :, point_idx]):
                    filtered_points_3d[frame_idx, point_idx] = np.nan
    else:
        n_points = points_3d.shape[0]
        n_cameras = projection_matrices.shape[0]

        # Если маска видимости не предоставлена, создаем ее
        if visibility is None:
            visibility = ~np.isnan(points_2d[:, :, 0])

        # Инициализируем массивы для результатов
        filtered_points_3d = np.copy(points_3d)

        # Проецируем 3D-точки на 2D
        projected_points = project_3d_to_2d(points_3d, projection_matrices)

        # Вычисляем ошибки репроекции
        errors = points_2d - projected_points

        # Применяем маску видимости
        if visibility is not None:
            errors[~visibility] = np.nan

        # Вычисляем евклидово расстояние
        error_distances = np.linalg.norm(errors, axis=2)

        # Идентифицируем выбросы
        _, outlier_mask = identify_outliers(
            error_distances,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            return_mask=True
        )

        # Заменяем выбросы на NaN
        for point_idx in range(n_points):
            # Если точка является выбросом хотя бы для одной камеры
            if np.any(outlier_mask[:, point_idx]):
                filtered_points_3d[point_idx] = np.nan

    if return_errors:
        return filtered_points_3d, error_distances
    else:
        return filtered_points_3d


def refine_points_3d(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        projection_matrices: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        method: str = 'levenberg_marquardt',
        max_iterations: int = 100,
        tolerance: float = 1e-6
) -> np.ndarray:
    """
    Уточняет 3D-точки на основе минимизации ошибок репроекции.

    Args:
        points_3d: Исходные 3D-точки размера (n_points, 3) или (n_frames, n_points, 3)
        points_2d: 2D-точки размера (n_cameras, n_points, 2) или (n_frames, n_cameras, n_points, 2)
        projection_matrices: Матрицы проекции камер размера (n_cameras, 3, 4)
        visibility: Маска видимости точек (если None, определяется из points_2d)
        method: Метод оптимизации ('levenberg_marquardt', 'gauss_newton')
        max_iterations: Максимальное число итераций
        tolerance: Порог сходимости

    Returns:
        np.ndarray: Уточненные 3D-точки
    """
    is_sequence = len(points_3d.shape) == 3

    # Функция оптимизации для одной 3D-точки
    def optimize_point(point_3d, point_2d, projection_matrices, visibility=None):
        n_cameras = len(projection_matrices)

        # Если маска видимости не предоставлена, создаем ее
        if visibility is None:
            visibility = ~np.isnan(point_2d[:, 0])

        # Функция для вычисления ошибки репроекции
        def reprojection_error(point):
            point = point.reshape(3)
            error = np.zeros(n_cameras * 2)

            # Проецируем 3D-точку на каждую камеру
            for cam_idx in range(n_cameras):
                if not visibility[cam_idx]:
                    continue

                # Преобразуем 3D-точку в однородные координаты
                point_homogeneous = np.append(point, 1)

                # Проецируем точку
                projected = projection_matrices[cam_idx] @ point_homogeneous
                projected = projected[:2] / projected[2]

                # Вычисляем ошибку (разницу между наблюдаемой и проецированной точками)
                error[cam_idx * 2:cam_idx * 2 + 2] = point_2d[cam_idx] - projected

            return error

        # Якобиан ошибки репроекции
        def jacobian(point):
            point = point.reshape(3)
            J = np.zeros((n_cameras * 2, 3))

            for cam_idx in range(n_cameras):
                if not visibility[cam_idx]:
                    continue

                # Преобразуем 3D-точку в однородные координаты
                point_homogeneous = np.append(point, 1)

                # Получаем матрицу проекции
                P = projection_matrices[cam_idx]

                # Вычисляем проекцию
                projected = P @ point_homogeneous
                z = projected[2]

                # Вычисляем частные производные
                # d(x')/dX = (P00*z - P20*x') / z^2
                # d(x')/dY = (P01*z - P21*x') / z^2
                # d(x')/dZ = (P02*z - P22*x') / z^2
                # d(y')/dX = (P10*z - P20*y') / z^2
                # d(y')/dY = (P11*z - P21*y') / z^2
                # d(y')/dZ = (P12*z - P22*y') / z^2

                x_proj = projected[0] / z
                y_proj = projected[1] / z

                J[cam_idx * 2, 0] = (P[0, 0] * z - P[2, 0] * x_proj) / (z * z)
                J[cam_idx * 2, 1] = (P[0, 1] * z - P[2, 1] * x_proj) / (z * z)
                J[cam_idx * 2, 2] = (P[0, 2] * z - P[2, 2] * x_proj) / (z * z)

                J[cam_idx * 2 + 1, 0] = (P[1, 0] * z - P[2, 0] * y_proj) / (z * z)
                J[cam_idx * 2 + 1, 1] = (P[1, 1] * z - P[2, 1] * y_proj) / (z * z)
                J[cam_idx * 2 + 1, 2] = (P[1, 2] * z - P[2, 2] * y_proj) / (z * z)

            return J

        # Начальное приближение
        x = point_3d.copy()

        # Метод Левенберга-Марквардта
        if method == 'levenberg_marquardt':
            # Параметр затухания
            lambda_param = 0.01

            for iteration in range(max_iterations):
                # Вычисляем ошибку и якобиан
                error = reprojection_error(x)
                J = jacobian(x)

                # Проверяем сходимость
                if np.max(np.abs(error)) < tolerance:
                    break

                # Вычисляем нормальные уравнения с регуляризацией
                H = J.T @ J
                g = J.T @ error

                # Решаем линейную систему (H + lambda*I) * delta = g
                delta = np.linalg.solve(H + lambda_param * np.eye(3), g)

                # Обновляем параметры
                x_new = x - delta

                # Вычисляем новую ошибку
                error_new = reprojection_error(x_new)

                # Проверяем, улучшилось ли решение
                if np.sum(error_new ** 2) < np.sum(error ** 2):
                    x = x_new
                    lambda_param /= 10  # Уменьшаем параметр затухания
                else:
                    lambda_param *= 10  # Увеличиваем параметр затухания

        # Метод Гаусса-Ньютона
        elif method == 'gauss_newton':
            for iteration in range(max_iterations):
                # Вычисляем ошибку и якобиан
                error = reprojection_error(x)
                J = jacobian(x)

                # Проверяем сходимость
                if np.max(np.abs(error)) < tolerance:
                    break

                # Вычисляем нормальные уравнения
                H = J.T @ J
                g = J.T @ error

                # Решаем линейную систему H * delta = g
                try:
                    delta = np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    # Если матрица вырожденная, используем псевдообратную
                    delta = np.linalg.pinv(H) @ g

                # Обновляем параметры
                x = x - delta

        else:
            raise ValueError(f"Неизвестный метод оптимизации: {method}")

        return x

    # Оптимизируем каждую точку
    if is_sequence:
        n_frames, n_points, _ = points_3d.shape
        n_cameras = len(projection_matrices)

        # Инициализируем результат
        refined_points_3d = np.copy(points_3d)

        # Для каждого кадра и каждой точки
        for frame_idx in range(n_frames):
            for point_idx in range(n_points):
                # Пропускаем точки с NaN
                if np.any(np.isnan(points_3d[frame_idx, point_idx])):
                    continue

                # Получаем 2D-координаты для текущего кадра и точки
                point_2d = points_2d[frame_idx, :, point_idx]

                # Получаем маску видимости
                if visibility is not None:
                    point_visibility = visibility[frame_idx, :, point_idx]
                else:
                    point_visibility = ~np.isnan(point_2d[:, 0])

                # Пропускаем точки, видимые менее чем на двух камерах
                if np.sum(point_visibility) < 2:
                    continue

                # Оптимизируем 3D-координаты
                refined_point = optimize_point(
                    points_3d[frame_idx, point_idx],
                    point_2d,
                    projection_matrices,
                    point_visibility
                )

                refined_points_3d[frame_idx, point_idx] = refined_point
    else:
        n_points = points_3d.shape[0]
        n_cameras = len(projection_matrices)

        # Инициализируем результат
        refined_points_3d = np.copy(points_3d)

        # Для каждой точки
        for point_idx in range(n_points):
            # Пропускаем точки с NaN
            if np.any(np.isnan(points_3d[point_idx])):
                continue

            # Получаем 2D-координаты для текущей точки
            point_2d = points_2d[:, point_idx]

            # Получаем маску видимости
            if visibility is not None:
                point_visibility = visibility[:, point_idx]
            else:
                point_visibility = ~np.isnan(point_2d[:, 0])

            # Пропускаем точки, видимые менее чем на двух камерах
            if np.sum(point_visibility) < 2:
                continue

            # Оптимизируем 3D-координаты
            refined_point = optimize_point(
                points_3d[point_idx],
                point_2d,
                projection_matrices,
                point_visibility
            )

            refined_points_3d[point_idx] = refined_point

    return refined_points_3d


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

    # Добавляем шум к 2D-точкам
    noise = np.random.normal(0, 1, points_2d.shape)
    noisy_points_2d = points_2d + noise

    # Создаем анализатор ошибок репроекции
    analyzer = ReprojectionErrorAnalyzer(
        camera_matrices,
        camera_names=["camera_1", "camera_2"],
        image_sizes=[(640, 480), (640, 480)]
    )

    # Вычисляем ошибки репроекции
    error_distances, error_vectors = analyzer.calculate_reprojection_errors(
        true_points_3d, noisy_points_2d
    )

    # Анализируем ошибки
    error_stats = analyzer.analyze_errors(error_distances)

    # Выводим статистику
    logger.info(f"Средняя ошибка репроекции: {error_stats['overall']['mean']:.2f} пикселей")
    logger.info(f"Медианная ошибка репроекции: {error_stats['overall']['median']:.2f} пикселей")

    # Визуализируем ошибки в виде тепловой карты
    analyzer.visualize_error_heatmap(error_distances, output_path="reprojection_heatmap.png")

    # Создаем отчет
    analyzer.create_reprojection_error_report(
        error_stats,
        output_path="reprojection_error_report.md",
        include_heatmap=True,
        error_distances=error_distances
    )

    logger.info("Анализ ошибок репроекции завершен!")