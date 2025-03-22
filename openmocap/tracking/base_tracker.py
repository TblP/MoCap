"""
Базовый класс для трекеров точек на видео.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.utils.video_utils import get_video_properties

logger = logging.getLogger(__name__)


class BaseTracker(ABC):
    """
    Абстрактный базовый класс для трекеров точек на видео.

    Все трекеры должны наследоваться от этого класса и реализовывать
    абстрактные методы для обеспечения единого интерфейса.

    Attributes:
        name (str): Имя трекера
        num_tracked_points (int): Количество отслеживаемых точек
        landmark_names (List[str]): Список имен точек (ориентиров)
        config (Dict): Словарь с конфигурационными параметрами трекера
    """

    def __init__(self, name: str = "base_tracker", config: Optional[Dict] = None):
        """
        Инициализирует объект базового трекера.

        Args:
            name: Имя трекера
            config: Словарь с конфигурационными параметрами
        """
        self.name = name
        self.config = config or {}
        self.num_tracked_points = 0
        self.landmark_names = []

        logger.info(f"Инициализирован базовый трекер: {name}")

    @abstractmethod
    def track_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Отслеживает точки на одном кадре.

        Args:
            frame: Кадр изображения в формате NumPy array (BGR)

        Returns:
            np.ndarray: Массив координат точек shape (num_tracked_points, 2)
                        или (num_tracked_points, 3) с вероятностями
        """
        pass

    @abstractmethod
    def track_video(self, video_path: Union[str, Path], **kwargs) -> np.ndarray:
        """
        Отслеживает точки на всех кадрах видео.

        Args:
            video_path: Путь к видеофайлу
            **kwargs: Дополнительные параметры для конкретного трекера

        Returns:
            np.ndarray: Массив координат точек shape (num_frames, num_tracked_points, 2)
                        или (num_frames, num_tracked_points, 3) с вероятностями
        """
        pass

    def track_videos(
            self,
            video_paths: List[Union[str, Path]],
            **kwargs
    ) -> np.ndarray:
        """
        Отслеживает точки на нескольких видео.

        Args:
            video_paths: Список путей к видеофайлам
            **kwargs: Дополнительные параметры для конкретного трекера

        Returns:
            np.ndarray: Массив координат точек shape (num_cameras, num_frames, num_tracked_points, 2)
                        или (num_cameras, num_frames, num_tracked_points, 3) с вероятностями
        """
        results = []

        for i, video_path in enumerate(video_paths):
            logger.info(f"Отслеживание точек на видео {i + 1}/{len(video_paths)}: {video_path}")
            try:
                video_result = self.track_video(video_path, **kwargs)
                results.append(video_result)
            except Exception as e:
                logger.error(f"Ошибка при отслеживании точек на видео {video_path}: {e}")
                # В случае ошибки добавляем массив NaN соответствующего размера
                # Определяем размер массива из свойств видео
                video_props = get_video_properties(video_path)
                frame_count = video_props["frame_count"]
                result_shape = (frame_count, self.num_tracked_points, 2)
                results.append(np.full(result_shape, np.nan))

        # Убедимся, что все результаты имеют одинаковое количество кадров
        max_frames = max(res.shape[0] for res in results)

        # Дополняем результаты с меньшим количеством кадров значениями NaN
        normalized_results = []
        for result in results:
            if result.shape[0] < max_frames:
                pad_shape = list(result.shape)
                pad_shape[0] = max_frames - result.shape[0]
                padding = np.full(pad_shape, np.nan)
                result = np.vstack([result, padding])
            normalized_results.append(result)

        # Объединяем результаты в единый массив
        return np.stack(normalized_results)

    def get_landmark_names(self) -> List[str]:
        """
        Возвращает список имен точек (ориентиров).

        Returns:
            List[str]: Список имен точек
        """
        return self.landmark_names

    def get_config(self) -> Dict:
        """
        Возвращает конфигурацию трекера.

        Returns:
            Dict: Словарь с конфигурационными параметрами
        """
        return self.config

    def set_config(self, config: Dict) -> None:
        """
        Устанавливает конфигурацию трекера.

        Args:
            config: Словарь с конфигурационными параметрами
        """
        self.config = config

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели трекера.

        Returns:
            Dict[str, Any]: Словарь с информацией о модели трекера
        """
        pass