"""
Модуль для работы с шахматной доской для калибровки камеры.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class ChessBoard:
    """
    Класс для работы с шахматной доской для калибровки камеры.

    Attributes:
        width (int): Количество внутренних углов доски по ширине
        height (int): Количество внутренних углов доски по высоте
        square_size (float): Размер клетки шахматной доски в миллиметрах
    """

    def __init__(self, width: int = 7, height: int = 7, square_size: float = 50.0):
        """
        Инициализирует объект шахматной доски.

        Args:
            width: Количество внутренних углов доски по ширине (по умолчанию 9)
            height: Количество внутренних углов доски по высоте (по умолчанию 6)
            square_size: Размер клетки шахматной доски в миллиметрах (по умолчанию 25.0)
        """
        self.width = width
        self.height = height
        self.square_size = square_size

        # Создаем массив 3D-точек для шахматной доски
        self.object_points = np.zeros((height * width, 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

        # Настройки поиска углов шахматной доски
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        logger.info(f"Создана шахматная доска {width}x{height} с размером клетки {square_size} мм")

    def detect_corners(self, image: np.ndarray, visualize: bool = False) -> Tuple[bool, np.ndarray]:
        """
        Обнаруживает углы шахматной доски на изображении.

        Args:
            image: Входное изображение (в формате BGR или оттенков серого)
            visualize: Если True, создает визуализацию найденных углов

        Returns:
            Tuple[bool, np.ndarray]: (успех, координаты углов)
        """
        # Преобразуем изображение в оттенки серого, если оно цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Находим углы шахматной доски
        ret, corners = cv2.findChessboardCorners(
            gray,
            (self.width, self.height),
            None
        )

        if ret:
            # Уточняем положение углов
            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                self.criteria
            )

            logger.debug(f"Обнаружены углы шахматной доски {self.width}x{self.height}")

            if visualize:
                img_copy = np.copy(image)
                cv2.drawChessboardCorners(img_copy, (self.width, self.height), refined_corners, ret)
                return ret, refined_corners, img_copy
            else:
                return ret, refined_corners
        else:
            logger.debug("Не удалось обнаружить все углы шахматной доски")
            if visualize:
                return ret, None, image
            else:
                return ret, None

    def detect_corners_from_file(self, image_path: Union[str, Path], visualize: bool = False) -> Tuple[
        bool, np.ndarray]:
        """
        Обнаруживает углы шахматной доски на изображении из файла.

        Args:
            image_path: Путь к файлу изображения
            visualize: Если True, создает визуализацию найденных углов

        Returns:
            Tuple[bool, np.ndarray]: (успех, координаты углов)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Изображение не найдено: {image_path}")
            if visualize:
                return False, None, None
            else:
                return False, None

        try:
            image = cv2.imread(str(image_path))
            return self.detect_corners(image, visualize)
        except Exception as e:
            logger.error(f"Ошибка при чтении изображения {image_path}: {e}")
            if visualize:
                return False, None, None
            else:
                return False, None

    def detect_corners_from_video(
            self,
            video_path: Union[str, Path],
            max_frames: int = 100,
            frame_step: int = 1,
            min_corners: int = 10,
            visualize: bool = False
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Обнаруживает углы шахматной доски на кадрах видео.

        Args:
            video_path: Путь к видеофайлу
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами
            min_corners: Минимальное количество углов для успешного обнаружения
            visualize: Если True, возвращает визуализации найденных углов

        Returns:
            List[Tuple[int, np.ndarray]]: Список кортежей (номер кадра, координаты углов)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Видеофайл не найден: {video_path}")
            return []

        # Открываем видеофайл
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеофайл: {video_path}")
            return []

        # Получаем общее количество кадров
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Всего кадров в видео: {total_frames}")

        # Список для хранения результатов
        corners_list = []

        frame_idx = 0
        processed_count = 0
        detected_count = 0

        while processed_count < max_frames and frame_idx < total_frames:
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Не удалось прочитать кадр {frame_idx}")
                break

            if frame_idx % frame_step == 0:
                processed_count += 1

                if visualize:
                    success, corners, vis_frame = self.detect_corners(frame, visualize=True)
                else:
                    success, corners = self.detect_corners(frame)

                if success and corners is not None and len(corners) >= min_corners:
                    detected_count += 1
                    if visualize:
                        corners_list.append((frame_idx, corners, vis_frame))
                    else:
                        corners_list.append((frame_idx, corners))

                if processed_count % 10 == 0:
                    logger.debug(f"Обработано {processed_count} кадров, найдена доска на {detected_count} кадрах")

            frame_idx += 1

        cap.release()

        logger.info(f"Обработано {processed_count} кадров, найдена доска на {detected_count} кадрах")
        return corners_list

    def generate_object_points(self, num_frames: int) -> np.ndarray:
        """
        Генерирует массив 3D-точек для калибровки камеры.

        Args:
            num_frames: Количество кадров с найденными углами

        Returns:
            np.ndarray: Массив 3D-точек для калибровки
        """
        return [self.object_points for _ in range(num_frames)]

    def calibrate_camera(self, image_points: List[np.ndarray], image_size: Tuple[int, int]) -> Dict:
        """
        Калибрует камеру на основе найденных углов шахматной доски.

        Args:
            image_points: Список массивов с координатами углов на каждом изображении
            image_size: Размер изображения (ширина, высота)

        Returns:
            Dict: Результаты калибровки (camera_matrix, dist_coeffs, rvecs, tvecs, error)
        """
        object_points = self.generate_object_points(len(image_points))

        logger.info(f"Калибровка камеры по {len(image_points)} изображениям размером {image_size}")

        # Калибровка камеры
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )

        # Расчет ошибки репроекции
        total_error = 0
        for i in range(len(object_points)):
            image_points2, _ = cv2.projectPoints(
                object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
            total_error += error

        mean_error = total_error / len(object_points)
        logger.info(f"Средняя ошибка репроекции: {mean_error}")

        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "error": mean_error
        }

    def save_visualization(
            self,
            image: np.ndarray,
            corners: np.ndarray,
            output_path: Union[str, Path]
    ) -> bool:
        """
        Сохраняет визуализацию обнаруженных углов шахматной доски.

        Args:
            image: Исходное изображение
            corners: Обнаруженные углы
            output_path: Путь для сохранения результата

        Returns:
            bool: True, если сохранение прошло успешно
        """
        output_path = Path(output_path)

        # Создаем директорию, если она не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Создаем копию изображения
        vis_image = np.copy(image)

        # Рисуем углы
        cv2.drawChessboardCorners(vis_image, (self.width, self.height), corners, True)

        try:
            cv2.imwrite(str(output_path), vis_image)
            logger.debug(f"Визуализация сохранена в {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении визуализации в {output_path}: {e}")
            return False


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем объект шахматной доски
    board = ChessBoard(width=7, height=5, square_size=20.0)

    # Путь к тестовому изображению (при наличии)
    test_image_path = "test_image.jpg"
    if Path(test_image_path).exists():
        success, corners, vis_image = board.detect_corners_from_file(test_image_path, visualize=True)

        if success:
            logger.info(f"Найдено {len(corners)} углов на изображении")
            board.save_visualization(vis_image, corners, "detected_corners.jpg")