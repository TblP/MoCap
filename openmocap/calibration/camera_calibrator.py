"""
Модуль для калибровки одиночной камеры.
"""

import cv2
import logging
import numpy as np
import os
from pathlib import Path
import json
import toml
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

from openmocap.calibration.chess_board import ChessBoard
from openmocap.utils.file_utils import save_json, save_toml

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """
    Класс для калибровки одиночной камеры с использованием шахматной доски.

    Attributes:
        chess_board (ChessBoard): Объект шахматной доски для обнаружения углов
        camera_params (Dict): Параметры калиброванной камеры
        is_calibrated (bool): Флаг, указывающий, что камера откалибрована
    """

    def __init__(self, chess_board: Optional[ChessBoard] = None):
        """
        Инициализирует объект калибровки камеры.

        Args:
            chess_board: Объект шахматной доски для обнаружения углов.
                        Если None, создается доска со стандартными параметрами
        """
        self.chess_board = chess_board or ChessBoard()
        self.camera_params = None
        self.is_calibrated = False

        logger.info("Инициализирован калибратор камеры")

    def calibrate_from_images(
            self,
            image_paths: List[Union[str, Path]],
            image_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Калибрует камеру по списку изображений с шахматной доской.

        Args:
            image_paths: Список путей к изображениям
            image_size: Размер изображения (ширина, высота). Если None, определяется из первого изображения

        Returns:
            Dict: Параметры калиброванной камеры

        Raises:
            ValueError: Если не удалось обнаружить доску хотя бы на одном изображении
        """
        if not image_paths:
            raise ValueError("Список изображений пуст")

        image_points = []
        used_paths = []

        # Если размер изображения не указан, определяем его из первого изображения
        if image_size is None:
            first_image = cv2.imread(str(image_paths[0]))
            if first_image is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_paths[0]}")
            image_size = (first_image.shape[1], first_image.shape[0])  # (width, height)

        # Обнаруживаем углы на каждом изображении
        for path in image_paths:
            success, corners = self.chess_board.detect_corners_from_file(path)

            if success and corners is not None:
                image_points.append(corners)
                used_paths.append(str(path))
                logger.debug(f"Найдена доска на изображении: {path}")

        if not image_points:
            raise ValueError("Не удалось обнаружить шахматную доску ни на одном изображении")

        # Калибруем камеру
        self.camera_params = self.chess_board.calibrate_camera(image_points, image_size)

        # Добавляем дополнительную информацию
        self.camera_params["image_size"] = image_size
        self.camera_params["num_images"] = len(image_points)
        self.camera_params["used_images"] = used_paths
        self.camera_params["calibration_time"] = datetime.now().isoformat()
        self.camera_params["chess_board"] = {
            "width": self.chess_board.width,
            "height": self.chess_board.height,
            "square_size": self.chess_board.square_size
        }

        self.is_calibrated = True

        logger.info(
            f"Калибровка завершена по {len(image_points)} изображениям. "
            f"Ошибка репроекции: {self.camera_params['error']:.6f}"
        )

        return self.camera_params

    def calibrate_from_video(
            self,
            video_path: Union[str, Path],
            max_frames: int = 50,
            frame_step: int = 10,
            min_corners_frames: int = 10
    ) -> Dict:
        """
        Калибрует камеру по видео с шахматной доской.

        Args:
            video_path: Путь к видеофайлу
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами
            min_corners_frames: Минимальное количество кадров с обнаруженной доской

        Returns:
            Dict: Параметры калиброванной камеры

        Raises:
            ValueError: Если не удалось обнаружить доску хотя бы на min_corners_frames кадрах
        """
        # Открываем видеофайл для определения размера кадра
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

        ret, first_frame = cap.read()

        if not ret:
            cap.release()
            raise ValueError(f"Не удалось прочитать первый кадр из видеофайла: {video_path}")

        image_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
        cap.release()

        # Обнаруживаем углы на кадрах видео
        corners_list = self.chess_board.detect_corners_from_video(
            video_path,
            max_frames=max_frames,
            frame_step=frame_step
        )

        if len(corners_list) < min_corners_frames:
            raise ValueError(
                f"Найдено только {len(corners_list)} кадров с шахматной доской. "
                f"Требуется минимум {min_corners_frames} кадров."
            )

        # Извлекаем координаты углов и номера кадров
        image_points = [corners for _, corners in corners_list]
        frame_indices = [idx for idx, _ in corners_list]

        # Калибруем камеру
        self.camera_params = self.chess_board.calibrate_camera(image_points, image_size)

        # Добавляем дополнительную информацию
        self.camera_params["image_size"] = image_size
        self.camera_params["num_frames"] = len(image_points)
        self.camera_params["frame_indices"] = frame_indices
        self.camera_params["video_path"] = str(video_path)
        self.camera_params["calibration_time"] = datetime.now().isoformat()
        self.camera_params["chess_board"] = {
            "width": self.chess_board.width,
            "height": self.chess_board.height,
            "square_size": self.chess_board.square_size
        }

        self.is_calibrated = True

        logger.info(
            f"Калибровка завершена по {len(image_points)} кадрам из видео. "
            f"Ошибка репроекции: {self.camera_params['error']:.6f}"
        )

        return self.camera_params

    def save_calibration(
            self,
            output_path: Union[str, Path],
            format: str = "json"
    ) -> str:
        """
        Сохраняет параметры калибровки в файл.

        Args:
            output_path: Путь для сохранения файла
            format: Формат файла (json или toml)

        Returns:
            str: Путь к сохраненному файлу

        Raises:
            ValueError: Если камера не откалибрована или формат не поддерживается
        """
        if not self.is_calibrated:
            raise ValueError("Камера не откалибрована")

        # Преобразуем параметры в формат, подходящий для сериализации
        serializable_params = self._make_serializable(self.camera_params)

        output_path = Path(output_path)

        # Создаем директорию, если она не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем файл в выбранном формате
        if format.lower() == "json":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".json")
            save_json(serializable_params, output_path)
        elif format.lower() == "toml":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".toml")
            save_toml(serializable_params, output_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}. Используйте 'json' или 'toml'")

        logger.info(f"Параметры калибровки сохранены в {output_path}")

        return str(output_path)

    def load_calibration(self, file_path: Union[str, Path]) -> Dict:
        """
        Загружает параметры калибровки из файла.

        Args:
            file_path: Путь к файлу с параметрами

        Returns:
            Dict: Параметры калиброванной камеры

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Определяем формат файла по расширению
        if file_path.suffix.lower() == ".json":
            with open(file_path, "r") as f:
                params = json.load(f)
        elif file_path.suffix.lower() == ".toml":
            params = toml.load(str(file_path))
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        # Преобразуем параметры из сериализованного формата
        self.camera_params = self._from_serializable(params)
        self.is_calibrated = True

        # Восстанавливаем параметры шахматной доски, если они есть
        if "chess_board" in params:
            self.chess_board = ChessBoard(
                width=params["chess_board"]["width"],
                height=params["chess_board"]["height"],
                square_size=params["chess_board"]["square_size"]
            )

        logger.info(f"Параметры калибровки загружены из {file_path}")

        return self.camera_params

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Устраняет искажения на изображении с использованием параметров калибровки.

        Args:
            image: Исходное изображение

        Returns:
            np.ndarray: Изображение с устраненными искажениями

        Raises:
            ValueError: Если камера не откалибрована
        """
        if not self.is_calibrated:
            raise ValueError("Камера не откалибрована")

        camera_matrix = self.camera_params["camera_matrix"]
        dist_coeffs = self.camera_params["dist_coeffs"]

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Устраняем искажения
        dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Обрезаем изображение по ROI
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Устраняет искажения для набора точек с использованием параметров калибровки.

        Args:
            points: Массив точек shape (N, 2) или (N, 1, 2)

        Returns:
            np.ndarray: Массив точек с устраненными искажениями

        Raises:
            ValueError: Если камера не откалибрована
        """
        if not self.is_calibrated:
            raise ValueError("Камера не откалибрована")

        camera_matrix = self.camera_params["camera_matrix"]
        dist_coeffs = self.camera_params["dist_coeffs"]

        # Преобразуем точки в формат (N, 1, 2) для cv2.undistortPoints
        points_reshaped = points.reshape(-1, 1, 2)

        # Устраняем искажения
        undistorted_points = cv2.undistortPoints(
            points_reshaped,
            camera_matrix,
            dist_coeffs,
            P=camera_matrix
        )

        # Возвращаем в исходный формат
        return undistorted_points.reshape(points.shape)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Возвращает матрицу проекции камеры (3x4).

        Returns:
            np.ndarray: Матрица проекции

        Raises:
            ValueError: Если камера не откалибрована
        """
        if not self.is_calibrated:
            raise ValueError("Камера не откалибрована")

        # Матрица проекции = [R|t]
        # Для камеры в мировой системе координат R = I, t = [0,0,0]
        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = np.eye(3)
        proj_matrix = self.camera_params["camera_matrix"] @ proj_matrix

        return proj_matrix

    def _make_serializable(self, params: Dict) -> Dict:
        """
        Преобразует параметры калибровки в сериализуемый формат.

        Args:
            params: Параметры калибровки

        Returns:
            Dict: Сериализуемые параметры
        """
        serializable = {}

        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable[key] = [item.tolist() for item in value]
            else:
                serializable[key] = value

        return serializable

    def _from_serializable(self, params: Dict) -> Dict:
        """
        Преобразует сериализованные параметры обратно в рабочий формат.

        Args:
            params: Сериализованные параметры

        Returns:
            Dict: Рабочие параметры
        """
        working_params = {}

        for key, value in params.items():
            if key in ["camera_matrix", "dist_coeffs"]:
                working_params[key] = np.array(value)
            elif key in ["rvecs", "tvecs"] and isinstance(value, list):
                working_params[key] = [np.array(item) for item in value]
            else:
                working_params[key] = value

        return working_params


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем объект калибратора
    calibrator = CameraCalibrator(chess_board=ChessBoard(width=7, height=5, square_size=20.0))

    # Пример калибровки по видеофайлу
    try:
        video_path = "calibration_video.mp4"
        if Path(video_path).exists():
            camera_params = calibrator.calibrate_from_video(
                video_path,
                max_frames=30,
                frame_step=5
            )

            # Сохраняем параметры
            calibrator.save_calibration("camera_calibration.json")

            # Загружаем параметры
            calibrator2 = CameraCalibrator()
            calibrator2.load_calibration("camera_calibration.json")

            print(f"Ошибка репроекции: {calibrator2.camera_params['error']}")
    except Exception as e:
        logger.error(f"Ошибка при калибровке: {e}")