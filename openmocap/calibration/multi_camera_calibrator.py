"""
Модуль для калибровки системы из нескольких камер.
"""

import cv2
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

from openmocap.calibration.camera_calibrator import CameraCalibrator
from openmocap.calibration.chess_board import ChessBoard
from openmocap.utils.file_utils import save_toml, save_json, get_calibrations_dir
from openmocap.utils.video_utils import get_video_paths, get_video_properties
from openmocap.utils.geometry import triangulate_points, rodrigues_to_rotation_matrix

logger = logging.getLogger(__name__)


class MultiCameraCalibrator:
    """
    Класс для калибровки системы из нескольких камер.

    Attributes:
        chess_board (ChessBoard): Объект шахматной доски для обнаружения углов
        camera_calibrators (List[CameraCalibrator]): Список калибраторов отдельных камер
        camera_params (List[Dict]): Список параметров всех камер
        extrinsics (Dict): Внешние параметры камер (положение и ориентация относительно друг друга)
        is_calibrated (bool): Флаг, указывающий, что система камер откалибрована
    """

    def __init__(self, chess_board: Optional[ChessBoard] = None):
        """
        Инициализирует объект калибровки системы камер.

        Args:
            chess_board: Объект шахматной доски для обнаружения углов.
                        Если None, создается доска со стандартными параметрами
        """
        self.chess_board = chess_board or ChessBoard()
        self.camera_calibrators = []
        self.camera_params = []
        self.extrinsics = {}
        self.is_calibrated = False
        self.calibration_id = None
        self.metadata = {}

        logger.info("Инициализирован калибратор системы камер")

    def add_camera(self, camera_calibrator: CameraCalibrator) -> int:
        """
        Добавляет откалиброванную камеру в систему.

        Args:
            camera_calibrator: Калибратор с параметрами одиночной камеры

        Returns:
            int: Индекс добавленной камеры

        Raises:
            ValueError: Если камера не откалибрована
        """
        if not camera_calibrator.is_calibrated:
            raise ValueError("Камера не откалибрована")

        self.camera_calibrators.append(camera_calibrator)
        self.camera_params.append(camera_calibrator.camera_params)

        # Сбрасываем флаг калибровки системы, так как добавлена новая камера
        self.is_calibrated = False

        logger.info(f"Добавлена камера #{len(self.camera_calibrators) - 1} в систему")

        return len(self.camera_calibrators) - 1

    def calibrate_cameras_independently(
            self,
            video_paths: List[Union[str, Path]],
            max_frames: int = 30,
            frame_step: int = 5,
            min_corners_frames: int = 10
    ) -> List[Dict]:
        """
        Калибрует каждую камеру независимо по видеофайлам с шахматной доской.

        Args:
            video_paths: Список путей к видеофайлам
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами
            min_corners_frames: Минимальное количество кадров с обнаруженной доской

        Returns:
            List[Dict]: Список параметров калиброванных камер

        Raises:
            ValueError: Если не удалось откалибровать хотя бы одну камеру
        """
        self.camera_calibrators = []
        self.camera_params = []
        self.is_calibrated = False

        # Калибруем каждую камеру по отдельности
        for i, video_path in enumerate(video_paths):
            logger.info(f"Калибровка камеры #{i} по видео: {video_path}")
            calibrator = CameraCalibrator(chess_board=self.chess_board)

            try:
                camera_params = calibrator.calibrate_from_video(
                    video_path,
                    max_frames=max_frames,
                    frame_step=frame_step,
                    min_corners_frames=min_corners_frames
                )
                self.camera_calibrators.append(calibrator)
                self.camera_params.append(camera_params)
                logger.info(f"Камера #{i} успешно откалибрована")
            except Exception as e:
                logger.error(f"Ошибка при калибровке камеры #{i}: {e}")
                raise ValueError(f"Не удалось откалибровать камеру #{i}") from e

        # Добавляем идентификаторы камерам
        for i, params in enumerate(self.camera_params):
            params["camera_id"] = i
            params["camera_name"] = f"camera_{i}"

        # Устанавливаем метаданные
        self.metadata["calibration_time"] = datetime.now().isoformat()
        self.metadata["num_cameras"] = len(self.camera_calibrators)
        self.metadata["video_paths"] = [str(path) for path in video_paths]
        self.metadata["chess_board"] = {
            "width": self.chess_board.width,
            "height": self.chess_board.height,
            "square_size": self.chess_board.square_size
        }

        logger.info(f"Все {len(self.camera_calibrators)} камеры успешно откалиброваны независимо")

        return self.camera_params

    def _find_common_chessboard_frames(
            self,
            video_paths: List[Union[str, Path]],
            max_frames: int = 100,
            frame_step: int = 1
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        Находит кадры, на которых шахматная доска видна всеми камерами.

        Args:
            video_paths: Список путей к видеофайлам
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами

        Returns:
            Dict[int, List[Tuple[int, np.ndarray]]]: Словарь {номер_кадра: [(номер_камеры, углы_шахматной_доски)]}
        """
        # Поиск шахматной доски на всех видео
        camera_corners = []
        for i, video_path in enumerate(video_paths):
            corners_list = self.chess_board.detect_corners_from_video(
                video_path,
                max_frames=max_frames,
                frame_step=frame_step
            )
            # Преобразуем в словарь {номер_кадра: углы_шахматной_доски}
            frame_to_corners = {frame_idx: corners for frame_idx, corners in corners_list}
            camera_corners.append((i, frame_to_corners))
            logger.info(f"Найдено {len(corners_list)} кадров с шахматной доской для камеры #{i}")

        # Находим кадры, общие для всех камер
        common_frames = {}
        for cam_idx, frame_corners in camera_corners:
            for frame_idx, corners in frame_corners.items():
                if frame_idx not in common_frames:
                    common_frames[frame_idx] = []
                common_frames[frame_idx].append((cam_idx, corners))

        # Оставляем только кадры, где доска видна всеми камерами
        num_cameras = len(video_paths)
        valid_common_frames = {
            frame_idx: corners_list
            for frame_idx, corners_list in common_frames.items()
            if len(corners_list) == num_cameras
        }

        logger.info(f"Найдено {len(valid_common_frames)} кадров, где шахматная доска видна всеми камерами")

        return valid_common_frames

    def calibrate_cameras_stereo(
            self,
            video_paths: List[Union[str, Path]],
            max_frames: int = 100,
            frame_step: int = 1,
            min_common_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Калибрует систему камер по видеофайлам с шахматной доской.
        Включает калибровку каждой камеры по отдельности и вычисление
        взаимного расположения камер.

        Args:
            video_paths: Список путей к видеофайлам
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами
            min_common_frames: Минимальное количество кадров, где доска видна всеми камерами

        Returns:
            Dict[str, Any]: Параметры калиброванной системы камер

        Raises:
            ValueError: Если не удалось откалибровать систему камер
        """
        # Сначала калибруем каждую камеру независимо
        self.calibrate_cameras_independently(video_paths, max_frames, frame_step)

        # Находим кадры, где шахматная доска видна всеми камерами
        common_frames = self._find_common_chessboard_frames(video_paths, max_frames, frame_step)

        if len(common_frames) < min_common_frames:
            raise ValueError(
                f"Найдено только {len(common_frames)} общих кадров. "
                f"Требуется минимум {min_common_frames} кадров."
            )

        # Вычисляем внешние параметры камер (взаимное расположение)
        self._calculate_extrinsics(common_frames)

        # Устанавливаем флаг калибровки
        self.is_calibrated = True

        # Генерируем идентификатор калибровки
        self.calibration_id = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Обновляем метаданные
        self.metadata["calibration_time"] = datetime.now().isoformat()
        self.metadata["num_cameras"] = len(self.camera_calibrators)
        self.metadata["num_common_frames"] = len(common_frames)
        self.metadata["video_paths"] = [str(path) for path in video_paths]
        self.metadata["calibration_id"] = self.calibration_id
        self.metadata["calibration_type"] = "stereo"

        logger.info(f"Стерео калибровка завершена успешно. Калибровка ID: {self.calibration_id}")

        # Создаем результирующий словарь
        result = {
            "metadata": self.metadata,
            "camera_params": self.camera_params,
            "extrinsics": self.extrinsics
        }

        return result

    def _calculate_extrinsics(self, common_frames: Dict[int, List[Tuple[int, np.ndarray]]]) -> None:
        """
        Вычисляет внешние параметры камер (взаимное расположение).

        Args:
            common_frames: Словарь {номер_кадра: [(номер_камеры, углы_шахматной_доски)]}
        """
        num_cameras = len(self.camera_calibrators)
        self.extrinsics = {
            "rotations": [],  # Матрицы поворота относительно камеры #0
            "translations": [],  # Векторы перемещения относительно камеры #0
            "projection_matrices": []  # Матрицы проекции для каждой камеры
        }

        # Стереокалибровка. Камера #0 считается центральной (мировая система координат)
        # Для каждой камеры рассчитываем матрицу поворота и вектор перемещения
        # относительно камеры #0
        for cam_idx in range(num_cameras):
            if cam_idx == 0:
                # Для камеры #0 поворот = единичная матрица, перемещение = нулевой вектор
                R = np.eye(3)
                t = np.zeros(3)
            else:
                # Для остальных камер вычисляем взаимное расположение относительно камеры #0
                R, t = self._calculate_camera_relative_pose(cam_idx, common_frames)

            # Сохраняем результаты
            self.extrinsics["rotations"].append(R.tolist())
            self.extrinsics["translations"].append(t.tolist())

            # Вычисляем матрицу проекции
            P = self._calculate_projection_matrix(cam_idx, R, t)
            self.extrinsics["projection_matrices"].append(P.tolist())

        logger.info(f"Рассчитано взаимное расположение {num_cameras} камер")

    def _calculate_camera_relative_pose(
            self,
            cam_idx: int,
            common_frames: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет положение и ориентацию камеры относительно камеры #0.

        Args:
            cam_idx: Индекс камеры
            common_frames: Словарь {номер_кадра: [(номер_камеры, углы_шахматной_доски)]}

        Returns:
            Tuple[np.ndarray, np.ndarray]: (матрица поворота, вектор перемещения)
        """
        # Извлекаем общие точки шахматной доски
        points_3d = []  # Мировые координаты точек (объектные координаты шахматной доски)
        points_2d_cam0 = []  # Проекции точек на камеру #0
        points_2d_cam = []  # Проекции точек на целевую камеру

        # Получаем параметры камер
        camera_matrix_0 = np.array(self.camera_params[0]["camera_matrix"])
        dist_coeffs_0 = np.array(self.camera_params[0]["dist_coeffs"])
        camera_matrix = np.array(self.camera_params[cam_idx]["camera_matrix"])
        dist_coeffs = np.array(self.camera_params[cam_idx]["dist_coeffs"])

        # Для каждого общего кадра
        for frame_idx, corners_list in common_frames.items():
            # Сортируем по индексу камеры
            corners_list.sort(key=lambda x: x[0])

            # Получаем углы шахматной доски для обеих камер
            corners_0 = None
            corners_cam = None

            for c_idx, corners in corners_list:
                if c_idx == 0:
                    corners_0 = corners
                elif c_idx == cam_idx:
                    corners_cam = corners

            if corners_0 is None or corners_cam is None:
                continue

            # Создаем объектные точки для шахматной доски
            object_points = self.chess_board.object_points

            # Добавляем точки
            points_3d.append(object_points)
            points_2d_cam0.append(corners_0)
            points_2d_cam.append(corners_cam)

        # Если у нас недостаточно общих точек, выводим предупреждение
        if len(points_3d) < 3:
            logger.warning(f"Для камеры #{cam_idx} найдено только {len(points_3d)} общих кадров. "
                          "Это может привести к неточной стерео калибровке.")

        # Устраняем искажения для точек камеры #0
        undistorted_points_0 = []
        for corners in points_2d_cam0:
            undistorted = cv2.undistortPoints(
                corners,
                camera_matrix_0,
                dist_coeffs_0,
                P=camera_matrix_0
            )
            undistorted_points_0.append(undistorted)

        # Устраняем искажения для точек целевой камеры
        undistorted_points = []
        for corners in points_2d_cam:
            undistorted = cv2.undistortPoints(
                corners,
                camera_matrix,
                dist_coeffs,
                P=camera_matrix
            )
            undistorted_points.append(undistorted)

        # Для стерео калибровки используем cv2.stereoCalibrate, но нам нужны только R и t
        flags = cv2.CALIB_FIX_INTRINSIC  # Фиксируем внутренние параметры камер

        ret, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
            points_3d,
            points_2d_cam0,
            points_2d_cam,
            camera_matrix_0,
            dist_coeffs_0,
            camera_matrix,
            dist_coeffs,
            None,
            flags=flags
        )

        logger.info(f"Стерео калибровка для камеры #{cam_idx} завершена с ошибкой: {ret}")

        return R, t.reshape(3)

    def _calculate_projection_matrix(
            self,
            cam_idx: int,
            R: np.ndarray,
            t: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет матрицу проекции для камеры.

        Args:
            cam_idx: Индекс камеры
            R: Матрица поворота относительно камеры #0
            t: Вектор перемещения относительно камеры #0

        Returns:
            np.ndarray: Матрица проекции 3x4
        """
        # Получаем внутренние параметры камеры
        camera_matrix = np.array(self.camera_params[cam_idx]["camera_matrix"])

        # Создаем матрицу внешних параметров [R|t]
        extrinsic_matrix = np.zeros((3, 4))
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t

        # Матрица проекции = K * [R|t]
        projection_matrix = camera_matrix @ extrinsic_matrix

        return projection_matrix

    def triangulate_3d_points(
            self,
            points_2d: np.ndarray,
            camera_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Триангулирует 3D координаты точек по их проекциям на камеры.

        Args:
            points_2d: Массив 2D-точек shape (n_cameras, n_points, 2)
            camera_indices: Список индексов камер для использования.
                           Если None, используются все камеры.

        Returns:
            np.ndarray: Массив 3D-точек shape (n_points, 3)

        Raises:
            ValueError: Если система камер не откалибрована
        """
        if not self.is_calibrated:
            raise ValueError("Система камер не откалибрована")

        if camera_indices is None:
            camera_indices = list(range(len(self.camera_calibrators)))

        # Получаем матрицы проекции для выбранных камер
        projection_matrices = []
        for idx in camera_indices:
            projection_matrices.append(
                np.array(self.extrinsics["projection_matrices"][idx])
            )

        projection_matrices = np.array(projection_matrices)

        # Выбираем только указанные камеры из points_2d
        points_2d_selected = points_2d[camera_indices]

        # Триангулируем точки
        points_3d = triangulate_points(
            points_2d=points_2d_selected,
            projection_matrices=projection_matrices,
            min_cameras=2
        )

        return points_3d

    def undistort_points(
            self,
            points_2d: np.ndarray,
            camera_idx: int
    ) -> np.ndarray:
        """
        Устраняет искажения для набора 2D-точек с конкретной камеры.

        Args:
            points_2d: Массив 2D-точек shape (n_points, 2) или (n_frames, n_points, 2)
            camera_idx: Индекс камеры

        Returns:
            np.ndarray: Массив точек с устраненными искажениями

        Raises:
            ValueError: Если система камер не откалибрована или индекс камеры некорректен
        """
        if not self.is_calibrated:
            raise ValueError("Система камер не откалибрована")

        if camera_idx < 0 or camera_idx >= len(self.camera_calibrators):
            raise ValueError(f"Некорректный индекс камеры: {camera_idx}")

        return self.camera_calibrators[camera_idx].undistort_points(points_2d)

    def save_calibration(
            self,
            output_path: Optional[Union[str, Path]] = None,
            format: str = "toml"
    ) -> str:
        """
        Сохраняет параметры калибровки в файл.

        Args:
            output_path: Путь для сохранения файла. Если None, файл сохраняется
                        в стандартный каталог с автоматическим именем.
            format: Формат файла (json или toml)

        Returns:
            str: Путь к сохраненному файлу

        Raises:
            ValueError: Если система камер не откалибрована или формат не поддерживается
        """
        if not self.is_calibrated:
            raise ValueError("Система камер не откалибрована")

        # Формируем полный словарь калибровки
        calibration_data = {
            "metadata": self.metadata,
            "camera_params": self.camera_params,
            "extrinsics": self.extrinsics
        }

        # Если путь не указан, создаем его автоматически
        if output_path is None:
            calibrations_dir = get_calibrations_dir()
            output_path = calibrations_dir / f"{self.calibration_id}.{format.lower()}"

        output_path = Path(output_path)

        # Создаем директорию, если она не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем файл в выбранном формате
        if format.lower() == "json":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".json")
            save_json(calibration_data, output_path)
        elif format.lower() == "toml":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".toml")
            save_toml(calibration_data, output_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}. Используйте 'json' или 'toml'")

        logger.info(f"Параметры калибровки системы камер сохранены в {output_path}")

        return str(output_path)

    def load_calibration(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Загружает параметры калибровки из файла.

        Args:
            file_path: Путь к файлу с параметрами

        Returns:
            Dict[str, Any]: Параметры калиброванной системы камер

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Определяем формат файла по расширению
        if file_path.suffix.lower() == ".json":
            import json
            with open(file_path, "r") as f:
                calibration_data = json.load(f)
        elif file_path.suffix.lower() == ".toml":
            import toml
            calibration_data = toml.load(str(file_path))
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")

        # Загружаем данные
        self.metadata = calibration_data.get("metadata", {})
        self.camera_params = calibration_data.get("camera_params", [])
        self.extrinsics = calibration_data.get("extrinsics", {})
        self.calibration_id = self.metadata.get("calibration_id")

        # Создаем калибраторы для отдельных камер
        self.camera_calibrators = []
        for cam_params in self.camera_params:
            calibrator = CameraCalibrator()
            # Настраиваем параметры калибратора
            calibrator.camera_params = cam_params
            calibrator.is_calibrated = True

            # Настраиваем шахматную доску, если информация о ней есть
            chess_board_info = self.metadata.get("chess_board")
            if chess_board_info:
                calibrator.chess_board = ChessBoard(
                    width=chess_board_info.get("width", 7),
                    height=chess_board_info.get("height", 5),
                    square_size=chess_board_info.get("square_size", 25.0)
                )

            self.camera_calibrators.append(calibrator)

        # Устанавливаем флаг калибровки
        self.is_calibrated = True

        logger.info(f"Параметры калибровки системы камер загружены из {file_path}")

        return calibration_data

    @classmethod
    def from_calibration_file(cls, file_path: Union[str, Path]) -> "MultiCameraCalibrator":
        """
        Создает объект MultiCameraCalibrator из файла калибровки.

        Args:
            file_path: Путь к файлу с параметрами калибровки

        Returns:
            MultiCameraCalibrator: Объект калибратора системы камер

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        calibrator = cls()
        calibrator.load_calibration(file_path)
        return calibrator


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем объект калибратора с нестандартной шахматной доской
    chess_board = ChessBoard(width=7, height=7, square_size=50.0)
    calibrator = MultiCameraCalibrator(chess_board=chess_board)

    # Пример калибровки по видеофайлам
    try:
        # Находим видеофайлы
        video_dir = r"C:\Users\vczyp\openmocap_data\recordings\recording_20250322_234228"
        if Path(video_dir).exists():
            video_paths = get_video_paths(video_dir)

            if len(video_paths) >= 2:
                # Калибруем систему камер
                calibration_data = calibrator.calibrate_cameras_stereo(
                    video_paths,
                    max_frames=30,
                    frame_step=5,
                    min_common_frames=5
                )

                # Сохраняем калибровку
                calibration_file = calibrator.save_calibration(format="toml")
                print(f"Калибровка сохранена в {calibration_file}")

                # Загружаем калибровку
                loaded_calibrator = MultiCameraCalibrator.from_calibration_file(calibration_file)
                print(f"Загружена калибровка с ID: {loaded_calibrator.calibration_id}")
                print(f"Количество камер: {len(loaded_calibrator.camera_calibrators)}")
            else:
                print(f"Для калибровки системы камер требуется минимум 2 видеофайла, найдено: {len(video_paths)}")
        else:
            print(f"Директория с видеофайлами {video_dir} не найдена")
    except Exception as e:
        logger.error(f"Ошибка при калибровке: {e}")