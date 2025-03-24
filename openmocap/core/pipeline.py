"""
Основной конвейер обработки данных для системы OpenMoCap.

Предоставляет классы и функции для построения и выполнения конвейера обработки
от калибровки камер до экспорта результатов.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.calibration.multi_camera_calibrator import MultiCameraCalibrator
from openmocap.tracking.base_tracker import BaseTracker
from openmocap.tracking.mediapipe_tracker import MediaPipeTracker
from openmocap.reconstruction.triangulation import Triangulator
from openmocap.reconstruction.reprojection import ReprojectionErrorAnalyzer, filter_outliers, refine_points_3d
from openmocap.export.csv_exporter import CSVExporter
from openmocap.export.animation_calculator import AnimationCalculator
from openmocap.export.json_exporter import JSONExporter
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Класс для построения и выполнения конвейера обработки данных в OpenMoCap.

    Attributes:
        calibrator (MultiCameraCalibrator): Калибратор системы камер
        tracker (BaseTracker): Трекер для отслеживания точек на видео
        triangulator (Triangulator): Объект для триангуляции 3D-точек
        results (Dict): Словарь с результатами обработки
    """

    def __init__(self):
        """
        Инициализирует конвейер обработки данных с настройками по умолчанию.
        """
        self.calibrator = None
        self.tracker = None
        self.triangulator = None
        self.results = {
            'camera_params': None,
            'points_2d': None,
            'points_3d': None,
            'reprojection_errors': None,
            'filtered_points_3d': None,
            'joint_angles': None
        }
        logger.info("Инициализирован конвейер обработки OpenMoCap")

    def set_calibrator(self, calibrator: MultiCameraCalibrator) -> 'Pipeline':
        """
        Устанавливает калибратор системы камер.

        Args:
            calibrator: Калибратор системы камер

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)
        """
        self.calibrator = calibrator
        logger.info("Установлен калибратор системы камер")
        return self

    def set_tracker(self, tracker: BaseTracker) -> 'Pipeline':
        """
        Устанавливает трекер для отслеживания точек.

        Args:
            tracker: Трекер для отслеживания точек

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)
        """
        self.tracker = tracker
        logger.info(f"Установлен трекер: {tracker.name}")
        return self

    def calibrate_cameras(
            self,
            video_paths: List[Union[str, Path]],
            max_frames: int = 100,
            frame_step: int = 5,
            min_common_frames: int = 5
    ) -> 'Pipeline':
        """
        Калибрует систему камер по видеофайлам с шахматной доской.

        Args:
            video_paths: Список путей к видеофайлам
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между обрабатываемыми кадрами
            min_common_frames: Минимальное количество кадров, где доска видна всеми камерами

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если калибратор не установлен
        """
        if self.calibrator is None:
            self.calibrator = MultiCameraCalibrator()
            logger.info("Создан калибратор системы камер по умолчанию")

        logger.info(f"Начинается калибровка {len(video_paths)} камер")
        calibration_data = self.calibrator.calibrate_cameras_stereo(
            video_paths=video_paths,
            max_frames=max_frames,
            frame_step=frame_step,
            min_common_frames=min_common_frames
        )

        self.results['camera_params'] = calibration_data

        # Создаем объект триангулятора на основе параметров камер
        projection_matrices = np.array([
            np.array(matrix) for matrix in calibration_data['extrinsics']['projection_matrices']
        ])

        camera_names = [f"camera_{i}" for i in range(len(projection_matrices))]
        self.triangulator = Triangulator(projection_matrices, camera_names)

        logger.info("Калибровка камер завершена успешно")
        return self

    def track_videos(
            self,
            video_paths: List[Union[str, Path]],
            start_frame: int = 0,
            end_frame: Optional[int] = None,
            step: int = 1
    ) -> 'Pipeline':
        """
        Отслеживает точки на всех видео.

        Args:
            video_paths: Список путей к видеофайлам
            start_frame: Начальный кадр для обработки
            end_frame: Конечный кадр для обработки
            step: Шаг между обрабатываемыми кадрами

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если трекер не установлен
        """
        if self.tracker is None:
            self.tracker = MediaPipeTracker()
            logger.info("Создан трекер MediaPipe по умолчанию")

        logger.info(f"Начинается отслеживание точек на {len(video_paths)} видео")
        points_2d = self.tracker.track_videos(
            video_paths=video_paths,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
            show_progress=True
        )

        self.results['points_2d'] = points_2d
        self.results['landmark_names'] = self.tracker.landmark_names

        logger.info("Отслеживание точек завершено успешно")
        return self

    def triangulate(self) -> 'Pipeline':
        """
        Триангулирует 3D-точки из 2D-координат.

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если триангулятор не создан или данные отслеживания не доступны
        """

        if self.triangulator is None:
            raise ValueError("Триангулятор не инициализирован. Сначала выполните калибровку камер.")

        if self.results['points_2d'] is None:
            raise ValueError("Данные отслеживания точек не доступны. Сначала выполните отслеживание точек.")

        logger.info("Начинается триангуляция 3D-точек")
        # Получаем размеры данных
        n_cameras, n_frames, n_landmarks, _ = self.results['points_2d'].shape

        # Создаем массив для хранения 3D-точек
        points_3d = np.full((n_frames, n_landmarks, 3), np.nan)
        logger.info(f"points_2d shape: {self.results['points_2d'].shape}")
        logger.info(f"points_3d shape: {points_3d.shape}")
        # Триангулируем каждый кадр
        for frame in range(n_frames):
            points_2d_frame = self.results['points_2d'][:, frame, :, :2]

            # Создаем маску видимости
            visibility = ~np.isnan(points_2d_frame[:, :, 0])

            # Триангулируем точки
            points_3d_frame = self.triangulator.triangulate_points(points_2d_frame, visibility)
            points_3d[frame] = points_3d_frame

        self.results['points_3d'] = points_3d

        # Вычисляем ошибки репроекции
        reprojection_analyzer = ReprojectionErrorAnalyzer(
            camera_matrices=self.triangulator.camera_matrices,
            camera_names=self.triangulator.camera_names
        )

        error_distances, _ = reprojection_analyzer.calculate_reprojection_errors(
            points_3d=points_3d,
            points_2d=self.results['points_2d']
        )
        logger.info(f"points_2d shape: {self.results['points_2d'].shape}")
        logger.info(f"points_3d shape: {points_3d.shape}")
        self.results['reprojection_errors'] = error_distances

        logger.info("Триангуляция 3D-точек завершена успешно")
        return self

    def filter_data(
            self,
            threshold_method: str = 'percentile',
            threshold_value: float = 95.0
    ) -> 'Pipeline':
        """
        Фильтрует данные для устранения выбросов.

        Args:
            threshold_method: Метод определения порога ('percentile', 'std', 'absolute')
            threshold_value: Значение порога

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если данные триангуляции не доступны
        """
        if self.results['points_3d'] is None:
            raise ValueError("Данные триангуляции не доступны. Сначала выполните триангуляцию.")

        logger.info(f"Начинается фильтрация данных (метод: {threshold_method}, порог: {threshold_value})")

        filtered_points_3d = filter_outliers(
            points_3d=self.results['points_3d'],
            points_2d=self.results['points_2d'],
            projection_matrices=self.triangulator.camera_matrices,
            threshold_method=threshold_method,
            threshold_value=threshold_value
        )

        self.results['filtered_points_3d'] = filtered_points_3d

        logger.info("Фильтрация данных завершена успешно")
        return self

    def refine_data(
            self,
            method: str = 'levenberg_marquardt',
            max_iterations: int = 100
    ) -> 'Pipeline':
        """
        Уточняет 3D-координаты точек.

        Args:
            method: Метод оптимизации ('levenberg_marquardt', 'gauss_newton')
            max_iterations: Максимальное число итераций

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если данные триангуляции не доступны
        """
        if self.results['points_3d'] is None and self.results['filtered_points_3d'] is None:
            raise ValueError("Данные триангуляции не доступны. Сначала выполните триангуляцию.")

        points_3d = self.results['filtered_points_3d'] if self.results['filtered_points_3d'] is not None else \
        self.results['points_3d']

        logger.info(f"Начинается уточнение 3D-координат (метод: {method})")

        refined_points_3d = refine_points_3d(
            points_3d=points_3d,
            points_2d=self.results['points_2d'],
            projection_matrices=self.triangulator.camera_matrices,
            method=method,
            max_iterations=max_iterations
        )

        self.results['refined_points_3d'] = refined_points_3d

        logger.info("Уточнение 3D-координат завершено успешно")
        return self

    def calculate_joint_angles(self) -> 'Pipeline':
        """
        Вычисляет углы суставов.

        Returns:
            Pipeline: Экземпляр конвейера (для цепочки вызовов)

        Raises:
            ValueError: Если данные триангуляции не доступны
        """
        # Используем обработанные данные, если доступны
        if self.results['refined_points_3d'] is not None:
            points_3d = self.results['refined_points_3d']
        elif self.results['filtered_points_3d'] is not None:
            points_3d = self.results['filtered_points_3d']
        elif self.results['points_3d'] is not None:
            points_3d = self.results['points_3d']
        else:
            raise ValueError("Данные триангуляции не доступны. Сначала выполните триангуляцию.")

        if not hasattr(self.tracker, 'skeleton_model'):
            logger.warning("Трекер не имеет модели скелета. Углы суставов не могут быть вычислены.")
            return self

        logger.info("Начинается вычисление углов суставов")

        joint_angles = self.tracker.skeleton_model.calculate_joint_angles(points_3d)
        self.results['joint_angles'] = joint_angles

        logger.info("Вычисление углов суставов завершено успешно")
        return self

    def export_results(
            self,
            output_folder: Union[str, Path],
            export_2d: bool = True,
            export_3d: bool = True,
            export_filtered: bool = True,
            export_refined: bool = True,
            export_angles: bool = True,
            export_skeleton: bool = True,
            export_animation: bool = True,  # Параметр для экспорта анимации
            export_format: str = 'csv'  # Новый параметр: 'csv', 'json' или 'both'
    ) -> Dict[str, str]:
        """
        Экспортирует результаты обработки в CSV и/или JSON файлы.

        Args:
            output_folder: Путь к папке для сохранения результатов
            export_2d: Экспортировать 2D-координаты
            export_3d: Экспортировать 3D-координаты
            export_filtered: Экспортировать отфильтрованные 3D-координаты
            export_refined: Экспортировать уточненные 3D-координаты
            export_angles: Экспортировать углы суставов
            export_skeleton: Экспортировать модель скелета
            export_animation: Экспортировать параметры анимации
            export_format: Формат экспорта данных ('csv', 'json' или 'both')

        Returns:
            Dict[str, str]: Словарь с путями к экспортированным файлам

        Raises:
            ValueError: Если нет данных для экспорта или неподдерживаемый формат
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Проверяем формат экспорта
        if export_format not in ["csv", "json", "both"]:
            raise ValueError(
                f"Неподдерживаемый формат экспорта: {export_format}. Поддерживаются 'csv', 'json' и 'both'")

        # Создаем экспортеры в зависимости от формата
        exporters = {}
        if export_format in ["csv", "both"]:
            exporters["csv"] = CSVExporter()
        if export_format in ["json", "both"]:
            exporters["json"] = JSONExporter()

        exported_files = {}

        # Подготовка метаданных
        metadata = {
            "tracker": self.tracker.name if self.tracker else "unknown",
            "num_cameras": len(self.triangulator.camera_matrices) if self.triangulator else 0,
            "num_landmarks": len(self.tracker.landmark_names) if self.tracker else 0
        }

        # Для каждого формата экспорта
        for format_name, exporter in exporters.items():
            # Экспорт 2D-координат (только для CSV, т.к. для JSON они не так актуальны)
            if export_2d and self.results['points_2d'] is not None and format_name == "csv":
                output_path = output_folder / f"points_2d.{format_name}"
                exporter.export_points_2d(
                    points_2d=self.results['points_2d'],
                    output_path=output_path,
                    landmark_names=self.results.get('landmark_names'),
                    camera_names=self.triangulator.camera_names if self.triangulator else None,
                    include_confidence=True,
                    metadata=metadata
                )
                exported_files[f'points_2d_{format_name}'] = str(output_path)
                logger.info(f"2D-координаты экспортированы в {output_path}")

            # Экспорт 3D-координат
            if export_3d and self.results['points_3d'] is not None:
                output_path = output_folder / f"points_3d.{format_name}"
                exporter.export_points_3d(
                    points_3d=self.results['points_3d'],
                    output_path=output_path,
                    landmark_names=self.results.get('landmark_names'),
                    metadata=metadata
                )
                exported_files[f'points_3d_{format_name}'] = str(output_path)
                logger.info(f"3D-координаты экспортированы в {output_path}")

            # Экспорт отфильтрованных 3D-координат
            if export_filtered and self.results['filtered_points_3d'] is not None:
                output_path = output_folder / f"filtered_points_3d.{format_name}"
                exporter.export_points_3d(
                    points_3d=self.results['filtered_points_3d'],
                    output_path=output_path,
                    landmark_names=self.results.get('landmark_names'),
                    metadata={**metadata, "filter_applied": True}
                )
                exported_files[f'filtered_points_3d_{format_name}'] = str(output_path)
                logger.info(f"Отфильтрованные 3D-координаты экспортированы в {output_path}")

            # Экспорт уточненных 3D-координат
            if export_refined and self.results['refined_points_3d'] is not None:
                output_path = output_folder / f"refined_points_3d.{format_name}"
                exporter.export_points_3d(
                    points_3d=self.results['refined_points_3d'],
                    output_path=output_path,
                    landmark_names=self.results.get('landmark_names'),
                    metadata={**metadata, "refinement_applied": True}
                )
                exported_files[f'refined_points_3d_{format_name}'] = str(output_path)
                logger.info(f"Уточненные 3D-координаты экспортированы в {output_path}")

            # Экспорт углов суставов
            if export_angles and self.results['joint_angles'] is not None:
                output_path = output_folder / f"joint_angles.{format_name}"
                exporter.export_joint_angles(
                    joint_angles=self.results['joint_angles'],
                    output_path=output_path,
                    metadata=metadata
                )
                exported_files[f'joint_angles_{format_name}'] = str(output_path)
                logger.info(f"Углы суставов экспортированы в {output_path}")

            # Экспорт модели скелета
            if export_skeleton and hasattr(self.tracker, 'skeleton_model'):
                output_path = output_folder / f"skeleton_model.{format_name}"
                exporter.export_skeleton_model(
                    landmark_names=self.tracker.landmark_names,
                    connections=self.tracker.skeleton_model.connections,
                    output_path=output_path,
                    segment_lengths=self.tracker.skeleton_model.segment_lengths,
                    segment_names=self.tracker.skeleton_model.segment_names,
                    metadata=metadata
                )
                exported_files[f'skeleton_model_{format_name}'] = str(output_path)
                logger.info(f"Модель скелета экспортирована в {output_path}")

        # Экспорт параметров анимации (только в CSV, так как для JSON требуется отдельный формат)
        if export_animation:

            # Определяем, какие 3D-точки использовать
            points_3d = self.results['refined_points_3d']
            test_frame_idx = 65
            print(f"Кадр {test_frame_idx} после заполнения пропусков:")
            print(points_3d[test_frame_idx])

            if points_3d is not None:
                # Создаем калькулятор анимации
                calculator = AnimationCalculator()

                # Вычисляем параметры анимации
                animation_params = calculator.process_sequence(points_3d)

                # Сохраняем результаты в памяти
                self.results['animation_params'] = animation_params

                # Для каждого формата, экспортируем соответствующие файлы анимации
                for format_name, exporter in exporters.items():
                    if format_name == "csv":  # Пока CSV формат для анимации
                        # Экспорт параметров анимации (углы Эйлера)
                        output_path = output_folder / f"animation_euler.{format_name}"

                        # Преобразуем параметры анимации в формат для экспорта
                        euler_data = {}
                        for joint_name, joint_data in self.results['animation_params'].items():
                            euler_angles = joint_data['euler']
                            for i, axis in enumerate(['x', 'y', 'z']):
                                euler_data[f"{joint_name}_{axis}"] = euler_angles[:, i]

                        # Используем экспортер для сохранения
                        exporter.export_joint_angles(
                            joint_angles=euler_data,
                            output_path=output_path,
                            metadata={**metadata, "animation_type": "euler_angles"}
                        )

                        exported_files[f'animation_euler_{format_name}'] = str(output_path)
                        logger.info(f"Углы Эйлера анимации экспортированы в {output_path}")

                        # Экспорт параметров анимации (кватернионы)
                        output_path = output_folder / f"animation_quaternion.{format_name}"

                        # Преобразуем параметры анимации в формат для экспорта
                        quat_data = {}
                        for joint_name, joint_data in self.results['animation_params'].items():
                            quaternions = joint_data['quaternion']
                            for i, component in enumerate(['w', 'x', 'y', 'z']):
                                quat_data[f"{joint_name}_{component}"] = quaternions[:, i]

                        # Используем экспортер для сохранения
                        exporter.export_joint_angles(
                            joint_angles=quat_data,
                            output_path=output_path,
                            metadata={**metadata, "animation_type": "quaternion"}
                        )

                        exported_files[f'animation_quaternion_{format_name}'] = str(output_path)
                        logger.info(f"Кватернионы анимации экспортированы в {output_path}")

                    elif format_name == "json":
                        # Для JSON предлагаем более структурированный формат для анимации
                        output_path = output_folder / f"animation.{format_name}"

                        # Создаем структуру для анимации в JSON-формате
                        animation_data = {
                            "metadata": {
                                **metadata,
                                "animation_format": "json",
                                "frames_count": points_3d.shape[0],
                                "joints_count": len(animation_params)
                            },
                            "frames": []
                        }

                        # Заполняем кадры анимации
                        for frame_idx in range(points_3d.shape[0]):
                            frame_data = {}

                            for joint_name, joint_data in animation_params.items():
                                # Добавляем информацию о суставе, если есть данные для этого кадра
                                joint_info = {}

                                # Позиция
                                if 'position' in joint_data and frame_idx < len(joint_data['position']):
                                    position = joint_data['position'][frame_idx]
                                    if not np.any(np.isnan(position)):
                                        joint_info['position'] = {
                                            'x': float(position[0]),
                                            'y': float(position[1]),
                                            'z': float(position[2])
                                        }

                                # Углы Эйлера
                                if 'euler' in joint_data and frame_idx < len(joint_data['euler']):
                                    euler = joint_data['euler'][frame_idx]
                                    if not np.any(np.isnan(euler)):
                                        joint_info['euler'] = {
                                            'x': float(euler[0]),
                                            'y': float(euler[1]),
                                            'z': float(euler[2])
                                        }

                                # Кватернионы
                                if 'quaternion' in joint_data and frame_idx < len(joint_data['quaternion']):
                                    quat = joint_data['quaternion'][frame_idx]
                                    if not np.any(np.isnan(quat)):
                                        joint_info['quaternion'] = {
                                            'w': float(quat[0]),
                                            'x': float(quat[1]),
                                            'y': float(quat[2]),
                                            'z': float(quat[3])
                                        }

                                # Добавляем информацию о суставе в кадр, если она не пуста
                                if joint_info:
                                    frame_data[joint_name] = joint_info

                            # Добавляем кадр только если в нем есть данные
                            if frame_data:
                                animation_data["frames"].append(frame_data)

                        # Сохраняем в файл
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(animation_data, f, indent=2, ensure_ascii=False)

                        exported_files[f'animation_{format_name}'] = str(output_path)
                        logger.info(f"Анимация экспортирована в {output_path}")
            else:
                logger.warning("Нет 3D-данных для экспорта анимации")

        return exported_files

    def get_results(self) -> Dict[str, Any]:
        """
        Возвращает все результаты обработки.

        Returns:
            Dict[str, Any]: Словарь с результатами обработки
        """
        return self.results


def create_pipeline() -> Pipeline:
    """
    Создает и возвращает экземпляр конвейера обработки.

    Returns:
        Pipeline: Экземпляр конвейера обработки
    """
    return Pipeline()


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем конвейер
    pipeline = create_pipeline()

    # Пути к видеофайлам
    calibration_videos = [
        r"C:\Users\vczyp\openmocap_data\recordings\rec_with_calibr\camera_0.mp4",
        r"C:\Users\vczyp\openmocap_data\recordings\rec_with_calibr\camera_1.mp4"
    ]

    recording_videos = [
        r"C:\Users\vczyp\openmocap_data\recordings\recording_20250322_235323\camera_0.mp4",
        r"C:\Users\vczyp\openmocap_data\recordings\recording_20250322_235323\camera_1.mp4"
    ]

    # Проверяем наличие видеофайлов
    if all(Path(video).exists() for video in calibration_videos) and all(
            Path(video).exists() for video in recording_videos):
        # Выполняем полный конвейер обработки
        pipeline.set_calibrator(MultiCameraCalibrator()) \
            .set_tracker(MediaPipeTracker()) \
            .calibrate_cameras(calibration_videos) \
            .track_videos(recording_videos) \
            .triangulate() \
            .filter_data() \
            .refine_data() \
            .calculate_joint_angles() \
            .export_results(
                output_folder="output_data",
                export_format="json"  # или "both" для экспорта и в CSV, и в JSON
            )

        logger.info("Обработка завершена успешно")
    else:
        logger.error("Не все видеофайлы найдены")