#!/usr/bin/env python
"""
Интерфейс командной строки для OpenMoCap.

Предоставляет команды для удобного использования основных функций системы
захвата движения, таких как калибровка камер, запись, обработка и экспорт данных.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

from openmocap.utils.logger import configure_logging, LogLevel
from openmocap.core.session import Session
from openmocap.calibration.multi_camera_calibrator import MultiCameraCalibrator
from openmocap.tracking.mediapipe_tracker import MediaPipeTracker
from openmocap.utils.video_utils import get_video_paths

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """
    Настраивает и возвращает парсер аргументов командной строки.

    Returns:
        argparse.ArgumentParser: Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="OpenMoCap - Открытая система захвата движения без маркеров",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Создание новой сессии
  openmocap create-session --name "моя_сессия"

  # Калибровка камер
  openmocap calibrate --session "моя_сессия" --videos "/путь/к/видео/*.mp4"

  # Обработка видео
  openmocap process --session "моя_сессия" --videos "/путь/к/видео/*.mp4"

  # Экспорт результатов
  openmocap export --session "моя_сессия" --format csv
        """
    )

    # Настройка общих параметров
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Уровень логирования (по умолчанию: info)"
    )

    # Создание подпарсеров для разных команд
    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")

    # Команда create-session
    create_session_parser = subparsers.add_parser(
        "create-session",
        help="Создать новую сессию захвата движения"
    )
    create_session_parser.add_argument(
        "--name",
        type=str,
        help="Имя сессии"
    )
    create_session_parser.add_argument(
        "--dir",
        type=str,
        help="Путь к директории для сессии"
    )
    create_session_parser.add_argument(
        "--description",
        type=str,
        help="Описание сессии"
    )

    # Команда calibrate
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Выполнить калибровку камер"
    )
    calibrate_parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Имя или путь к директории сессии"
    )
    calibrate_parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="Путь к видеофайлам для калибровки (можно использовать шаблоны)"
    )
    calibrate_parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Максимальное количество кадров для обработки"
    )
    calibrate_parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Шаг между обрабатываемыми кадрами"
    )
    calibrate_parser.add_argument(
        "--min-common-frames",
        type=int,
        default=5,
        help="Минимальное количество кадров, где доска видна всеми камерами"
    )
    calibrate_parser.add_argument(
        "--save-to",
        type=str,
        help="Путь для сохранения файла калибровки (по умолчанию сохраняется в директорию сессии)"
    )

    # Команда process
    process_parser = subparsers.add_parser(
        "process",
        help="Обработать видео захвата движения"
    )
    process_parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Имя или путь к директории сессии"
    )
    process_parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="Путь к видеофайлам для обработки (можно использовать шаблоны)"
    )
    process_parser.add_argument(
        "--calibration",
        type=str,
        help="Путь к файлу калибровки"
    )
    process_parser.add_argument(
        "--tracker",
        type=str,
        choices=["mediapipe"],
        default="mediapipe",
        help="Трекер для отслеживания точек (по умолчанию: mediapipe)"
    )
    process_parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Начальный кадр для обработки"
    )
    process_parser.add_argument(
        "--end-frame",
        type=int,
        help="Конечный кадр для обработки"
    )
    process_parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Шаг между обрабатываемыми кадрами"
    )
    process_parser.add_argument(
        "--filter",
        type=str,
        choices=["butterworth", "savgol", "kalman", "median", "none"],
        default="butterworth",
        help="Фильтр для сглаживания данных (по умолчанию: butterworth)"
    )
    process_parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Частота среза для фильтра Баттерворта (по умолчанию: 6.0 Гц)"
    )
    process_parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Заполнять пропуски в данных"
    )
    process_parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="Максимальный размер пропуска для заполнения (по умолчанию: 10 кадров)"
    )
    process_parser.add_argument(
        "--enforce-limits",
        action="store_true",
        help="Применять ограничения жесткого тела"
    )

    # Команда export
    export_parser = subparsers.add_parser(
        "export",
        help="Экспортировать результаты обработки"
    )
    export_parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Имя или путь к директории сессии"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "all"],
        default="csv",
        help="Формат для экспорта (по умолчанию: csv)"
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        help="Директория для сохранения экспортированных файлов"
    )
    export_parser.add_argument(
        "--export-2d",
        action="store_true",
        help="Экспортировать 2D-координаты"
    )
    export_parser.add_argument(
        "--export-3d",
        action="store_true",
        help="Экспортировать 3D-координаты"
    )
    export_parser.add_argument(
        "--export-angles",
        action="store_true",
        help="Экспортировать углы суставов"
    )
    export_parser.add_argument(
        "--export-all",
        action="store_true",
        help="Экспортировать все доступные данные"
    )

    # Команда info
    info_parser = subparsers.add_parser(
        "info",
        help="Показать информацию о сессии"
    )
    info_parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Имя или путь к директории сессии"
    )
    info_parser.add_argument(
        "--list-files",
        action="store_true",
        help="Показать список файлов в сессии"
    )

    return parser


def find_session_dir(session_name_or_path: str) -> Path:
    """
    Находит директорию сессии по имени или пути.

    Args:
        session_name_or_path: Имя сессии или путь к директории сессии

    Returns:
        Path: Путь к директории сессии

    Raises:
        FileNotFoundError: Если сессия не найдена
    """
    from openmocap.utils.file_utils import get_sessions_dir

    # Сначала проверяем, является ли это путем
    path = Path(session_name_or_path)
    if path.exists() and path.is_dir():
        return path

    # Если это имя сессии, ищем в стандартной директории
    sessions_dir = get_sessions_dir()
    session_path = sessions_dir / session_name_or_path

    if session_path.exists() and session_path.is_dir():
        return session_path

    # Проверяем, существует ли сессия с таким именем в стандартной директории
    # (проверяем все поддиректории)
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith(session_name_or_path):
            return session_dir

    raise FileNotFoundError(f"Сессия '{session_name_or_path}' не найдена")


def create_session_command(args: argparse.Namespace) -> int:
    """
    Обрабатывает команду create-session.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    try:
        session = Session(
            name=args.name,
            session_dir=args.dir,
            metadata={"description": args.description} if args.description else None
        )
        print(f"Создана сессия: {session.name}")
        print(f"ID сессии: {session.session_id}")
        print(f"Путь к директории сессии: {session.session_dir}")
        return 0
    except Exception as e:
        logger.error(f"Ошибка при создании сессии: {e}")
        return 1


def calibrate_command(args: argparse.Namespace) -> int:
    """
    Обрабатывает команду calibrate.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    try:
        # Находим директорию сессии
        session_dir = find_session_dir(args.session)

        # Загружаем сессию
        session = Session.load(session_dir)

        # Получаем пути к видеофайлам
        import glob
        video_paths = sorted(glob.glob(args.videos))

        if not video_paths:
            logger.error(f"Не найдены видеофайлы по пути: {args.videos}")
            return 1

        logger.info(f"Найдено {len(video_paths)} видеофайлов для калибровки")
        for i, path in enumerate(video_paths):
            logger.info(f"  {i + 1}. {path}")

        # Создаем калибратор
        calibrator = MultiCameraCalibrator()

        # Калибруем камеры
        logger.info("Начинаю калибровку камер...")
        try:
            calibration_data = calibrator.calibrate_cameras_stereo(
                video_paths=video_paths,
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                min_common_frames=args.min_common_frames
            )

            # Определяем путь для сохранения калибровки
            if args.save_to:
                save_path = args.save_to
            else:
                # Сохраняем в директорию сессии
                save_path = session.session_dir / f"{session.name}_calibration.toml"

            # Сохраняем калибровку
            calibration_file = calibrator.save_calibration(save_path, format="toml")

            # Обновляем метаданные сессии
            session.update_metadata(
                calibration_file=calibration_file,
                num_cameras=len(video_paths),
                calibration_timestamp=calibration_data['metadata']['calibration_time']
            )

            logger.info(f"Калибровка успешно завершена и сохранена в {calibration_file}")
            return 0
        except Exception as e:
            logger.error(f"Ошибка при калибровке камер: {e}")
            return 1
    except Exception as e:
        logger.error(f"Ошибка в команде calibrate: {e}")
        return 1


def process_command(args: argparse.Namespace) -> int:
    """
    Обрабатывает команду process.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    try:
        # Находим директорию сессии
        session_dir = find_session_dir(args.session)

        # Загружаем сессию
        session = Session.load(session_dir)

        # Получаем пути к видеофайлам
        import glob
        video_paths = sorted(glob.glob(args.videos))

        if not video_paths:
            logger.error(f"Не найдены видеофайлы по пути: {args.videos}")
            return 1

        logger.info(f"Найдено {len(video_paths)} видеофайлов для обработки")
        for i, path in enumerate(video_paths):
            logger.info(f"  {i + 1}. {path}")

        # Получаем путь к файлу калибровки
        calibration_file = args.calibration
        if not calibration_file:
            # Ищем файл калибровки в метаданных сессии
            calibration_file = session.metadata.get('calibration_file')
            if not calibration_file:
                logger.error("Не указан файл калибровки. Используйте --calibration или сначала выполните калибровку.")
                return 1

        # Загружаем калибровку
        calibrator = MultiCameraCalibrator.from_calibration_file(calibration_file)

        # Создаем трекер
        if args.tracker.lower() == "mediapipe":
            tracker = MediaPipeTracker()
        else:
            logger.error(f"Неизвестный трекер: {args.tracker}")
            return 1

        # Настраиваем конвейер обработки
        pipeline = session.get_pipeline()
        pipeline.set_calibrator(calibrator)
        pipeline.set_tracker(tracker)

        # Выполняем обработку
        logger.info("Начинаю отслеживание точек...")
        pipeline.track_videos(
            video_paths=video_paths,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step
        )

        logger.info("Выполняю триангуляцию 3D-точек...")
        pipeline.triangulate()

        # Фильтрация и обработка данных
        if args.filter.lower() != "none":
            logger.info(f"Применяю фильтр: {args.filter}")

            filter_params = {}
            if args.filter.lower() == "butterworth":
                filter_params["cutoff"] = args.cutoff

            pipeline.filter_data(threshold_method="absolute", threshold_value=args.cutoff)

        # Уточнение данных
        pipeline.refine_data(method="levenberg_marquardt")

        # Вычисление углов суставов
        pipeline.calculate_joint_angles()

        # Обновляем метаданные сессии
        session.update_metadata(
            processing_timestamp=session.metadata.get('updated_at'),
            num_frames=len(pipeline.results.get('points_3d', [])),
            filter_applied=args.filter.lower() if args.filter.lower() != "none" else None,
            gaps_filled=args.fill_gaps,
            max_gap=args.max_gap if args.fill_gaps else None,
            constraints_applied=args.enforce_limits
        )

        logger.info("Обработка успешно завершена")
        return 0
    except Exception as e:
        logger.error(f"Ошибка в команде process: {e}")
        return 1


def export_command(args: argparse.Namespace) -> int:
    """
    Обрабатывает команду export.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    try:
        # Находим директорию сессии
        session_dir = find_session_dir(args.session)

        # Загружаем сессию
        session = Session.load(session_dir)

        # Определяем директорию для экспорта
        output_dir = args.output_dir
        if not output_dir:
            output_dir = session.get_output_dir()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Определяем, что экспортировать
        export_2d = args.export_2d or args.export_all
        export_3d = args.export_3d or args.export_all
        export_filtered = args.export_all
        export_refined = args.export_all
        export_angles = args.export_angles or args.export_all
        export_skeleton = args.export_all

        # Экспортируем результаты
        exported_files = session.export_results(
            export_2d=export_2d,
            export_3d=export_3d,
            export_filtered=export_filtered,
            export_refined=export_refined,
            export_angles=export_angles,
            export_skeleton=export_skeleton
        )

        logger.info(f"Результаты экспортированы в {output_dir}")

        # Выводим список экспортированных файлов
        for file_type, file_path in exported_files.items():
            logger.info(f"  {file_type}: {file_path}")

        return 0
    except Exception as e:
        logger.error(f"Ошибка в команде export: {e}")
        return 1


def info_command(args: argparse.Namespace) -> int:
    """
    Обрабатывает команду info.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    try:
        # Находим директорию сессии
        session_dir = find_session_dir(args.session)

        # Загружаем сессию
        session = Session.load(session_dir)

        # Выводим информацию о сессии
        print(f"Информация о сессии:")
        print(f"  Имя: {session.name}")
        print(f"  ID: {session.session_id}")
        print(f"  Директория: {session.session_dir}")
        print(f"  Создана: {session.metadata.get('created_at')}")

        if 'updated_at' in session.metadata:
            print(f"  Последнее обновление: {session.metadata.get('updated_at')}")

        if 'description' in session.metadata:
            print(f"  Описание: {session.metadata.get('description')}")

        if 'calibration_file' in session.metadata:
            print(f"  Файл калибровки: {session.metadata.get('calibration_file')}")

        if 'num_cameras' in session.metadata:
            print(f"  Количество камер: {session.metadata.get('num_cameras')}")

        if 'processing_timestamp' in session.metadata:
            print(f"  Дата обработки: {session.metadata.get('processing_timestamp')}")

        if 'export_timestamp' in session.metadata:
            print(f"  Дата экспорта: {session.metadata.get('export_timestamp')}")

        # Выводим список файлов, если запрошено
        if args.list_files:
            print("\nФайлы в директории сессии:")
            for path in sorted(session.session_dir.glob('**/*')):
                if path.is_file():
                    rel_path = path.relative_to(session.session_dir)
                    print(f"  {rel_path}")

        return 0
    except Exception as e:
        logger.error(f"Ошибка в команде info: {e}")
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Основная функция программы.

    Args:
        args: Аргументы командной строки (если None, используется sys.argv)

    Returns:
        int: Код возврата (0 - успех, не 0 - ошибка)
    """
    # Парсим аргументы командной строки
    parser = setup_parser()
    parsed_args = parser.parse_args(args)

    # Если не указана команда, выводим помощь и выходим
    if not parsed_args.command:
        parser.print_help()
        return 0

    # Настраиваем логирование
    log_level_map = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL
    }

    log_level = log_level_map.get(parsed_args.log_level, LogLevel.INFO)
    if parsed_args.verbose:
        log_level = LogLevel.DEBUG

    configure_logging(log_level)

    # Выполняем соответствующую команду
    if parsed_args.command == "create-session":
        return create_session_command(parsed_args)
    elif parsed_args.command == "calibrate":
        return calibrate_command(parsed_args)
    elif parsed_args.command == "process":
        return process_command(parsed_args)
    elif parsed_args.command == "export":
        return export_command(parsed_args)
    elif parsed_args.command == "info":
        return info_command(parsed_args)
    else:
        logger.error(f"Неизвестная команда: {parsed_args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())