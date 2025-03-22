"""
Утилиты для работы с видео файлами.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Iterator

logger = logging.getLogger(__name__)


def get_video_paths(directory: Union[str, Path]) -> List[Path]:
    """
    Находит все видеофайлы в указанной директории.

    Args:
        directory: Путь к директории с видеофайлами.

    Returns:
        List[Path]: Список путей к видеофайлам (.mp4, .avi, .mov)
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        logger.error(f"Директория {directory} не существует или не является директорией")
        return []

    # Ищем видеофайлы с различными расширениями
    video_files = []
    for ext in [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV"]:
        video_files.extend(directory.glob(f"*{ext}"))

    # Удаляем дубликаты, сохраняя порядок
    unique_video_files = []
    for file in video_files:
        if file not in unique_video_files:
            unique_video_files.append(file)

    logger.info(f"Найдено {len(unique_video_files)} видеофайлов в {directory}")
    return unique_video_files


def get_video_properties(video_path: Union[str, Path]) -> Dict[str, Union[int, float]]:
    """
    Получает свойства видеофайла (разрешение, FPS, количество кадров).

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        Dict: Словарь со свойствами видео (width, height, fps, frame_count)

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если файл не является валидным видео.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Видеофайл {video_path} не найден")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл {video_path}")

    properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    cap.release()

    logger.debug(f"Свойства видео {video_path}: {properties}")
    return properties


def get_frame_count(video_path: Union[str, Path]) -> int:
    """
    Получает количество кадров в видеофайле.

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        int: Количество кадров
    """
    return get_video_properties(video_path)["frame_count"]


def video_frame_generator(video_path: Union[str, Path],
                          start_frame: int = 0,
                          end_frame: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Генератор кадров из видеофайла.

    Args:
        video_path: Путь к видеофайлу.
        start_frame: Номер кадра, с которого начать (по умолчанию 0).
        end_frame: Номер кадра, на котором закончить (по умолчанию None - до конца).

    Yields:
        Tuple[int, np.ndarray]: Кортеж (номер кадра, изображение в формате numpy array)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Видеофайл {video_path} не найден")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = frame_count

    # Валидация параметров
    start_frame = max(0, min(start_frame, frame_count - 1))
    end_frame = max(start_frame + 1, min(end_frame, frame_count))

    logger.debug(f"Чтение кадров из {video_path} с {start_frame} по {end_frame}")

    # Перейти к начальному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Не удалось прочитать кадр {frame_idx} из {video_path}")
            break

        yield frame_idx, frame
        frame_idx += 1

    cap.release()


def save_frame(frame: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Сохраняет кадр в файл.

    Args:
        frame: Кадр в формате numpy array.
        output_path: Путь для сохранения кадра.

    Returns:
        bool: True, если сохранение прошло успешно, иначе False.
    """
    output_path = Path(output_path)

    # Создаем директорию, если она не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cv2.imwrite(str(output_path), frame)
        logger.debug(f"Кадр сохранен в {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении кадра в {output_path}: {e}")
        return False


def extract_frames(video_path: Union[str, Path],
                   output_dir: Union[str, Path],
                   frame_step: int = 1,
                   start_frame: int = 0,
                   end_frame: Optional[int] = None) -> int:
    """
    Извлекает кадры из видеофайла и сохраняет их в указанную директорию.

    Args:
        video_path: Путь к видеофайлу.
        output_dir: Директория для сохранения кадров.
        frame_step: Шаг извлечения кадров (по умолчанию 1 - каждый кадр).
        start_frame: Начальный кадр (по умолчанию 0).
        end_frame: Конечный кадр (по умолчанию None - до конца видео).

    Returns:
        int: Количество извлеченных кадров.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_idx, frame in video_frame_generator(video_path, start_frame, end_frame):
        if frame_idx % frame_step == 0:
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            if save_frame(frame, frame_path):
                count += 1

    logger.info(f"Извлечено {count} кадров из {video_path} в {output_dir}")
    return count


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    test_dir = "test_videos"
    Path(test_dir).mkdir(exist_ok=True)

    # Искусственный пример - здесь нужно будет иметь видеофайлы для тестирования
    videos = get_video_paths(test_dir)
    print(f"Найдены видео: {videos}")

    # Если есть видеофайлы, можно их обработать
    if videos:
        video = videos[0]
        properties = get_video_properties(video)
        print(f"Свойства видео: {properties}")

        # Пример извлечения кадров
        extract_frames(video, "extracted_frames", frame_step=30)  # Каждую секунду при 30 FPS