"""
Утилиты для работы с файлами и директориями.
"""

import json
import logging
import os
import shutil
import toml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)


def get_project_root_dir() -> Path:
    """
    Возвращает корневую директорию данных проекта.

    По умолчанию: ~/openmocap_data/

    Returns:
        Path: Путь к корневой директории
    """
    data_dir = os.environ.get("OPENMOCAP_DATA_DIR")

    if data_dir:
        root_dir = Path(data_dir)
    else:
        root_dir = Path.home() / "openmocap_data"

    root_dir.mkdir(parents=True, exist_ok=True)

    return root_dir


def get_sessions_dir() -> Path:
    """
    Возвращает директорию для хранения сессий захвата движения.

    Returns:
        Path: Путь к директории сессий
    """
    sessions_dir = get_project_root_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    return sessions_dir


def get_calibrations_dir() -> Path:
    """
    Возвращает директорию для хранения калибровок камер.

    Returns:
        Path: Путь к директории калибровок
    """
    calibrations_dir = get_project_root_dir() / "calibrations"
    calibrations_dir.mkdir(parents=True, exist_ok=True)

    return calibrations_dir


def create_session_dir(session_name: Optional[str] = None) -> Path:
    """
    Создает директорию для новой сессии захвата движения.

    Args:
        session_name: Название сессии. Если не указано, генерируется автоматически.

    Returns:
        Path: Путь к созданной директории сессии
    """
    if session_name is None:
        session_name = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    session_dir = get_sessions_dir() / session_name

    # Если директория уже существует, добавляем уникальный суффикс
    if session_dir.exists():
        i = 1
        while (get_sessions_dir() / f"{session_name}_{i}").exists():
            i += 1
        session_dir = get_sessions_dir() / f"{session_name}_{i}"

    session_dir.mkdir(parents=True, exist_ok=True)

    # Создаем основные поддиректории
    (session_dir / "videos").mkdir(exist_ok=True)
    (session_dir / "output_data").mkdir(exist_ok=True)

    logger.info(f"Создана директория сессии: {session_dir}")

    return session_dir


def save_json(data: Any, file_path: Union[str, Path], indent: int = 4) -> bool:
    """
    Сохраняет данные в JSON-файл.

    Args:
        data: Данные для сохранения (должны быть сериализуемыми в JSON).
        file_path: Путь для сохранения файла.
        indent: Отступ для форматирования JSON (по умолчанию 4).

    Returns:
        bool: True, если сохранение прошло успешно, иначе False.
    """
    file_path = Path(file_path)

    # Создаем директорию, если она не существует
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"Данные успешно сохранены в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON в {file_path}: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Загружает данные из JSON-файла.

    Args:
        file_path: Путь к JSON-файлу.

    Returns:
        Any: Загруженные данные.

    Raises:
        FileNotFoundError: Если файл не найден.
        json.JSONDecodeError: Если файл содержит некорректный JSON.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"JSON-файл не найден: {file_path}")
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.debug(f"Данные успешно загружены из {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON из {file_path}: {e}")
        raise


def save_toml(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Сохраняет данные в TOML-файл.

    Args:
        data: Данные для сохранения (должны быть сериализуемыми в TOML).
        file_path: Путь для сохранения файла.

    Returns:
        bool: True, если сохранение прошло успешно, иначе False.
    """
    file_path = Path(file_path)

    # Создаем директорию, если она не существует
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

        logger.debug(f"Данные успешно сохранены в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении TOML в {file_path}: {e}")
        return False


def load_toml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Загружает данные из TOML-файла.

    Args:
        file_path: Путь к TOML-файлу.

    Returns:
        Dict[str, Any]: Загруженные данные.

    Raises:
        FileNotFoundError: Если файл не найден.
        toml.TomlDecodeError: Если файл содержит некорректный TOML.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"TOML-файл не найден: {file_path}")
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        data = toml.load(str(file_path))

        logger.debug(f"Данные успешно загружены из {file_path}")
        return data
    except Exception as e:
        logger.error(f"Ошибка декодирования TOML из {file_path}: {e}")
        raise


def save_numpy_array(array: Any, file_path: Union[str, Path]) -> bool:
    """
    Сохраняет массив NumPy в файл.

    Args:
        array: Массив NumPy.
        file_path: Путь для сохранения файла.

    Returns:
        bool: True, если сохранение прошло успешно, иначе False.
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("Библиотека NumPy не установлена")
        return False

    file_path = Path(file_path)

    # Создаем директорию, если она не существует
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(file_path, array)

        logger.debug(f"Массив NumPy успешно сохранен в {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении массива NumPy в {file_path}: {e}")
        return False


def load_numpy_array(file_path: Union[str, Path]) -> Any:
    """
    Загружает массив NumPy из файла.

    Args:
        file_path: Путь к файлу NumPy.

    Returns:
        Any: Загруженный массив.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат файла некорректен.
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("Библиотека NumPy не установлена")
        raise ImportError("Библиотека NumPy не установлена")

    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"Файл NumPy не найден: {file_path}")
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        array = np.load(file_path, allow_pickle=True)

        logger.debug(f"Массив NumPy успешно загружен из {file_path}")
        return array
    except Exception as e:
        logger.error(f"Ошибка при загрузке массива NumPy из {file_path}: {e}")
        raise


def list_sessions() -> List[Dict[str, Any]]:
    """
    Возвращает список доступных сессий.

    Returns:
        List[Dict[str, Any]]: Список сессий с метаданными.
    """
    sessions_dir = get_sessions_dir()
    sessions = []

    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            # Проверяем наличие обязательных поддиректорий
            has_videos = (session_dir / "videos").exists()
            has_output = (session_dir / "output_data").exists()

            # Ищем метаданные сессии
            metadata_path = session_dir / "session_info.json"
            metadata = {}

            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить метаданные сессии {session_dir.name}: {e}")

            # Добавляем информацию о сессии в список
            sessions.append({
                "name": session_dir.name,
                "path": str(session_dir),
                "has_videos": has_videos,
                "has_output": has_output,
                "created": metadata.get("created", None),
                "metadata": metadata
            })

    # Сортируем по времени создания (если доступно) или по имени
    sessions.sort(key=lambda s: s.get("created") or s["name"], reverse=True)

    return sessions


def remove_empty_directories(root_dir: Union[str, Path]) -> int:
    """
    Рекурсивно удаляет пустые директории.

    Args:
        root_dir: Корневая директория для начала поиска.

    Returns:
        int: Количество удаленных директорий.
    """
    root_dir = Path(root_dir)

    if not root_dir.exists() or not root_dir.is_dir():
        logger.warning(f"Директория {root_dir} не существует или не является директорией")
        return 0

    count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                Path(dirpath).rmdir()
                logger.debug(f"Удалена пустая директория: {dirpath}")
                count += 1
            except OSError as e:
                logger.warning(f"Не удалось удалить директорию {dirpath}: {e}")

    logger.info(f"Удалено {count} пустых директорий в {root_dir}")
    return count


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовую сессию
    test_session_dir = create_session_dir("test_session")

    # Сохраняем тестовые данные
    test_data = {
        "created": datetime.now().isoformat(),
        "description": "Тестовая сессия",
        "cameras": ["camera1", "camera2"],
        "settings": {
            "resolution": [1920, 1080],
            "fps": 30
        }
    }

    save_json(test_data, test_session_dir / "session_info.json")

    # Выводим список сессий
    print("Доступные сессии:")
    sessions = list_sessions()
    for session in sessions:
        print(f" - {session['name']} (путь: {session['path']})")

    # Удаляем пустые директории
    removed = remove_empty_directories(get_project_root_dir())
    print(f"Удалено пустых директорий: {removed}")