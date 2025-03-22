"""
Модуль для управления сессией записи в OpenMoCap.

Предоставляет класс Session для создания и управления сессией записи,
включая хранение метаданных и результатов.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from openmocap.utils.file_utils import create_session_dir, save_json
from openmocap.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Session:
    """
    Класс для управления сессией записи в OpenMoCap.

    Attributes:
        name (str): Имя сессии
        session_id (str): Уникальный идентификатор сессии
        session_dir (Path): Путь к директории сессии
        metadata (Dict): Метаданные сессии
        pipeline (Pipeline): Конвейер обработки данных
    """

    def __init__(
            self,
            name: Optional[str] = None,
            session_dir: Optional[Union[str, Path]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует объект сессии.

        Args:
            name: Имя сессии. Если не указано, генерируется автоматически.
            session_dir: Путь к директории сессии. Если не указано, создается автоматически.
            metadata: Дополнительные метаданные сессии.
        """
        # Генерируем имя, если не указано
        self.name = name or f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Генерируем уникальный идентификатор сессии
        self.session_id = str(uuid.uuid4())

        # Создаем или используем указанную директорию сессии
        if session_dir is None:
            self.session_dir = create_session_dir(self.name)
        else:
            self.session_dir = Path(session_dir)
            self.session_dir.mkdir(parents=True, exist_ok=True)

        # Инициализируем метаданные
        self.metadata = metadata or {}
        self.metadata.update({
            'name': self.name,
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'session_dir': str(self.session_dir)
        })

        # Создаем конвейер обработки
        self.pipeline = Pipeline()

        # Сохраняем метаданные сессии
        self.save_metadata()

        logger.info(f"Создана сессия {self.name} (ID: {self.session_id})")
        logger.info(f"Путь к директории сессии: {self.session_dir}")

    def save_metadata(self) -> str:
        """
        Сохраняет метаданные сессии в файл.

        Returns:
            str: Путь к сохраненному файлу
        """
        metadata_path = self.session_dir / "session_info.json"
        save_json(self.metadata, metadata_path)
        logger.debug(f"Метаданные сессии сохранены в {metadata_path}")
        return str(metadata_path)

    def update_metadata(self, **kwargs) -> None:
        """
        Обновляет метаданные сессии.

        Args:
            **kwargs: Ключи и значения для обновления метаданных
        """
        self.metadata.update(kwargs)
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.save_metadata()
        logger.debug("Метаданные сессии обновлены")

    def get_pipeline(self) -> Pipeline:
        """
        Возвращает конвейер обработки данных.

        Returns:
            Pipeline: Конвейер обработки данных
        """
        return self.pipeline

    def get_calibration_videos_dir(self) -> Path:
        """
        Возвращает путь к директории с видео калибровки.

        Returns:
            Path: Путь к директории
        """
        calibration_dir = self.session_dir / "calibration_videos"
        calibration_dir.mkdir(exist_ok=True)
        return calibration_dir

    def get_recording_videos_dir(self) -> Path:
        """
        Возвращает путь к директории с видео записи.

        Returns:
            Path: Путь к директории
        """
        recording_dir = self.session_dir / "recording_videos"
        recording_dir.mkdir(exist_ok=True)
        return recording_dir

    def get_output_dir(self) -> Path:
        """
        Возвращает путь к директории с результатами обработки.

        Returns:
            Path: Путь к директории
        """
        output_dir = self.session_dir / "output_data"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def export_results(self, **kwargs) -> Dict[str, str]:
        """
        Экспортирует результаты обработки.

        Args:
            **kwargs: Аргументы, передаваемые в метод export_results конвейера

        Returns:
            Dict[str, str]: Словарь с путями к экспортированным файлам

        Raises:
            ValueError: Если в конвейере нет результатов для экспорта
        """
        output_dir = self.get_output_dir()
        exported_files = self.pipeline.export_results(output_dir, **kwargs)

        # Обновляем метаданные
        self.update_metadata(
            export_timestamp=datetime.now().isoformat(),
            exported_files=exported_files
        )

        return exported_files

    @classmethod
    def load(cls, session_dir: Union[str, Path]) -> 'Session':
        """
        Загружает сессию из директории.

        Args:
            session_dir: Путь к директории сессии

        Returns:
            Session: Загруженная сессия

        Raises:
            FileNotFoundError: Если директория не существует или не содержит метаданные сессии
        """
        session_dir = Path(session_dir)

        if not session_dir.exists():
            raise FileNotFoundError(f"Директория сессии не найдена: {session_dir}")

        metadata_path = session_dir / "session_info.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных сессии не найден: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Создаем сессию с загруженными метаданными
        session = cls(
            name=metadata.get('name'),
            session_dir=session_dir,
            metadata=metadata
        )

        logger.info(f"Загружена сессия {session.name} (ID: {session.session_id})")

        return session


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем новую сессию
    session = Session("test_session")

    # Получаем директории
    calibration_dir = session.get_calibration_videos_dir()
    recording_dir = session.get_recording_videos_dir()
    output_dir = session.get_output_dir()

    print(f"Директория калибровки: {calibration_dir}")
    print(f"Директория записи: {recording_dir}")
    print(f"Директория результатов: {output_dir}")

    # Пример обновления метаданных
    session.update_metadata(
        participant="Тестовый участник",
        description="Тестовая сессия для проверки OpenMoCap"
    )

    # Пример загрузки существующей сессии
    try:
        loaded_session = Session.load(session.session_dir)
        print(f"Загружена сессия: {loaded_session.name}")
        print(f"Метаданные: {loaded_session.metadata}")
    except FileNotFoundError as e:
        print(f"Ошибка загрузки сессии: {e}")