"""
Модуль настройки логирования для OpenMoCap
"""

import enum
import logging
import os
import sys
from pathlib import Path


class LogLevel(enum.Enum):
    """Перечисление уровней логирования"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    TRACE = 5  # Пользовательский уровень ниже DEBUG


def get_log_file_path(base_path: str = None) -> Path:
    """
    Возвращает путь к файлу лога

    Args:
        base_path: Базовый путь для хранения лога. Если не указан,
                  используется домашняя директория пользователя.

    Returns:
        Path: Путь к файлу лога
    """
    if base_path is None:
        base_path = os.path.join(Path.home(), "openmocap_data", "logs")

    log_dir = Path(base_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir / "openmocap.log"


def configure_logging(log_level: LogLevel = LogLevel.INFO, log_to_file: bool = True) -> None:
    """
    Настраивает систему логирования

    Args:
        log_level: Уровень логирования
        log_to_file: Флаг, указывающий, нужно ли записывать логи в файл
    """
    # Добавляем пользовательский уровень TRACE
    logging.addLevelName(LogLevel.TRACE.value, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(LogLevel.TRACE.value):
            self._log(LogLevel.TRACE.value, message, args, **kwargs)

    logging.Logger.trace = trace

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)

    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Формат сообщений
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Обработчик для файла, если нужно
    if log_to_file:
        log_file_path = get_log_file_path()
        file_handler = logging.FileHandler(str(log_file_path), encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Устанавливаем уровень логирования для сторонних библиотек
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("chardet.charsetprober").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Пример использования
    configure_logging(LogLevel.DEBUG)
    logger = logging.getLogger(__name__)

    logger.trace("Это сообщение уровня TRACE")
    logger.debug("Это сообщение уровня DEBUG")
    logger.info("Это сообщение уровня INFO")
    logger.warning("Это сообщение уровня WARNING")
    logger.error("Это сообщение уровня ERROR")
    logger.critical("Это сообщение уровня CRITICAL")