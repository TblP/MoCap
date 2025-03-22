"""OpenMoCap - Открытая система захвата движений без маркеров"""

__author__ = "OpenMoCap Team"
__email__ = "info@openmocap.org"
__version__ = "0.1.0"
__description__ = "Открытая система захвата движения без маркеров"

__package_name__ = "openmocap"
__repo_url__ = f"https://github.com/yourorganization/{__package_name__}"
__repo_issues_url__ = f"{__repo_url__}/issues"

import logging
from .utils.logger import configure_logging, LogLevel

configure_logging(LogLevel.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Инициализация пакета {__package_name__}, версия: {__version__}")