"""
Модуль для сглаживания данных скелета.

Содержит функции для фильтрации данных о движении, включая фильтр Баттерворта,
савицкого-голея и экспоненциальное сглаживание.
"""

import logging
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)


class LowPassFilter:
    """
    Базовый класс для фильтров нижних частот.

    Attributes:
        fs (float): Частота дискретизации данных (частота кадров, Гц)
    """

    def __init__(self, fs: float = 30.0):
        """
        Инициализирует базовый фильтр нижних частот.

        Args:
            fs: Частота дискретизации данных (частота кадров, Гц)
        """
        self.fs = fs
        logger.debug(f"Инициализирован базовый фильтр нижних частот с fs={fs} Гц")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет фильтр к данным.

        Args:
            data: Входные данные для фильтрации

        Returns:
            np.ndarray: Отфильтрованные данные
        """
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")


class ButterworthFilter(LowPassFilter):
    """
    Фильтр Баттерворта для сглаживания данных.

    Attributes:
        fs (float): Частота дискретизации данных (частота кадров, Гц)
        cutoff (float): Частота среза фильтра (Гц)
        order (int): Порядок фильтра
    """

    def __init__(self, fs: float = 30.0, cutoff: float = 6.0, order: int = 4):
        """
        Инициализирует фильтр Баттерворта.

        Args:
            fs: Частота дискретизации данных (частота кадров, Гц)
            cutoff: Частота среза фильтра (Гц)
            order: Порядок фильтра
        """
        super().__init__(fs)
        self.cutoff = cutoff
        self.order = order

        # Проверка параметров
        if cutoff >= fs / 2:
            logger.warning(
                f"Частота среза ({cutoff} Гц) должна быть меньше половины частоты дискретизации ({fs / 2} Гц). Снижаем до {fs / 2 - 0.1} Гц.")
            self.cutoff = fs / 2 - 0.1

        # Нормализованная частота среза (частота среза / частота Найквиста)
        self.nyquist = 0.5 * fs
        self.normal_cutoff = self.cutoff / self.nyquist

        # Расчет коэффициентов фильтра
        self.b, self.a = signal.butter(self.order, self.normal_cutoff, btype='low', analog=False)

        logger.info(f"Инициализирован фильтр Баттерворта с частотой среза {self.cutoff} Гц, порядком {order}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет фильтр Баттерворта к данным.

        Args:
            data: Входные данные для фильтрации.
                 Может иметь форму (n_points, n_dims) или (n_frames, n_points, n_dims)

        Returns:
            np.ndarray: Отфильтрованные данные той же формы
        """
        # Проверка размерности входных данных
        original_shape = data.shape
        is_3d = len(original_shape) == 3

        # Если данные имеют 3 измерения (кадры, точки, координаты)
        if is_3d:
            n_frames, n_points, n_dims = original_shape
            # Преобразуем в 2D формат для фильтрации
            data_reshaped = data.reshape(n_frames, -1)
        else:
            # Данные уже в 2D формате (точки, координаты)
            data_reshaped = data

        # Создаем массив для отфильтрованных данных
        filtered_data = np.copy(data_reshaped)

        # Обрабатываем каждую колонку отдельно
        for col in range(data_reshaped.shape[1]):
            # Извлекаем колонку данных
            col_data = data_reshaped[:, col]

            # Находим маску пропущенных значений
            nan_mask = np.isnan(col_data)

            # Если все значения NaN, пропускаем колонку
            if np.all(nan_mask):
                continue

            # Если есть пропущенные значения, линейно интерполируем их для фильтрации
            if np.any(nan_mask):
                valid_indices = np.where(~nan_mask)[0]

                # Если недостаточно валидных точек, пропускаем колонку
                if len(valid_indices) < 4:  # Минимум точек для фильтра 4 порядка
                    continue

                # Создаем временный массив с интерполяцией пропущенных значений
                temp_data = np.copy(col_data)
                interp_indices = np.arange(len(col_data))
                temp_data[nan_mask] = np.interp(
                    interp_indices[nan_mask],
                    interp_indices[~nan_mask],
                    col_data[~nan_mask]
                )

                # Применяем фильтр к временным данным
                filtered_col = signal.filtfilt(self.b, self.a, temp_data)

                # Сохраняем только валидные значения обратно
                filtered_data[:, col] = np.where(nan_mask, np.nan, filtered_col)
            else:
                # Применяем фильтр к полным данным
                filtered_data[:, col] = signal.filtfilt(self.b, self.a, col_data)

        # Преобразуем обратно в исходную форму
        if is_3d:
            filtered_data = filtered_data.reshape(original_shape)

        logger.debug(f"Фильтр Баттерворта применен к данным формы {original_shape}")

        return filtered_data


class SavitzkyGolayFilter(LowPassFilter):
    """
    Фильтр Савицкого-Голея для сглаживания данных.

    Attributes:
        window_length (int): Длина окна (должна быть нечетной)
        polyorder (int): Порядок полинома
    """

    def __init__(self, window_length: int = 15, polyorder: int = 3, fs: float = 30.0):
        """
        Инициализирует фильтр Савицкого-Голея.

        Args:
            window_length: Длина окна (должна быть нечетной)
            polyorder: Порядок полинома
            fs: Частота дискретизации данных (частота кадров, Гц)
        """
        super().__init__(fs)

        # Убедимся, что длина окна нечетная
        if window_length % 2 == 0:
            window_length += 1
            logger.warning(f"Длина окна должна быть нечетной. Увеличена до {window_length}")

        # Убедимся, что порядок полинома меньше длины окна
        if polyorder >= window_length:
            polyorder = window_length - 1
            logger.warning(f"Порядок полинома должен быть меньше длины окна. Снижен до {polyorder}")

        self.window_length = window_length
        self.polyorder = polyorder

        logger.info(f"Инициализирован фильтр Савицкого-Голея с длиной окна {window_length}, порядком {polyorder}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет фильтр Савицкого-Голея к данным.

        Args:
            data: Входные данные для фильтрации.
                 Может иметь форму (n_points, n_dims) или (n_frames, n_points, n_dims)

        Returns:
            np.ndarray: Отфильтрованные данные той же формы
        """
        # Проверка размерности входных данных
        original_shape = data.shape
        is_3d = len(original_shape) == 3

        # Если данные имеют 3 измерения (кадры, точки, координаты)
        if is_3d:
            n_frames, n_points, n_dims = original_shape
            # Преобразуем в 2D формат для фильтрации
            data_reshaped = data.reshape(n_frames, -1)
        else:
            # Данные уже в 2D формате (точки, координаты)
            data_reshaped = data

        # Создаем массив для отфильтрованных данных
        filtered_data = np.copy(data_reshaped)

        # Если длина окна больше количества кадров, уменьшаем её
        window_length = min(self.window_length, data_reshaped.shape[0])
        if window_length % 2 == 0:
            window_length -= 1

        # Если длина окна менее 3, фильтрация невозможна
        if window_length < 3:
            logger.warning(f"Недостаточно данных для фильтрации. Возвращаем исходные данные.")
            return data

        # Обрабатываем каждую колонку отдельно
        for col in range(data_reshaped.shape[1]):
            # Извлекаем колонку данных
            col_data = data_reshaped[:, col]

            # Находим маску пропущенных значений
            nan_mask = np.isnan(col_data)

            # Если все значения NaN, пропускаем колонку
            if np.all(nan_mask):
                continue

            # Если есть пропущенные значения, линейно интерполируем их для фильтрации
            if np.any(nan_mask):
                valid_indices = np.where(~nan_mask)[0]

                # Если недостаточно валидных точек, пропускаем колонку
                if len(valid_indices) < window_length:
                    continue

                # Создаем временный массив с интерполяцией пропущенных значений
                temp_data = np.copy(col_data)
                interp_indices = np.arange(len(col_data))
                temp_data[nan_mask] = np.interp(
                    interp_indices[nan_mask],
                    interp_indices[~nan_mask],
                    col_data[~nan_mask]
                )

                # Применяем фильтр к временным данным
                filtered_col = signal.savgol_filter(temp_data, window_length, self.polyorder)

                # Сохраняем только валидные значения обратно
                filtered_data[:, col] = np.where(nan_mask, np.nan, filtered_col)
            else:
                # Применяем фильтр к полным данным
                filtered_data[:, col] = signal.savgol_filter(col_data, window_length, self.polyorder)

        # Преобразуем обратно в исходную форму
        if is_3d:
            filtered_data = filtered_data.reshape(original_shape)

        logger.debug(f"Фильтр Савицкого-Голея применен к данным формы {original_shape}")

        return filtered_data


class ExponentialFilter(LowPassFilter):
    """
    Экспоненциальный фильтр сглаживания данных.

    Attributes:
        alpha (float): Коэффициент сглаживания (0 < alpha < 1)
    """

    def __init__(self, alpha: float = 0.1, fs: float = 30.0):
        """
        Инициализирует экспоненциальный фильтр.

        Args:
            alpha: Коэффициент сглаживания (0 < alpha < 1)
            fs: Частота дискретизации данных (частота кадров, Гц)
        """
        super().__init__(fs)

        # Проверка параметров
        if alpha <= 0 or alpha >= 1:
            logger.warning(f"Параметр alpha ({alpha}) должен быть между 0 и 1. Устанавливаем в 0.1")
            self.alpha = 0.1
        else:
            self.alpha = alpha

        logger.info(f"Инициализирован экспоненциальный фильтр с alpha={self.alpha}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет экспоненциальный фильтр к данным.

        Args:
            data: Входные данные для фильтрации.
                 Может иметь форму (n_points, n_dims) или (n_frames, n_points, n_dims)

        Returns:
            np.ndarray: Отфильтрованные данные той же формы
        """
        # Проверка размерности входных данных
        original_shape = data.shape
        is_3d = len(original_shape) == 3

        # Если данные имеют 3 измерения (кадры, точки, координаты)
        if is_3d:
            n_frames, n_points, n_dims = original_shape
            # Преобразуем в 2D формат для фильтрации
            data_reshaped = data.reshape(n_frames, -1)
        else:
            # Данные уже в 2D формате (точки, координаты)
            data_reshaped = data

        # Создаем массив для отфильтрованных данных
        filtered_data = np.copy(data_reshaped)

        # Обрабатываем каждую колонку отдельно
        for col in range(data_reshaped.shape[1]):
            # Извлекаем колонку данных
            col_data = data_reshaped[:, col]

            # Находим маску пропущенных значений
            nan_mask = np.isnan(col_data)

            # Если все значения NaN, пропускаем колонку
            if np.all(nan_mask):
                continue

            # Находим первое валидное значение
            first_valid_idx = np.where(~nan_mask)[0][0]

            # Инициализируем результат с первым валидным значением
            result = np.full_like(col_data, np.nan)
            result[first_valid_idx] = col_data[first_valid_idx]

            # Применяем экспоненциальное сглаживание
            for i in range(first_valid_idx + 1, len(col_data)):
                if ~nan_mask[i]:
                    # Если текущее и предыдущее значения валидны
                    if ~np.isnan(result[i - 1]):
                        result[i] = self.alpha * col_data[i] + (1 - self.alpha) * result[i - 1]
                    else:
                        # Если предыдущее значение не валидно, используем текущее
                        result[i] = col_data[i]

            filtered_data[:, col] = result

        # Преобразуем обратно в исходную форму
        if is_3d:
            filtered_data = filtered_data.reshape(original_shape)

        logger.debug(f"Экспоненциальный фильтр применен к данным формы {original_shape}")

        return filtered_data


class MedianFilter(LowPassFilter):
    """
    Медианный фильтр для удаления выбросов.

    Attributes:
        kernel_size (int): Размер окна (должен быть нечетным)
    """

    def __init__(self, kernel_size: int = 3, fs: float = 30.0):
        """
        Инициализирует медианный фильтр.

        Args:
            kernel_size: Размер окна (должен быть нечетным)
            fs: Частота дискретизации данных (частота кадров, Гц)
        """
        super().__init__(fs)

        # Убедимся, что размер ядра нечетный
        if kernel_size % 2 == 0:
            kernel_size += 1
            logger.warning(f"Размер ядра должен быть нечетным. Увеличен до {kernel_size}")

        self.kernel_size = kernel_size

        logger.info(f"Инициализирован медианный фильтр с размером ядра {kernel_size}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет медианный фильтр к данным.

        Args:
            data: Входные данные для фильтрации.
                 Может иметь форму (n_points, n_dims) или (n_frames, n_points, n_dims)

        Returns:
            np.ndarray: Отфильтрованные данные той же формы
        """
        # Проверка размерности входных данных
        original_shape = data.shape
        is_3d = len(original_shape) == 3

        # Если данные имеют 3 измерения (кадры, точки, координаты)
        if is_3d:
            n_frames, n_points, n_dims = original_shape
            # Преобразуем в 2D формат для фильтрации
            data_reshaped = data.reshape(n_frames, -1)
        else:
            # Данные уже в 2D формате (точки, координаты)
            data_reshaped = data

        # Создаем массив для отфильтрованных данных
        filtered_data = np.copy(data_reshaped)

        # Если размер ядра больше количества кадров, уменьшаем его
        kernel_size = min(self.kernel_size, data_reshaped.shape[0])
        if kernel_size % 2 == 0:
            kernel_size -= 1

        # Если размер ядра менее 3, фильтрация невозможна
        if kernel_size < 3:
            logger.warning(f"Недостаточно данных для фильтрации. Возвращаем исходные данные.")
            return data

        # Обрабатываем каждую колонку отдельно
        for col in range(data_reshaped.shape[1]):
            # Извлекаем колонку данных
            col_data = data_reshaped[:, col]

            # Находим маску пропущенных значений
            nan_mask = np.isnan(col_data)

            # Если все значения NaN, пропускаем колонку
            if np.all(nan_mask):
                continue

            # Если есть пропущенные значения, обрабатываем только непрерывные участки
            if np.any(nan_mask):
                # Находим непрерывные участки валидных данных
                valid_segments = []
                start = None

                for i in range(len(col_data)):
                    if ~nan_mask[i] and start is None:
                        start = i
                    elif nan_mask[i] and start is not None:
                        valid_segments.append((start, i))
                        start = None

                # Добавляем последний сегмент, если он не закончился
                if start is not None:
                    valid_segments.append((start, len(col_data)))

                # Применяем медианный фильтр к каждому сегменту
                for start, end in valid_segments:
                    if end - start >= kernel_size:
                        segment = col_data[start:end]
                        filtered_segment = signal.medfilt(segment, kernel_size)
                        filtered_data[start:end, col] = filtered_segment
            else:
                # Применяем фильтр к полным данным
                filtered_data[:, col] = signal.medfilt(col_data, kernel_size)

        # Преобразуем обратно в исходную форму
        if is_3d:
            filtered_data = filtered_data.reshape(original_shape)

        logger.debug(f"Медианный фильтр применен к данным формы {original_shape}")

        return filtered_data


def apply_butterworth_filter(
        data: np.ndarray,
        fs: float = 30.0,
        cutoff: float = 6.0,
        order: int = 4
) -> np.ndarray:
    """
    Применяет фильтр Баттерворта к данным.

    Args:
        data: Входные данные для фильтрации
        fs: Частота дискретизации данных (частота кадров, Гц)
        cutoff: Частота среза фильтра (Гц)
        order: Порядок фильтра

    Returns:
        np.ndarray: Отфильтрованные данные
    """
    filter_obj = ButterworthFilter(fs=fs, cutoff=cutoff, order=order)
    return filter_obj.apply(data)


def apply_savgol_filter(
        data: np.ndarray,
        window_length: int = 15,
        polyorder: int = 3,
        fs: float = 30.0
) -> np.ndarray:
    """
    Применяет фильтр Савицкого-Голея к данным.

    Args:
        data: Входные данные для фильтрации
        window_length: Длина окна (должна быть нечетной)
        polyorder: Порядок полинома
        fs: Частота дискретизации данных (частота кадров, Гц)

    Returns:
        np.ndarray: Отфильтрованные данные
    """
    filter_obj = SavitzkyGolayFilter(window_length=window_length, polyorder=polyorder, fs=fs)
    return filter_obj.apply(data)


def apply_exponential_filter(
        data: np.ndarray,
        alpha: float = 0.1,
        fs: float = 30.0
) -> np.ndarray:
    """
    Применяет экспоненциальный фильтр к данным.

    Args:
        data: Входные данные для фильтрации
        alpha: Коэффициент сглаживания (0 < alpha < 1)
        fs: Частота дискретизации данных (частота кадров, Гц)

    Returns:
        np.ndarray: Отфильтрованные данные
    """
    filter_obj = ExponentialFilter(alpha=alpha, fs=fs)
    return filter_obj.apply(data)


def apply_median_filter(
        data: np.ndarray,
        kernel_size: int = 3,
        fs: float = 30.0
) -> np.ndarray:
    """
    Применяет медианный фильтр к данным.

    Args:
        data: Входные данные для фильтрации
        kernel_size: Размер окна (должен быть нечетным)
        fs: Частота дискретизации данных (частота кадров, Гц)

    Returns:
        np.ndarray: Отфильтрованные данные
    """
    filter_obj = MedianFilter(kernel_size=kernel_size, fs=fs)
    return filter_obj.apply(data)


def filter_data(
        data: np.ndarray,
        filter_type: str = "butterworth",
        fs: float = 30.0,
        **kwargs
) -> np.ndarray:
    """
    Применяет выбранный фильтр к данным.

    Args:
        data: Входные данные для фильтрации
        filter_type: Тип фильтра ("butterworth", "savgol", "exponential", "median")
        fs: Частота дискретизации данных (частота кадров, Гц)
        **kwargs: Дополнительные параметры для конкретного фильтра

    Returns:
        np.ndarray: Отфильтрованные данные
    """
    if filter_type.lower() == "butterworth":
        cutoff = kwargs.get("cutoff", 6.0)
        order = kwargs.get("order", 4)
        return apply_butterworth_filter(data, fs=fs, cutoff=cutoff, order=order)

    elif filter_type.lower() == "savgol":
        window_length = kwargs.get("window_length", 15)
        polyorder = kwargs.get("polyorder", 3)
        return apply_savgol_filter(data, window_length=window_length, polyorder=polyorder, fs=fs)

    elif filter_type.lower() == "exponential":
        alpha = kwargs.get("alpha", 0.1)
        return apply_exponential_filter(data, alpha=alpha, fs=fs)

    elif filter_type.lower() == "median":
        kernel_size = kwargs.get("kernel_size", 3)
        return apply_median_filter(data, kernel_size=kernel_size, fs=fs)

    else:
        logger.warning(f"Неизвестный тип фильтра: {filter_type}. Возвращаем исходные данные.")
        return data


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    n_frames = 100
    n_points = 3
    n_dims = 3

    # Синусоидальный сигнал с шумом
    t = np.linspace(0, 10, n_frames)
    clean_signal = np.sin(t)
    noise = np.random.normal(0, 0.2, n_frames)
    noisy_signal = clean_signal + noise

    # Создаем 3D данные для тестирования
    data_3d = np.zeros((n_frames, n_points, n_dims))
    for i in range(n_points):
        for j in range(n_dims):
            data_3d[:, i, j] = noisy_signal + i * 0.5 + j * 0.2

    # Добавляем пропущенные значения
    data_3d[10:15, 0, 0] = np.nan
    data_3d[50:60, 1, 1] = np.nan

    # Применяем различные фильтры
    filtered_butterworth = filter_data(data_3d, filter_type="butterworth", fs=10.0, cutoff=2.0, order=4)
    filtered_savgol = filter_data(data_3d, filter_type="savgol", window_length=11, polyorder=2)
    filtered_exp = filter_data(data_3d, filter_type="exponential", alpha=0.1)
    filtered_median = filter_data(data_3d, filter_type="median", kernel_size=5)

    logger.info("Фильтрация завершена")
