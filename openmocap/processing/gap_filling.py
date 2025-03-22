"""
Модуль для заполнения пропусков в данных захвата движения.

Содержит функции для обнаружения и заполнения пропусков в данных скелета,
которые могут возникать из-за окклюзий, ошибок отслеживания и других проблем.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy import interpolate, signal

logger = logging.getLogger(__name__)


def detect_gaps(data: np.ndarray) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Обнаруживает пропуски (NaN значения) в данных захвата движения.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)

    Returns:
        Tuple[List[Tuple[int, int, int]], np.ndarray]:
            - Список кортежей (индекс_точки, начало_пропуска, конец_пропуска)
            - Булева маска пропусков shape (n_frames, n_landmarks)
    """
    # Проверяем, что массив имеет правильную размерность
    if len(data.shape) != 3:
        raise ValueError("Входной массив должен иметь размерность (n_frames, n_landmarks, n_dims)")

    n_frames, n_landmarks, n_dims = data.shape

    # Создаем маску пропусков (True, если хотя бы одна координата точки - NaN)
    gaps_mask = np.isnan(data).any(axis=2)

    # Список для хранения информации о пропусках
    gaps_info = []

    # Для каждой точки
    for landmark_idx in range(n_landmarks):
        # Извлекаем маску пропусков для текущей точки
        landmark_gaps = gaps_mask[:, landmark_idx]

        # Если пропусков нет, пропускаем точку
        if not np.any(landmark_gaps):
            continue

        # Находим начала и концы последовательных пропусков
        # Используем разность между соседними элементами для обнаружения изменений
        changes = np.diff(landmark_gaps.astype(int))

        # Индексы, где начинаются пропуски (изменение с 0 на 1)
        gap_starts = np.where(changes == 1)[0] + 1

        # Индексы, где заканчиваются пропуски (изменение с 1 на 0)
        gap_ends = np.where(changes == -1)[0]

        # Обработка краевых случаев
        if landmark_gaps[0]:
            gap_starts = np.insert(gap_starts, 0, 0)

        if landmark_gaps[-1]:
            gap_ends = np.append(gap_ends, n_frames - 1)

        # Добавляем информацию о пропусках
        for start, end in zip(gap_starts, gap_ends):
            gaps_info.append((landmark_idx, start, end))

    return gaps_info, gaps_mask


def fill_gaps_linear(data: np.ndarray, max_gap_size: int = 10) -> np.ndarray:
    """
    Заполняет пропуски в данных с использованием линейной интерполяции.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        max_gap_size: Максимальный размер пропуска для заполнения

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    # Создаем копию входных данных
    filled_data = data.copy()

    # Получаем информацию о пропусках
    gaps_info, _ = detect_gaps(data)

    # Заполняем каждый пропуск
    for landmark_idx, start, end in gaps_info:
        gap_size = end - start + 1

        # Пропускаем пропуски, размер которых превышает максимальный
        if gap_size > max_gap_size:
            logger.info(
                f"Пропуск размером {gap_size} кадров для точки {landmark_idx} не заполнен (превышает максимум {max_gap_size})")
            continue

        # Находим ближайшие валидные точки до и после пропуска
        prev_valid_idx = start - 1
        next_valid_idx = end + 1

        # Проверяем, что точки находятся в пределах массива
        if prev_valid_idx < 0 or next_valid_idx >= data.shape[0]:
            logger.info(
                f"Пропуск для точки {landmark_idx} (кадры {start}-{end}) не может быть заполнен: находится на границе данных")
            continue

        # Получаем значения до и после пропуска
        prev_values = filled_data[prev_valid_idx, landmark_idx, :]
        next_values = filled_data[next_valid_idx, landmark_idx, :]

        # Проверяем, что эти значения валидны
        if np.isnan(prev_values).any() or np.isnan(next_values).any():
            logger.info(
                f"Пропуск для точки {landmark_idx} (кадры {start}-{end}) не может быть заполнен: нет валидных соседних точек")
            continue

        # Линейно интерполируем значения
        for dim in range(data.shape[2]):
            for i, frame_idx in enumerate(range(start, end + 1)):
                t = (i + 1) / (gap_size + 1)  # Нормализованное время (от 0 до 1)
                filled_data[frame_idx, landmark_idx, dim] = prev_values[dim] + t * (next_values[dim] - prev_values[dim])

        logger.debug(f"Заполнен пропуск размером {gap_size} кадров для точки {landmark_idx} (кадры {start}-{end})")

    return filled_data


def fill_gaps_spline(data: np.ndarray, max_gap_size: int = 20, order: int = 3) -> np.ndarray:
    """
    Заполняет пропуски в данных с использованием сплайн-интерполяции.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        max_gap_size: Максимальный размер пропуска для заполнения
        order: Порядок сплайна (1=линейный, 2=квадратичный, 3=кубический)

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    # Создаем копию входных данных
    filled_data = data.copy()

    # Получаем информацию о пропусках
    gaps_info, _ = detect_gaps(data)

    # Параметр k для сплайна (порядок - 1)
    k = min(order, 5)  # Ограничиваем максимальным значением 5

    # Заполняем каждый пропуск
    for landmark_idx, start, end in gaps_info:
        gap_size = end - start + 1

        # Пропускаем пропуски, размер которых превышает максимальный
        if gap_size > max_gap_size:
            logger.info(
                f"Пропуск размером {gap_size} кадров для точки {landmark_idx} не заполнен (превышает максимум {max_gap_size})")
            continue

        # Контекстное окно для сплайна (берем точки до и после пропуска)
        context_size = min(gap_size * 2, 20)  # Размер контекста зависит от размера пропуска, но не более 20 кадров

        prev_start = max(0, start - context_size)
        next_end = min(data.shape[0] - 1, end + context_size)

        # Используем только валидные данные в контексте
        x_all = np.arange(prev_start, next_end + 1)
        valid_data_mask = ~np.isnan(data[x_all, landmark_idx, :]).any(axis=1)

        # Если недостаточно валидных точек для сплайна, используем линейную интерполяцию
        if np.sum(valid_data_mask) < k + 1:
            logger.debug(
                f"Недостаточно валидных данных для сплайна порядка {order}, используется линейная интерполяция")
            # Заполняем пропуск линейной интерполяцией
            for dim in range(data.shape[2]):
                dimension_data = data[x_all, landmark_idx, dim]
                valid_mask = ~np.isnan(dimension_data)

                if np.sum(valid_mask) >= 2:  # Нужно минимум 2 точки для линейной интерполяции
                    x_valid = x_all[valid_mask]
                    y_valid = dimension_data[valid_mask]

                    # Создаем линейный интерполятор
                    interp_func = interpolate.interp1d(x_valid, y_valid, bounds_error=False, fill_value=np.nan)

                    # Заполняем пропуски
                    gap_idxs = np.arange(start, end + 1)
                    filled_data[gap_idxs, landmark_idx, dim] = interp_func(gap_idxs)

            continue

        # Если достаточно валидных точек для сплайна, используем сплайн-интерполяцию
        x_valid = x_all[valid_data_mask]
        y_valid = data[x_valid, landmark_idx, :]

        # Интерполируем каждое измерение отдельно
        for dim in range(data.shape[2]):
            # Создаем сплайн
            spline = interpolate.splrep(x_valid, y_valid[:, dim], k=k)

            # Вычисляем значения в пропущенных точках
            gap_idxs = np.arange(start, end + 1)
            filled_data[gap_idxs, landmark_idx, dim] = interpolate.splev(gap_idxs, spline)

        logger.debug(
            f"Заполнен пропуск размером {gap_size} кадров для точки {landmark_idx} (кадры {start}-{end}) с использованием сплайна порядка {order}")

    return filled_data


def fill_gaps_kalman(data: np.ndarray, max_gap_size: int = 30, process_noise: float = 0.001,
                     measurement_noise: float = 0.01) -> np.ndarray:
    """
    Заполняет пропуски в данных с использованием фильтра Калмана.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        max_gap_size: Максимальный размер пропуска для заполнения
        process_noise: Шум процесса (определяет гибкость модели)
        measurement_noise: Шум измерений (определяет доверие к измерениям)

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    try:
        from filterpy.kalman import KalmanFilter
    except ImportError:
        logger.error("Библиотека filterpy не установлена. Установите её с помощью pip: pip install filterpy")
        # Возвращаем исходные данные, если не можем использовать фильтр Калмана
        return fill_gaps_spline(data, max_gap_size)

    # Создаем копию входных данных
    filled_data = data.copy()

    # Получаем информацию о пропусках
    gaps_info, _ = detect_gaps(data)

    n_frames, n_landmarks, n_dims = data.shape

    # Для каждой точки скелета создаем и применяем фильтр Калмана
    for landmark_idx in range(n_landmarks):
        # Проверяем, есть ли пропуски для этой точки
        landmark_gaps = [gap for gap in gaps_info if gap[0] == landmark_idx]
        if not landmark_gaps:
            continue

        # Создаем фильтр Калмана для этой точки
        # Состояние: [x, y, z, vx, vy, vz] (позиция и скорость)
        kf = KalmanFilter(dim_x=2 * n_dims, dim_z=n_dims)

        # Матрица перехода (A)
        kf.F = np.eye(2 * n_dims)
        dt = 1.0  # Шаг времени (1 кадр)
        for i in range(n_dims):
            kf.F[i, i + n_dims] = dt

        # Матрица измерений (H)
        kf.H = np.zeros((n_dims, 2 * n_dims))
        for i in range(n_dims):
            kf.H[i, i] = 1.0

        # Ковариационная матрица шума процесса (Q)
        kf.Q = np.eye(2 * n_dims) * process_noise

        # Ковариационная матрица шума измерений (R)
        kf.R = np.eye(n_dims) * measurement_noise

        # Начальная ковариационная матрица ошибки (P)
        kf.P = np.eye(2 * n_dims)

        # Запускаем фильтр для всей последовательности
        landmark_data = filled_data[:, landmark_idx, :]

        # Инициализируем состояние из первых двух валидных точек
        valid_indices = np.where(~np.isnan(landmark_data).any(axis=1))[0]
        if len(valid_indices) < 2:
            logger.warning(f"Недостаточно валидных данных для точки {landmark_idx}, пропуски не заполнены")
            continue

        # Получаем первую валидную точку
        first_valid_idx = valid_indices[0]
        first_valid_point = landmark_data[first_valid_idx]

        # Получаем вторую валидную точку для вычисления начальной скорости
        second_valid_idx = valid_indices[1]
        second_valid_point = landmark_data[second_valid_idx]

        # Вычисляем начальную скорость
        initial_velocity = (second_valid_point - first_valid_point) / (second_valid_idx - first_valid_idx)

        # Устанавливаем начальное состояние
        initial_state = np.zeros(2 * n_dims)
        initial_state[:n_dims] = first_valid_point
        initial_state[n_dims:] = initial_velocity
        kf.x = initial_state

        # Прогноз и обновление для каждого кадра
        for frame_idx in range(n_frames):
            # Прогнозируем следующее состояние
            kf.predict()

            # Если у нас есть валидные данные, обновляем состояние
            if not np.isnan(landmark_data[frame_idx]).any():
                kf.update(landmark_data[frame_idx])

            # Для пропусков в пределах максимального размера, используем прогнозированные значения
            current_gap = None
            for gap in landmark_gaps:
                if gap[1] <= frame_idx <= gap[2]:
                    current_gap = gap
                    break

            if current_gap and (current_gap[2] - current_gap[1] + 1) <= max_gap_size:
                filled_data[frame_idx, landmark_idx, :] = kf.x[:n_dims]

    return filled_data


def fill_gaps_pattern(data: np.ndarray, max_gap_size: int = 20,
                      reference_landmarks: Optional[List[int]] = None) -> np.ndarray:
    """
    Заполняет пропуски в данных, используя паттерны движения от других ориентиров.

    Этот метод полезен, когда определенные точки (например, колено и лодыжка) двигаются схожим образом.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        max_gap_size: Максимальный размер пропуска для заполнения
        reference_landmarks: Список индексов ориентиров, используемых в качестве референсных.
                           Если None, используются все валидные ориентиры.

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    # Создаем копию входных данных
    filled_data = data.copy()

    # Получаем информацию о пропусках
    gaps_info, gaps_mask = detect_gaps(data)

    # Размеры данных
    n_frames, n_landmarks, n_dims = data.shape

    # Если не указаны референсные ориентиры, используем все
    if reference_landmarks is None:
        reference_landmarks = list(range(n_landmarks))

    # Для каждого пропуска
    for landmark_idx, start, end in gaps_info:
        gap_size = end - start + 1

        # Пропускаем пропуски, размер которых превышает максимальный
        if gap_size > max_gap_size:
            logger.info(
                f"Пропуск размером {gap_size} кадров для точки {landmark_idx} не заполнен (превышает максимум {max_gap_size})")
            continue

        # Пропускаем, если это референсная точка
        if landmark_idx in reference_landmarks:
            continue

        # Находим контекст до и после пропуска
        context_size = min(gap_size * 2, 30)  # Размер контекста зависит от размера пропуска, но не более 30 кадров
        context_start = max(0, start - context_size)
        context_end = min(n_frames - 1, end + context_size)

        # Получаем данные для точки с пропуском (включая контекст)
        landmark_data = filled_data[context_start:context_end + 1, landmark_idx, :]
        landmark_mask = ~np.isnan(landmark_data).any(axis=1)

        # Если нет валидных данных в контексте, пропускаем
        if not np.any(landmark_mask):
            continue

        # Находим наиболее подходящий референсный ориентир
        best_reference = None
        best_correlation = -1

        for ref_idx in reference_landmarks:
            if ref_idx == landmark_idx:
                continue

            # Получаем данные референсного ориентира
            ref_data = filled_data[context_start:context_end + 1, ref_idx, :]
            ref_mask = ~np.isnan(ref_data).any(axis=1)

            # Находим общие валидные кадры
            common_valid = landmark_mask & ref_mask

            # Если недостаточно общих валидных кадров, пропускаем
            if np.sum(common_valid) < 10:  # Минимум 10 общих валидных кадров
                continue

            # Вычисляем корреляцию между ориентирами
            correlation = 0
            for dim in range(n_dims):
                corr = np.corrcoef(landmark_data[common_valid, dim], ref_data[common_valid, dim])[0, 1]
                correlation += abs(corr)  # Абсолютное значение корреляции

            correlation /= n_dims  # Среднее по всем измерениям

            # Если это лучшая корреляция, запоминаем
            if correlation > best_correlation:
                best_correlation = correlation
                best_reference = ref_idx

        # Если не найден подходящий референсный ориентир, используем сплайн-интерполяцию
        if best_reference is None or best_correlation < 0.5:  # Порог корреляции
            logger.debug(
                f"Не найден подходящий референсный ориентир для точки {landmark_idx}, используется сплайн-интерполяция")

            # Заполняем конкретный пропуск с помощью сплайна
            for dim in range(n_dims):
                x_valid = np.arange(context_start, context_end + 1)[landmark_mask]
                y_valid = landmark_data[landmark_mask, dim]

                if len(x_valid) < 4:  # Недостаточно точек для кубического сплайна
                    if len(x_valid) >= 2:  # Но достаточно для линейной интерполяции
                        interp_func = interpolate.interp1d(x_valid, y_valid, bounds_error=False,
                                                           fill_value="extrapolate")
                        filled_data[start:end + 1, landmark_idx, dim] = interp_func(np.arange(start, end + 1))
                else:
                    # Используем кубический сплайн
                    spline = interpolate.splrep(x_valid, y_valid, k=min(3, len(x_valid) - 1))
                    filled_data[start:end + 1, landmark_idx, dim] = interpolate.splev(np.arange(start, end + 1), spline)
        else:
            # Используем паттерн движения референсного ориентира
            logger.debug(
                f"Используется паттерн движения ориентира {best_reference} для заполнения пропуска в точке {landmark_idx} (корреляция: {best_correlation:.2f})")

            # Вычисляем контекстное смещение между ориентирами
            ref_data = filled_data[context_start:context_end + 1, best_reference, :]
            ref_mask = ~np.isnan(ref_data).any(axis=1)

            common_valid = landmark_mask & ref_mask

            if not np.any(common_valid):
                continue

            # Вычисляем среднее смещение
            offset = np.mean(landmark_data[common_valid] - ref_data[common_valid], axis=0)

            # Проверяем, что референсный ориентир имеет валидные данные в пропуске
            gap_ref_data = filled_data[start:end + 1, best_reference, :]
            gap_ref_mask = ~np.isnan(gap_ref_data).any(axis=1)

            # Заполняем пропуск
            for i, frame_idx in enumerate(range(start, end + 1)):
                if gap_ref_mask[i]:
                    filled_data[frame_idx, landmark_idx, :] = gap_ref_data[i] + offset

    return filled_data


def fill_gaps_combined(data: np.ndarray, max_gap_size: int = 30, methods: Optional[List[str]] = None) -> np.ndarray:
    """
    Заполняет пропуски в данных, используя комбинацию методов в зависимости от размера пропуска.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        max_gap_size: Максимальный размер пропуска для заполнения
        methods: Список методов для использования. Если None, используются все доступные методы.
                Варианты: 'linear', 'spline', 'kalman', 'pattern'

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    # Создаем копию входных данных
    filled_data = data.copy()

    # Если не указаны методы, используем все доступные
    if methods is None:
        methods = ['linear', 'spline', 'kalman', 'pattern']

    # Проверяем, что все методы валидны
    valid_methods = {'linear', 'spline', 'kalman', 'pattern'}
    for method in methods:
        if method not in valid_methods:
            raise ValueError(f"Неизвестный метод заполнения пропусков: {method}. Доступные методы: {valid_methods}")

    # Получаем информацию о пропусках
    gaps_info, gaps_mask = detect_gaps(data)

    # Группируем пропуски по размеру
    small_gaps = []  # <5 кадров
    medium_gaps = []  # 5-15 кадров
    large_gaps = []  # >15 кадров

    for gap in gaps_info:
        landmark_idx, start, end = gap
        gap_size = end - start + 1

        if gap_size > max_gap_size:
            logger.info(
                f"Пропуск размером {gap_size} кадров для точки {landmark_idx} не заполнен (превышает максимум {max_gap_size})")
            continue

        if gap_size < 5:
            small_gaps.append(gap)
        elif gap_size < 15:
            medium_gaps.append(gap)
        else:
            large_gaps.append(gap)

    # Заполняем пропуски в зависимости от их размера

    # 1. Маленькие пропуски - линейная интерполяция (быстро и эффективно)
    if 'linear' in methods and small_gaps:
        # Создаем массив с индикаторами пропусков для маленьких пропусков
        small_gaps_mask = np.zeros_like(gaps_mask, dtype=bool)
        for landmark_idx, start, end in small_gaps:
            small_gaps_mask[start:end + 1, landmark_idx] = True

        # Создаем временный массив только с маленькими пропусками
        temp_data = filled_data.copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if gaps_mask[i, j] and not small_gaps_mask[i, j]:
                    # Если это не маленький пропуск, заменяем его валидным значением
                    temp_data[i, j] = np.zeros(data.shape[2])

        # Заполняем маленькие пропуски
        logger.info(
            f"Заполнение {len(small_gaps)} маленьких пропусков (<5 кадров) с использованием линейной интерполяции")
        filled_small = fill_gaps_linear(temp_data)

        # Копируем заполненные значения обратно
        for landmark_idx, start, end in small_gaps:
            filled_data[start:end + 1, landmark_idx] = filled_small[start:end + 1, landmark_idx]

    # 2. Средние пропуски - сплайн-интерполяция (хорошо для плавных движений)
    if 'spline' in methods and medium_gaps:
        # Создаем массив с индикаторами пропусков для средних пропусков
        medium_gaps_mask = np.zeros_like(gaps_mask, dtype=bool)
        for landmark_idx, start, end in medium_gaps:
            medium_gaps_mask[start:end + 1, landmark_idx] = True

        # Создаем временный массив только со средними пропусками
        temp_data = filled_data.copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if gaps_mask[i, j] and not medium_gaps_mask[i, j]:
                    # Если это не средний пропуск, заменяем его валидным значением
                    temp_data[i, j] = np.zeros(data.shape[2])

        # Заполняем средние пропуски
        logger.info(
            f"Заполнение {len(medium_gaps)} средних пропусков (5-15 кадров) с использованием сплайн-интерполяции")
        filled_medium = fill_gaps_spline(temp_data)

        # Копируем заполненные значения обратно
        for landmark_idx, start, end in medium_gaps:
            filled_data[start:end + 1, landmark_idx] = filled_medium[start:end + 1, landmark_idx]

    # 3. Большие пропуски - комбинация Калмана и паттернов (для сложных случаев)
    if large_gaps:
        large_gaps_mask = np.zeros_like(gaps_mask, dtype=bool)
        for landmark_idx, start, end in large_gaps:
            large_gaps_mask[start:end + 1, landmark_idx] = True

        # Создаем временный массив только с большими пропусками
        temp_data = filled_data.copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if gaps_mask[i, j] and not large_gaps_mask[i, j]:
                    # Если это не большой пропуск, заменяем его валидным значением
                    temp_data[i, j] = np.zeros(data.shape[2])

        # Заполняем большие пропуски комбинацией методов
        logger.info(f"Заполнение {len(large_gaps)} больших пропусков (>15 кадров)")

        if 'pattern' in methods:
            logger.info("Используется метод заполнения на основе паттернов движения")
            filled_pattern = fill_gaps_pattern(temp_data)

            # Копируем заполненные значения обратно
            for landmark_idx, start, end in large_gaps:
                filled_data[start:end + 1, landmark_idx] = filled_pattern[start:end + 1, landmark_idx]

        # Для оставшихся пропусков используем фильтр Калмана
        if 'kalman' in methods:
            # Проверяем, остались ли незаполненные пропуски
            remaining_gaps_mask = np.isnan(filled_data).any(axis=2) & large_gaps_mask

            if np.any(remaining_gaps_mask):
                logger.info("Используется фильтр Калмана для оставшихся пропусков")

                # Создаем временный массив только с оставшимися пропусками
                temp_data = filled_data.copy()
                filled_kalman = fill_gaps_kalman(temp_data)

                # Копируем заполненные значения обратно только для оставшихся пропусков
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if remaining_gaps_mask[i, j]:
                            filled_data[i, j] = filled_kalman[i, j]

    # Проверяем, остались ли незаполненные пропуски
    remaining_gaps = np.isnan(filled_data).any(axis=2)
    if np.any(remaining_gaps):
        logger.warning(f"После применения всех методов осталось {np.sum(remaining_gaps)} незаполненных точек")
    else:
        logger.info("Все пропуски успешно заполнены")

    return filled_data


def weighted_average_fill(data: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Заполняет пропуски в данных с использованием взвешенного среднего от соседних точек.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        weights: Массив весов для соседних кадров shape (window_size)
                Если None, используются экспоненциально убывающие веса

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    # Создаем копию входных данных
    filled_data = data.copy()

    # Получаем информацию о пропусках
    gaps_info, _ = detect_gaps(data)

    # Если не указаны веса, используем экспоненциально убывающие
    window_size = 5  # Размер окна по умолчанию
    if weights is None:
        # Экспоненциально убывающие веса
        weights = np.exp(-np.arange(window_size))
        weights = weights / np.sum(weights)  # Нормализация
    else:
        window_size = len(weights)

    # Для каждого пропуска
    for landmark_idx, start, end in gaps_info:
        gap_size = end - start + 1

        # Находим соседние валидные точки
        valid_before = []
        for i in range(1, window_size + 1):
            idx = start - i
            if idx >= 0 and not np.isnan(filled_data[idx, landmark_idx]).any():
                valid_before.append((idx, window_size - i))

        valid_after = []
        for i in range(1, window_size + 1):
            idx = end + i
            if idx < data.shape[0] and not np.isnan(filled_data[idx, landmark_idx]).any():
                valid_after.append((idx, window_size - i))

        # Если нет валидных соседей, пропускаем
        if not valid_before and not valid_after:
            continue

        # Вычисляем взвешенное среднее
        for frame_idx in range(start, end + 1):
            total_weight = 0
            weighted_value = np.zeros(data.shape[2])

            # Учитываем точки до пропуска
            for idx, w_idx in valid_before:
                w = weights[w_idx] * (
                            1 - (frame_idx - start) / gap_size)  # Уменьшаем вес с удалением от начала пропуска
                total_weight += w
                weighted_value += w * filled_data[idx, landmark_idx]

            # Учитываем точки после пропуска
            for idx, w_idx in valid_after:
                w = weights[w_idx] * (1 - (end - frame_idx) / gap_size)  # Уменьшаем вес с удалением от конца пропуска
                total_weight += w
                weighted_value += w * filled_data[idx, landmark_idx]

            # Нормализуем веса
            if total_weight > 0:
                filled_data[frame_idx, landmark_idx] = weighted_value / total_weight

    return filled_data


def apply_constraints(data: np.ndarray, constraints: Dict[Tuple[int, int], float],
                      tolerance: float = 0.05) -> np.ndarray:
    """
    Применяет ограничения длины костей к заполненным данным.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        constraints: Словарь ограничений {(индекс_точки1, индекс_точки2): длина}
        tolerance: Допустимое отклонение от заданной длины (в процентах)

    Returns:
        np.ndarray: Массив с примененными ограничениями
    """
    # Создаем копию входных данных
    constrained_data = data.copy()

    n_frames = data.shape[0]

    # Для каждого кадра
    for frame_idx in range(n_frames):
        # Для каждого ограничения
        for (point1_idx, point2_idx), target_length in constraints.items():
            # Проверяем, что обе точки валидны
            if np.isnan(constrained_data[frame_idx, point1_idx]).any() or np.isnan(
                    constrained_data[frame_idx, point2_idx]).any():
                continue

            # Получаем координаты точек
            point1 = constrained_data[frame_idx, point1_idx]
            point2 = constrained_data[frame_idx, point2_idx]

            # Вычисляем текущую длину
            current_length = np.linalg.norm(point2 - point1)

            # Если длина не соответствует ограничению
            if abs(current_length - target_length) > target_length * tolerance:
                # Вычисляем направление вектора
                direction = (point2 - point1) / current_length

                # Корректируем точки
                delta_length = target_length - current_length

                # Равномерно распределяем корректировку между точками
                point1_new = point1 - direction * delta_length / 2
                point2_new = point2 + direction * delta_length / 2

                # Обновляем координаты
                constrained_data[frame_idx, point1_idx] = point1_new
                constrained_data[frame_idx, point2_idx] = point2_new

    return constrained_data


def fill_gaps(data: np.ndarray, method: str = "combined", max_gap_size: int = 30, **kwargs) -> np.ndarray:
    """
    Заполняет пропуски в данных захвата движения с использованием указанного метода.

    Args:
        data: Массив данных shape (n_frames, n_landmarks, n_dims)
        method: Метод заполнения пропусков
                ('linear', 'spline', 'kalman', 'pattern', 'combined', 'weighted')
        max_gap_size: Максимальный размер пропуска для заполнения
        **kwargs: Дополнительные параметры для конкретного метода

    Returns:
        np.ndarray: Массив с заполненными пропусками
    """
    if method == "linear":
        return fill_gaps_linear(data, max_gap_size=max_gap_size)
    elif method == "spline":
        order = kwargs.get("order", 3)
        return fill_gaps_spline(data, max_gap_size=max_gap_size, order=order)
    elif method == "kalman":
        process_noise = kwargs.get("process_noise", 0.001)
        measurement_noise = kwargs.get("measurement_noise", 0.01)
        return fill_gaps_kalman(data, max_gap_size=max_gap_size,
                                process_noise=process_noise,
                                measurement_noise=measurement_noise)
    elif method == "pattern":
        reference_landmarks = kwargs.get("reference_landmarks", None)
        return fill_gaps_pattern(data, max_gap_size=max_gap_size,
                                 reference_landmarks=reference_landmarks)
    elif method == "combined":
        methods = kwargs.get("methods", None)
        return fill_gaps_combined(data, max_gap_size=max_gap_size, methods=methods)
    elif method == "weighted":
        weights = kwargs.get("weights", None)
        return weighted_average_fill(data, weights=weights)
    else:
        raise ValueError(f"Неизвестный метод заполнения пропусков: {method}")


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем тестовые данные
    n_frames = 100
    n_landmarks = 10
    n_dims = 3
    data = np.random.rand(n_frames, n_landmarks, n_dims)

    # Вносим искусственные пропуски
    mask = np.random.rand(n_frames, n_landmarks) < 0.1  # 10% пропусков
    for i in range(n_frames):
        for j in range(n_landmarks):
            if mask[i, j]:
                # Создаем пропуск случайной длины
                gap_length = np.random.randint(1, 10)
                end = min(i + gap_length, n_frames)
                data[i:end, j, :] = np.nan

    # Заполняем пропуски различными методами
    filled_linear = fill_gaps(data, method="linear")
    filled_spline = fill_gaps(data, method="spline")
    filled_combined = fill_gaps(data, method="combined")

    # Выводим статистику
    gaps_info, gaps_mask = detect_gaps(data)
    filled_linear_gaps, filled_linear_mask = detect_gaps(filled_linear)
    filled_spline_gaps, filled_spline_mask = detect_gaps(filled_spline)
    filled_combined_gaps, filled_combined_mask = detect_gaps(filled_combined)

    print(f"Исходное количество пропусков: {len(gaps_info)}")
    print(f"Осталось пропусков после линейной интерполяции: {len(filled_linear_gaps)}")
    print(f"Осталось пропусков после сплайн-интерполяции: {len(filled_spline_gaps)}")
    print(f"Осталось пропусков после комбинированного метода: {len(filled_combined_gaps)}")