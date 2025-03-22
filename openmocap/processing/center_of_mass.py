"""
Модуль для расчета центра масс.

Содержит функции и классы для расчета центра масс сегментов тела и всего тела
на основе данных скелета и антропометрической информации.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass

from openmocap.tracking.skeleton_model import SkeletonModel

logger = logging.getLogger(__name__)


@dataclass
class SegmentMassInfo:
    """
    Класс для хранения антропометрической информации о сегменте тела.

    Attributes:
        segment_name (str): Название сегмента тела
        mass_percentage (float): Процент от общей массы тела (в диапазоне 0-1)
        com_position (float): Положение центра масс сегмента относительно проксимальной точки (в диапазоне 0-1)
    """
    segment_name: str
    mass_percentage: float
    com_position: float


class AnthropometricData:
    """
    Класс для хранения и предоставления антропометрических данных.

    Содержит информацию о распределении массы тела и положении центров масс для различных сегментов.
    Основано на антропометрических таблицах из биомеханической литературы.

    Attributes:
        segment_data (Dict[str, SegmentMassInfo]): Словарь с информацией о сегментах
    """

    # Антропометрические данные по Winter D.A., "Biomechanics and Motor Control of Human Movement"
    # Значения: (процент от общей массы, положение центра масс от проксимального конца)
    DEFAULT_SEGMENT_DATA = {
        # Туловище и голова
        "head": (0.081, 0.5),  # Голова
        "trunk": (0.497, 0.5),  # Туловище

        # Руки
        "upper_arm_right": (0.028, 0.436),  # Плечо правое
        "upper_arm_left": (0.028, 0.436),  # Плечо левое
        "forearm_right": (0.016, 0.430),  # Предплечье правое
        "forearm_left": (0.016, 0.430),  # Предплечье левое
        "hand_right": (0.006, 0.506),  # Кисть правая
        "hand_left": (0.006, 0.506),  # Кисть левая

        # Ноги
        "thigh_right": (0.100, 0.433),  # Бедро правое
        "thigh_left": (0.100, 0.433),  # Бедро левое
        "shank_right": (0.047, 0.433),  # Голень правая
        "shank_left": (0.047, 0.433),  # Голень левая
        "foot_right": (0.014, 0.500),  # Стопа правая
        "foot_left": (0.014, 0.500),  # Стопа левая
    }

    def __init__(self, custom_data: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Инициализирует объект антропометрических данных.

        Args:
            custom_data: Пользовательские антропометрические данные, переопределяющие стандартные.
                        Формат: {"segment_name": (mass_percentage, com_position)}
        """
        # Объединяем стандартные данные с пользовательскими, если они предоставлены
        data = self.DEFAULT_SEGMENT_DATA.copy()
        if custom_data:
            data.update(custom_data)

        # Создаем словарь сегментов
        self.segment_data = {}
        for segment_name, (mass_percentage, com_position) in data.items():
            self.segment_data[segment_name] = SegmentMassInfo(
                segment_name=segment_name,
                mass_percentage=mass_percentage,
                com_position=com_position
            )

        # Проверяем, что сумма процентов массы примерно равна 1
        total_mass = sum(segment.mass_percentage for segment in self.segment_data.values())
        if not 0.99 <= total_mass <= 1.01:
            logger.warning(
                f"Сумма процентов массы всех сегментов ({total_mass}) значительно отличается от 1.0. "
                "Это может привести к неточным результатам расчета центра масс."
            )

    def get_segment_info(self, segment_name: str) -> SegmentMassInfo:
        """
        Возвращает информацию о сегменте.

        Args:
            segment_name: Название сегмента

        Returns:
            SegmentMassInfo: Информация о сегменте

        Raises:
            KeyError: Если сегмент с указанным именем не найден
        """
        if segment_name not in self.segment_data:
            raise KeyError(f"Сегмент с именем '{segment_name}' не найден в антропометрических данных")

        return self.segment_data[segment_name]

    def get_all_segment_names(self) -> List[str]:
        """
        Возвращает список имен всех сегментов.

        Returns:
            List[str]: Список имен сегментов
        """
        return list(self.segment_data.keys())


class CenterOfMassCalculator:
    """
    Класс для расчета центра масс тела на основе скелетных данных.

    Attributes:
        skeleton_model (SkeletonModel): Модель скелета
        anthropometric_data (AnthropometricData): Антропометрические данные
        segment_mapping (Dict[str, Tuple[int, int]]): Соответствие сегментов точкам скелета
    """

    def __init__(
            self,
            skeleton_model: SkeletonModel,
            anthropometric_data: Optional[AnthropometricData] = None,
            segment_mapping: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Инициализирует объект для расчета центра масс.

        Args:
            skeleton_model: Модель скелета
            anthropometric_data: Антропометрические данные. Если None, используются стандартные.
            segment_mapping: Соответствие сегментов точкам скелета. Если None, используется автоматическое сопоставление.
        """
        self.skeleton_model = skeleton_model
        self.anthropometric_data = anthropometric_data or AnthropometricData()

        # Если соответствие сегментов не предоставлено, создаем его автоматически
        if segment_mapping is None:
            segment_mapping = self._create_segment_mapping_mediapipe()

        self.segment_mapping = segment_mapping

        # Проверяем, что все необходимые сегменты присутствуют в маппинге
        missing_segments = [
            segment for segment in self.anthropometric_data.get_all_segment_names()
            if segment not in self.segment_mapping
        ]
        if missing_segments:
            logger.warning(
                f"Следующие сегменты не имеют соответствия в скелетной модели: {missing_segments}. "
                "Эти сегменты будут пропущены при расчете центра масс."
            )

    def _create_segment_mapping_mediapipe(self) -> Dict[str, Tuple[int, int]]:
        """
        Создает соответствие между сегментами тела и точками скелета MediaPipe.

        Returns:
            Dict[str, Tuple[int, int]]: Словарь соответствия сегментов точкам скелета
        """
        # Проверяем, что у нас модель MediaPipe
        landmark_names = self.skeleton_model.landmark_names
        if not all(name in landmark_names for name in
                   ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            logger.warning(
                "Скелетная модель не похожа на MediaPipe. Автоматическое создание соответствия может быть неточным.")

        # Для удобства получаем индексы ключевых точек
        indices = {name: i for i, name in enumerate(landmark_names)}

        # Создаем соответствие сегментов точкам скелета
        mapping = {
            # Голова
            "head": (indices.get("nose", -1), indices.get("left_shoulder", -1)),

            # Туловище (между центрами плеч и бедер)
            "trunk": (
                # Середина между плечами
                -1,  # Этот индекс будет вычислен специальным образом
                # Середина между бедрами
                -2,  # Этот индекс будет вычислен специальным образом
            ),

            # Руки
            "upper_arm_right": (indices.get("right_shoulder", -1), indices.get("right_elbow", -1)),
            "upper_arm_left": (indices.get("left_shoulder", -1), indices.get("left_elbow", -1)),
            "forearm_right": (indices.get("right_elbow", -1), indices.get("right_wrist", -1)),
            "forearm_left": (indices.get("left_elbow", -1), indices.get("left_wrist", -1)),
            "hand_right": (indices.get("right_wrist", -1), indices.get("right_index", -1)),
            "hand_left": (indices.get("left_wrist", -1), indices.get("left_index", -1)),

            # Ноги
            "thigh_right": (indices.get("right_hip", -1), indices.get("right_knee", -1)),
            "thigh_left": (indices.get("left_hip", -1), indices.get("left_knee", -1)),
            "shank_right": (indices.get("right_knee", -1), indices.get("right_ankle", -1)),
            "shank_left": (indices.get("left_knee", -1), indices.get("left_ankle", -1)),
            "foot_right": (indices.get("right_ankle", -1), indices.get("right_foot_index", -1)),
            "foot_left": (indices.get("left_ankle", -1), indices.get("left_foot_index", -1)),
        }

        # Удаляем сегменты с отсутствующими точками
        valid_mapping = {
            segment: (start, end)
            for segment, (start, end) in mapping.items()
            if start >= -2 and end >= -2  # Разрешаем специальные индексы -1 и -2
        }

        if len(valid_mapping) < len(mapping):
            logger.warning(
                f"Не удалось создать соответствие для {len(mapping) - len(valid_mapping)} сегментов "
                "из-за отсутствия необходимых точек в скелетной модели."
            )

        return valid_mapping

    def calculate_segment_com(
            self,
            skeleton_data: np.ndarray,
            frame_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Рассчитывает центр масс для каждого сегмента тела.

        Args:
            skeleton_data: Массив координат точек скелета shape (num_frames, num_landmarks, 3)
            frame_idx: Индекс кадра для расчета (если None, расчеты выполняются для всех кадров)

        Returns:
            Dict[str, np.ndarray]: Словарь с центрами масс для каждого сегмента
                                {segment_name: com_coordinates}
        """
        # Проверяем, является ли skeleton_data многокадровым
        is_sequence = len(skeleton_data.shape) == 3 and skeleton_data.shape[0] > 1

        # Если передан конкретный кадр, извлекаем его данные
        if is_sequence and frame_idx is not None:
            frame_data = skeleton_data[frame_idx]
        else:
            frame_data = skeleton_data

        # Словарь для хранения результатов
        segment_com = {}

        # Обрабатываем каждый сегмент
        for segment_name, (start_idx, end_idx) in self.segment_mapping.items():
            # Получаем антропометрическую информацию о сегменте
            try:
                segment_info = self.anthropometric_data.get_segment_info(segment_name)
            except KeyError:
                continue  # Пропускаем сегмент, если для него нет антропометрической информации

            # Обработка специальных индексов для туловища
            if segment_name == "trunk":
                # Для туловища рассчитываем центры плеч и бедер вручную
                try:
                    left_shoulder_idx = self.skeleton_model.landmark_indices.get("left_shoulder")
                    right_shoulder_idx = self.skeleton_model.landmark_indices.get("right_shoulder")
                    left_hip_idx = self.skeleton_model.landmark_indices.get("left_hip")
                    right_hip_idx = self.skeleton_model.landmark_indices.get("right_hip")

                    if is_sequence:
                        # Для последовательности кадров
                        shoulder_center = (frame_data[:, left_shoulder_idx] + frame_data[:, right_shoulder_idx]) / 2
                        hip_center = (frame_data[:, left_hip_idx] + frame_data[:, right_hip_idx]) / 2
                    else:
                        # Для одного кадра
                        shoulder_center = (frame_data[left_shoulder_idx] + frame_data[right_shoulder_idx]) / 2
                        hip_center = (frame_data[left_hip_idx] + frame_data[right_hip_idx]) / 2

                    # Расчет COM согласно позиции из антропометрических данных
                    com = shoulder_center + segment_info.com_position * (hip_center - shoulder_center)
                    segment_com[segment_name] = com

                except (KeyError, TypeError) as e:
                    logger.warning(f"Не удалось рассчитать центр масс для туловища: {e}")
                    continue
            else:
                # Для остальных сегментов используем указанные индексы
                try:
                    if is_sequence:
                        # Для последовательности кадров
                        proximal = frame_data[:, start_idx]
                        distal = frame_data[:, end_idx]
                    else:
                        # Для одного кадра
                        proximal = frame_data[start_idx]
                        distal = frame_data[end_idx]

                    # Расчет COM согласно позиции из антропометрических данных
                    com = proximal + segment_info.com_position * (distal - proximal)
                    segment_com[segment_name] = com

                except (IndexError, ValueError) as e:
                    logger.warning(f"Не удалось рассчитать центр масс для сегмента {segment_name}: {e}")
                    continue

        return segment_com

    def calculate_total_body_com(
            self,
            skeleton_data: np.ndarray,
            return_segment_com: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Рассчитывает общий центр масс тела.

        Args:
            skeleton_data: Массив координат точек скелета shape (num_frames, num_landmarks, 3)
            return_segment_com: Если True, также возвращает центры масс сегментов

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
                - Координаты общего центра масс shape (num_frames, 3) или (3,)
                - Если return_segment_com=True, также словарь с центрами масс сегментов
        """
        # Проверяем, является ли skeleton_data многокадровым
        is_sequence = len(skeleton_data.shape) == 3 and skeleton_data.shape[0] > 1

        if is_sequence:
            num_frames = skeleton_data.shape[0]
            # Инициализируем массив для общего центра масс
            total_body_com = np.zeros((num_frames, 3))

            # Словарь для хранения центров масс сегментов
            all_segment_com = {}

            # Рассчитываем центр масс для каждого кадра
            for frame_idx in range(num_frames):
                # Рассчитываем центры масс сегментов для текущего кадра
                segment_com = self.calculate_segment_com(skeleton_data, frame_idx)

                # Сохраняем центры масс сегментов, если нужно
                if return_segment_com and frame_idx == 0:
                    all_segment_com = {
                        segment: np.zeros((num_frames, 3)) for segment in segment_com
                    }

                # Рассчитываем общий центр масс как взвешенную сумму центров масс сегментов
                weighted_com = np.zeros(3)
                total_mass = 0.0

                for segment_name, com in segment_com.items():
                    try:
                        segment_info = self.anthropometric_data.get_segment_info(segment_name)
                        weighted_com += com * segment_info.mass_percentage
                        total_mass += segment_info.mass_percentage

                        # Сохраняем центр масс сегмента, если нужно
                        if return_segment_com and segment_name in all_segment_com:
                            all_segment_com[segment_name][frame_idx] = com

                    except KeyError:
                        continue

                # Нормализуем общий центр масс
                if total_mass > 0:
                    total_body_com[frame_idx] = weighted_com / total_mass
                else:
                    logger.warning(f"Общая масса для кадра {frame_idx} равна 0. Невозможно рассчитать центр масс.")
                    total_body_com[frame_idx] = np.nan

            if return_segment_com:
                return total_body_com, all_segment_com
            else:
                return total_body_com
        else:
            # Для одного кадра
            # Рассчитываем центры масс сегментов
            segment_com = self.calculate_segment_com(skeleton_data)

            # Рассчитываем общий центр масс как взвешенную сумму центров масс сегментов
            weighted_com = np.zeros(3)
            total_mass = 0.0

            for segment_name, com in segment_com.items():
                try:
                    segment_info = self.anthropometric_data.get_segment_info(segment_name)
                    weighted_com += com * segment_info.mass_percentage
                    total_mass += segment_info.mass_percentage
                except KeyError:
                    continue

            # Нормализуем общий центр масс
            if total_mass > 0:
                total_body_com = weighted_com / total_mass
            else:
                logger.warning("Общая масса равна 0. Невозможно рассчитать центр масс.")
                total_body_com = np.full(3, np.nan)

            if return_segment_com:
                return total_body_com, segment_com
            else:
                return total_body_com


def calculate_center_of_mass(
        skeleton_data: np.ndarray,
        skeleton_model: SkeletonModel,
        anthropometric_data: Optional[AnthropometricData] = None,
        segment_mapping: Optional[Dict[str, Tuple[int, int]]] = None,
        return_segment_com: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Рассчитывает центр масс тела на основе скелетных данных.

    Args:
        skeleton_data: Массив координат точек скелета shape (num_frames, num_landmarks, 3)
        skeleton_model: Модель скелета
        anthropometric_data: Антропометрические данные. Если None, используются стандартные.
        segment_mapping: Соответствие сегментов точкам скелета. Если None, используется автоматическое сопоставление.
        return_segment_com: Если True, также возвращает центры масс сегментов

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
            - Координаты общего центра масс shape (num_frames, 3) или (3,)
            - Если return_segment_com=True, также словарь с центрами масс сегментов
    """
    calculator = CenterOfMassCalculator(
        skeleton_model=skeleton_model,
        anthropometric_data=anthropometric_data,
        segment_mapping=segment_mapping
    )

    return calculator.calculate_total_body_com(
        skeleton_data=skeleton_data,
        return_segment_com=return_segment_com
    )


if __name__ == "__main__":
    # Пример использования
    from openmocap.utils.logger import configure_logging, LogLevel

    configure_logging(LogLevel.DEBUG)

    # Создаем модель скелета для тестирования
    landmark_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                      "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee", "left_ankle", "right_ankle",
                      "left_foot_index", "right_foot_index"]
    connections = [(0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                   (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                   (15, 17), (16, 18)]

    skeleton_model = SkeletonModel(landmark_names=landmark_names, connections=connections)

    # Создаем тестовые данные скелета
    num_frames = 10
    skeleton_data = np.zeros((num_frames, len(landmark_names), 3))

    # Заполняем данные случайными значениями для теста
    for i, name in enumerate(landmark_names):
        skeleton_data[:, i, 0] = np.random.normal(i * 0.1, 0.01, num_frames)  # x
        skeleton_data[:, i, 1] = np.random.normal(i * 0.05, 0.01, num_frames)  # y
        skeleton_data[:, i, 2] = np.random.normal(i * 0.02, 0.01, num_frames)  # z

    # Создаем объект для расчета центра масс
    com_calculator = CenterOfMassCalculator(skeleton_model=skeleton_model)

    # Рассчитываем центр масс для всего тела
    total_body_com, segment_com = com_calculator.calculate_total_body_com(
        skeleton_data=skeleton_data,
        return_segment_com=True
    )

    print("Общий центр масс (первый кадр):")
    print(total_body_com[0])

    print("\nЦентры масс сегментов (первый кадр):")
    for segment_name, com in segment_com.items():
        print(f"  {segment_name}: {com[0]}")