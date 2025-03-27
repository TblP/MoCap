#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Исправленный скрипт для визуализации скелета из JSON файла.
С возможностью покадрового переключения и клавиатурным управлением.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d  # Добавляем модуль proj3d для проецирования

# Соединения между точками MediaPipe Pose
POSE_CONNECTIONS = [
    # Туловище
    (23, 11), (23, 12),  # левое бедро - левое/правое плечо
    (24, 11), (24, 12),  # правое бедро - левое/правое плечо
    (23, 24),  # соединение бедер (pelvis/hips)
    (11, 12),  # соединение плеч (chest)

    # Левая рука
    (11, 13),  # левое плечо - левый локоть
    (13, 15),  # левый локоть - левое запястье
    # Пальцы левой руки
    (15, 17),  # запястье - мизинец
    (15, 19),  # запястье - указательный
    (15, 21),  # запястье - большой

    # Правая рука
    (12, 14),  # правое плечо - правый локоть
    (14, 16),  # правый локоть - правое запястье
    # Пальцы правой руки
    (16, 18),  # запястье - мизинец
    (16, 20),  # запястье - указательный
    (16, 22),  # запястье - большой

    # Левая нога
    (23, 25),  # левое бедро - левое колено
    (25, 27),  # левое колено - левая лодыжка
    (27, 31),  # левая лодыжка - левый носок
    (27, 29),  # левая лодыжка - левая пятка

    # Правая нога
    (24, 26),  # правое бедро - правое колено
    (26, 28),  # правое колено - правая лодыжка
    (28, 32),  # правая лодыжка - правый носок
    (28, 30),  # правая лодыжка - правая пятка

    # Голова и шея
    (0, 11), (0, 12),  # нос - левое/правое плечо
]

# Имена точек MediaPipe Pose
POSE_LANDMARK_NAMES = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
]


def load_json(filepath):
    """Загружает данные из JSON файла."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_points_from_json(data):
    """Извлекает 3D точки из JSON данных."""
    if 'frames' in data:
        frames = data['frames']
        if not frames:
            raise ValueError("В JSON файле нет кадров")

        num_frames = len(frames)
        num_landmarks = len(POSE_LANDMARK_NAMES)

        # Создаем массив для хранения точек
        points = np.full((num_frames, num_landmarks, 3), np.nan)

        for frame_idx, frame in enumerate(frames):
            for landmark_idx, name in enumerate(POSE_LANDMARK_NAMES):
                if name in frame:
                    landmarks = frame[name]
                    points[frame_idx, landmark_idx, 0] = landmarks.get('x', np.nan)
                    points[frame_idx, landmark_idx, 1] = landmarks.get('y', np.nan)
                    points[frame_idx, landmark_idx, 2] = landmarks.get('z', np.nan)
    else:
        # Пробуем другие форматы
        try:
            metadata = data.get('metadata', {})
            frames_data = []

            # Проверяем, есть ли данные кадров в списке или другой структуре
            if isinstance(data, list):
                frames_data = data
            elif 'frames' in metadata:
                frames_data = metadata['frames']

            if frames_data:
                num_frames = len(frames_data)
                num_landmarks = len(POSE_LANDMARK_NAMES)

                points = np.full((num_frames, num_landmarks, 3), np.nan)

                for frame_idx, frame in enumerate(frames_data):
                    for landmark_idx, name in enumerate(POSE_LANDMARK_NAMES):
                        if name in frame:
                            landmarks = frame[name]
                            points[frame_idx, landmark_idx, 0] = landmarks.get('x', np.nan)
                            points[frame_idx, landmark_idx, 1] = landmarks.get('y', np.nan)
                            points[frame_idx, landmark_idx, 2] = landmarks.get('z', np.nan)
            else:
                raise ValueError("Не удалось найти данные кадров")

        except Exception as e:
            raise ValueError(f"Неподдерживаемый формат JSON: {e}")

    return points


class SkeletonViewer:
    def __init__(self, points):
        self.points = points
        self.current_frame = 0
        self.total_frames = points.shape[0]

        # Режим отображения координат
        self.label_mode = "all"  # "all", "none", "selected"
        self.selected_point = None
        self.show_coords = True

        # Создаем фигуру
        self.fig = plt.figure(figsize=(12, 8))

        # Создаем 3D оси
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Настраиваем оси
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Скелет')

        # Настраиваем угол обзора
        self.ax.view_init(elev=20, azim=-60)

        # Определяем границы для всех кадров
        valid_points = points[~np.isnan(points).any(axis=2)]
        if len(valid_points) > 0:
            min_vals = np.min(valid_points, axis=0)
            max_vals = np.max(valid_points, axis=0)

            center = (min_vals + max_vals) / 2
            size = max(max_vals - min_vals) * 0.7

            self.ax.set_xlim([center[0] - size, center[0] + size])
            self.ax.set_ylim([center[1] - size, center[1] + size])
            self.ax.set_zlim([center[2] - size, center[2] + size])

        # Добавляем информацию о текущем кадре
        self.frame_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes)

        # Область для кнопок и слайдера
        self.fig.subplots_adjust(bottom=0.25)

        # Добавляем слайдер для выбора кадра
        self.slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.frame_slider = Slider(
            self.slider_ax, 'Кадр', 0, self.total_frames - 1,
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self.update_slider)

        # Добавляем кнопки
        self.prev_button_ax = plt.axes([0.2, 0.02, 0.1, 0.04])
        self.next_button_ax = plt.axes([0.7, 0.02, 0.1, 0.04])
        self.prev_button = Button(self.prev_button_ax, '< Пред')
        self.next_button = Button(self.next_button_ax, 'След >')
        self.prev_button.on_clicked(self.prev_frame)
        self.next_button.on_clicked(self.next_frame)

        # Добавляем переключатель режима отображения координат
        self.label_ax = plt.axes([0.85, 0.75, 0.12, 0.15])
        self.label_radio = RadioButtons(
            self.label_ax,
            ('Все', 'Выбранная', 'Нет'),
            active=0
        )
        self.label_radio.on_clicked(self.set_label_mode)

        # Инициализируем элементы для отображения
        self.lines = []
        for _ in POSE_CONNECTIONS:
            line, = self.ax.plot([], [], [], 'b-', linewidth=2)
            self.lines.append(line)

        self.dots = []
        self.labels = []
        for _ in range(len(POSE_LANDMARK_NAMES)):
            dot, = self.ax.plot([], [], [], 'ro', markersize=5)
            label = self.ax.text(0, 0, 0, '', fontsize=7, color='black')
            self.dots.append(dot)
            self.labels.append(label)

        # Подключаем обработчик клика для выбора точки
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Подключаем обработчик клавиш
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Отображаем первый кадр
        self.update_frame()

    def on_key(self, event):
        """Обработчик нажатия клавиш."""
        if event.key == 'right':
            self.next_frame(None)
        elif event.key == 'left':
            self.prev_frame(None)
        elif event.key == 'c':
            # Циклически переключаем режим отображения координат
            if self.label_mode == "all":
                self.label_mode = "selected"
                self.label_radio.set_active(1)
            elif self.label_mode == "selected":
                self.label_mode = "none"
                self.label_radio.set_active(2)
            else:
                self.label_mode = "all"
                self.label_radio.set_active(0)
            self.update_frame()

    def on_click(self, event):
        """Обработчик клика мыши для выбора точки."""
        if event.inaxes != self.ax:
            return

        if event.button != 1:  # Только левая кнопка мыши
            return

        # Получаем текущие данные
        frame_data = self.points[self.current_frame]

        # Конвертируем 2D координаты клика в 3D
        # Используем proj3d.proj_transform вместо метода project
        closest_point = None
        min_dist = float('inf')

        # Для каждой точки скелета
        for i, point in enumerate(frame_data):
            if np.isnan(point).any():
                continue

            # Проецируем 3D точку на 2D экран с помощью proj3d
            x3d, y3d, z3d = point
            x2d, y2d, _ = proj3d.proj_transform(x3d, y3d, z3d, self.ax.get_proj())

            # Вычисляем расстояние в пикселях
            dist = np.sqrt((event.x - self.fig.canvas.renderer.points_to_pixels(x2d)) ** 2 +
                           (event.y - self.fig.canvas.renderer.points_to_pixels(y2d)) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_point = i

        # Проверяем, достаточно ли близко точка
        if min_dist < 50:  # Пороговое значение в пикселях (увеличено для лучшего захвата)
            self.selected_point = closest_point
            print(f"Выбрана точка {closest_point}: {POSE_LANDMARK_NAMES[closest_point]}")

            # Обновляем отображение
            self.update_frame()

    def set_label_mode(self, label):
        """Устанавливает режим отображения координат."""
        if label == 'Все':
            self.label_mode = "all"
        elif label == 'Выбранная':
            self.label_mode = "selected"
        elif label == 'Нет':
            self.label_mode = "none"

        self.update_frame()

    def update_slider(self, val):
        """Обновляет текущий кадр при изменении слайдера."""
        self.current_frame = int(val)
        self.update_frame()

    def prev_frame(self, event):
        """Переходит к предыдущему кадру."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set_val(self.current_frame)
            self.update_frame()

    def next_frame(self, event):
        """Переходит к следующему кадру."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set_val(self.current_frame)
            self.update_frame()

    def update_frame(self):
        """Обновляет отображение скелета для текущего кадра."""
        frame_data = self.points[self.current_frame]

        # Очищаем предыдущие элементы
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])

        for dot in self.dots:
            dot.set_data([], [])
            dot.set_3d_properties([])

        for label in self.labels:
            label.set_text('')

        # Обновляем соединения
        for i, (connection, line) in enumerate(zip(POSE_CONNECTIONS, self.lines)):
            start_idx, end_idx = connection

            # Проверяем, что индексы в пределах и точки не содержат NaN
            if (start_idx < frame_data.shape[0] and end_idx < frame_data.shape[0] and
                    not np.isnan(frame_data[start_idx]).any() and
                    not np.isnan(frame_data[end_idx]).any()):
                start_point = frame_data[start_idx]
                end_point = frame_data[end_idx]

                line.set_data([start_point[0], end_point[0]], [start_point[1], end_point[1]])
                line.set_3d_properties([start_point[2], end_point[2]])

        # Обновляем точки и метки
        for i, (dot, label) in enumerate(zip(self.dots, self.labels)):
            if i < frame_data.shape[0] and not np.isnan(frame_data[i]).any():
                x, y, z = frame_data[i]

                # Отображаем точку
                dot.set_data([x], [y])
                dot.set_3d_properties([z])

                # Определяем, нужно ли отображать координаты
                if self.label_mode == "all" or (self.label_mode == "selected" and i == self.selected_point):
                    name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"point_{i}"
                    coords_text = f"{name}\n({x:.1f}, {y:.1f}, {z:.1f})"

                    label.set_text(coords_text)
                    label.set_position((x, y))
                    label.set_3d_properties(z)

                # Выделяем выбранную точку
                if i == self.selected_point:
                    dot.set_color('green')
                    dot.set_markersize(8)
                else:
                    dot.set_color('red')
                    dot.set_markersize(5)

        # Обновляем текст с информацией о кадре
        self.frame_text.set_text(f'Кадр: {self.current_frame + 1}/{self.total_frames}')

        # Перерисовываем график
        self.fig.canvas.draw_idle()

    def show(self):
        """Отображает визуализацию."""
        # Добавляем инструкции по управлению
        instructions = (
            "Управление:\n"
            "- Стрелки влево/вправо: переключение кадров\n"
            "- Клавиша 'C': изменить режим отображения координат\n"
            "- Клик на точке: выбрать точку\n"
            "- Прокрутка: масштабирование\n"
            "- Правая кнопка мыши: вращение 3D сцены"
        )
        self.fig.text(0.01, 0.01, instructions, fontsize=8)

        plt.show()


def main():
    # Просим пользователя ввести путь к файлу
    filepath = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\refined_points_3d.json"

    try:
        # Загружаем данные
        data = load_json(filepath)

        # Извлекаем точки
        points = extract_points_from_json(data)

        print(f"Загружено {points.shape[0]} кадров с {points.shape[1]} точками")

        # Создаем и запускаем визуализатор
        viewer = SkeletonViewer(points)
        viewer.show()

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()