import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parse_csv(file_path):
    """
    Парсит CSV-файл с 3D-координатами точек.
    Ожидаемый формат:
    frame, landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, landmark_1_y, landmark_1_z, ...
    """
    print(f"Чтение файла: {file_path}")

    # Чтение CSV-файла
    df = pd.read_csv(file_path)

    # Получаем количество кадров
    num_frames = len(df)

    # Вычисляем количество ориентиров по заголовкам (исключая 'frame')
    # Каждый ориентир имеет 3 координаты (x, y, z)
    headers = df.columns.tolist()
    headers = [h for h in headers if h != 'frame']
    num_landmarks = len(headers) // 3

    print(f"Обнаружено {num_frames} кадров и {num_landmarks} ориентиров")

    # Создаем трехмерный массив для хранения координат
    # Размерность: (num_frames, num_landmarks, 3)
    motion_data = np.zeros((num_frames, num_landmarks, 3))

    # Заполняем массив координатами
    for frame in range(num_frames):
        for landmark in range(num_landmarks):
            x_col = f"landmark_{landmark}_x"
            y_col = f"landmark_{landmark}_y"
            z_col = f"landmark_{landmark}_z"

            # Проверяем, используется ли другой формат заголовков
            if x_col not in df.columns:
                # Пробуем другие возможные форматы, например, из MediaPipe (pose_landmark_name_x)
                for col in df.columns:
                    if col.endswith(f"_{landmark}_x"):
                        x_col = col
                    if col.endswith(f"_{landmark}_y"):
                        y_col = col
                    if col.endswith(f"_{landmark}_z"):
                        z_col = col

            # Если столбцы все еще не найдены, просто берем их по порядку
            if x_col not in df.columns:
                x_col = headers[landmark * 3]
                y_col = headers[landmark * 3 + 1]
                z_col = headers[landmark * 3 + 2]

            # Читаем координаты
            try:
                motion_data[frame, landmark, 0] = df.iloc[frame][x_col]
                motion_data[frame, landmark, 1] = df.iloc[frame][y_col]
                motion_data[frame, landmark, 2] = df.iloc[frame][z_col]
            except KeyError as e:
                print(f"Ошибка при чтении координат для ориентира {landmark}: {e}")
                print(f"Доступные столбцы: {df.columns.tolist()}")
                raise

    return motion_data


def get_connections(model_type="mediapipe"):
    """
    Возвращает список соединений между ориентирами для визуализации скелета.
    """
    if model_type == "mediapipe":
        # MediaPipe Pose connections (33 ориентира)
        return [
            # Лицо
            (0, 1), (1, 2), (2, 3), (3, 7),  # Левая сторона лица
            (0, 4), (4, 5), (5, 6), (6, 8),  # Правая сторона лица
            (9, 10),  # Рот

            # Туловище
            (11, 12), (11, 23), (12, 24), (23, 24),  # Плечи и бедра

            # Руки
            (11, 13), (13, 15),  # Левая рука
            (12, 14), (14, 16),  # Правая рука

            # Пальцы
            (15, 17), (15, 19), (15, 21),  # Левая кисть
            (16, 18), (16, 20), (16, 22),  # Правая кисть

            # Ноги
            (23, 25), (25, 27), (27, 29), (27, 31),  # Левая нога
            (24, 26), (26, 28), (28, 30), (28, 32)  # Правая нога
        ]
    else:
        # Простой скелет (для других моделей)
        return []


def animate_skeleton(motion_data, connections=None, interval=50, rotate=True, save_path=None):
    """
    Создает 3D-анимацию движения скелета.

    Args:
        motion_data: Массив координат (num_frames, num_landmarks, 3)
        connections: Список пар индексов ориентиров, которые нужно соединить линиями
        interval: Интервал между кадрами анимации в миллисекундах
        rotate: Вращать ли вид вокруг центра во время анимации
        save_path: Путь для сохранения анимации (если None, анимация только отображается)
    """
    num_frames, num_landmarks, _ = motion_data.shape

    if connections is None:
        connections = get_connections()

    # Создаем фигуру и оси
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Определяем границы осей
    min_val = np.nanmin(motion_data)
    max_val = np.nanmax(motion_data)
    center = (min_val + max_val) / 2

    # Находим крайние точки для осей, чтобы скелет был виден полностью
    x_min, y_min, z_min = np.nanmin(motion_data, axis=(0, 1))
    x_max, y_max, z_max = np.nanmax(motion_data, axis=(0, 1))

    # Добавляем запас для лучшей видимости
    x_center, y_center, z_center = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2 * 1.2

    # Инициализируем объекты для анимации
    points = ax.plot([], [], [], 'ro', markersize=4)[0]
    lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]

    # Устанавливаем равные масштабы для осей
    ax.set_box_aspect([1, 1, 1])

    # Функция инициализации анимации
    def init():
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Анимация скелета OpenMoCap')
        return [points] + lines

    # Функция обновления для каждого кадра анимации
    def update(frame):
        # Обновляем позиции точек
        xs = motion_data[frame, :, 0]
        ys = motion_data[frame, :, 1]
        zs = motion_data[frame, :, 2]

        # Исключаем NaN значения
        valid = ~np.isnan(xs) & ~np.isnan(ys) & ~np.isnan(zs)
        points.set_data(xs[valid], ys[valid])
        points.set_3d_properties(zs[valid])

        # Обновляем линии для соединений
        for i, (start, end) in enumerate(connections):
            if (start < num_landmarks and end < num_landmarks and
                    not np.isnan(motion_data[frame, start]).any() and
                    not np.isnan(motion_data[frame, end]).any()):
                lines[i].set_data([motion_data[frame, start, 0], motion_data[frame, end, 0]],
                                  [motion_data[frame, start, 1], motion_data[frame, end, 1]])
                lines[i].set_3d_properties([motion_data[frame, start, 2], motion_data[frame, end, 2]])
            else:
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])

        # Вращаем вид, если нужно
        if rotate:
            ax.view_init(elev=30, azim=frame % 360)

        # Отображаем номер кадра
        ax.set_title(f'Анимация скелета OpenMoCap - Кадр {frame + 1}/{num_frames}')

        return [points] + lines

    # Создаем анимацию
    anim = FuncAnimation(fig, update, frames=num_frames,
                         init_func=init, blit=False, interval=interval)

    # Сохраняем анимацию, если указан путь
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
        print(f"Анимация сохранена в {save_path}")

    plt.tight_layout()
    plt.show()

    return anim


def main():
    # Обработка аргументов командной строки
    #parser = argparse.ArgumentParser(description='Воспроизведение анимации из CSV-файла')
    #parser.add_argument('csv_file', help='Путь к CSV-файлу с данными движения')
    #parser.add_argument('--save', help='Путь для сохранения анимации (опционально)')
    #parser.add_argument('--interval', type=int, default=50, help='Интервал между кадрами (мс)')
    #parser.add_argument('--no-rotate', action='store_true', help='Отключить вращение камеры')

    #args = parser.parse_args()

    # Анализ CSV-файла
    motion_data = parse_csv(r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\filtered_no_nan_frames.csv")

    # Запуск анимации
    animate_skeleton(
        motion_data,
        #interval=args.interval,
        #rotate=not args.no_rotate,
        #save_path=args.save
    )


if __name__ == "__main__":
    main()