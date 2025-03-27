import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def visualize_skeleton_animation(json_data, skeleton_model_path, start_frame=0, end_frame=None):
    # Загружаем модель скелета
    skeleton_model = load_json(skeleton_model_path)
    connections = skeleton_model['connections']

    # Если end_frame не указан, берем все кадры
    frames = json_data['frames'][start_frame:end_frame]
    if not frames:
        print("No frames found in the specified range")
        return

    # Создаем 3D график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Функция для обновления кадра анимации
    def update(frame_idx):
        ax.clear()
        frame = frames[frame_idx % len(frames)]  # Используем модуль для бесконечного цикла

        # Собираем позиции всех суставов
        joint_positions = {}
        x_vals, y_vals, z_vals = [], [], []

        for joint, pos in frame.items():
            joint_positions[joint] = np.array([pos['x'], pos['y'], pos['z']])
            x_vals.append(pos['x'])
            y_vals.append(pos['y'])
            z_vals.append(pos['z'])

        # Автоматическое масштабирование осей
        if x_vals and y_vals and z_vals:
            ax.set_xlim(min(x_vals) - 0.1, max(x_vals) + 0.1)
            ax.set_ylim(min(y_vals) - 0.1, max(y_vals) + 0.1)
            ax.set_zlim(min(z_vals) - 0.1, max(z_vals) + 0.1)

        ax.set_title(f'Frame {start_frame + (frame_idx % len(frames))}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Рисуем точки суставов
        for joint, pos in joint_positions.items():
            ax.scatter(*pos, s=50, label=joint)

        # Рисуем соединения между суставами
        for connection in connections:
            start_joint = connection['start_name']
            end_joint = connection['end_name']

            if start_joint in joint_positions and end_joint in joint_positions:
                start_pos = joint_positions[start_joint]
                end_pos = joint_positions[end_joint]

                ax.plot([start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]],
                        [start_pos[2], end_pos[2]],
                        'b-', linewidth=2)

        if frame_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:10], labels[:10], loc='upper right', bbox_to_anchor=(1.15, 1))

    # Создаем анимацию с бесконечным повторением
    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames) * 3,  # Проиграть 3 цикла (можно любое число)
        interval=200,
        repeat=True,
        repeat_delay=500  # Пауза 500 мс между повторениями
    )

    plt.tight_layout()
    plt.show()
    return ani


# Пример использования
data_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\refined_points_3d.json"
skeleton_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\skeleton_model.json"
json_data = load_json(data_path)
visualize_skeleton_animation(json_data, skeleton_path, start_frame=120, end_frame=210)