import json
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def calculate_relative_rotation(v_start, v_end):
    """Вычисляет относительный поворот между двумя векторами"""
    if np.allclose(v_start, v_end):
        return R.identity()

    v1 = v_start / (np.linalg.norm(v_start) + 1e-10)
    v2 = v_end / (np.linalg.norm(v_end) + 1e-10)

    axis = np.cross(v1, v2)
    axis_norm = axis / (np.linalg.norm(axis) + 1e-10)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    return R.from_rotvec(axis_norm * angle)


def process_animation_fixed_reference(json_data, skeleton_model_path, start_frame, end_frame, amplitude_scale=1.0,
                                      fps=30.0):
    skeleton_model = load_json(skeleton_model_path)
    connections = skeleton_model['connections']
    frames = json_data['frames'][start_frame:end_frame]

    # Получение fps из метаданных, если они есть, иначе используем переданное значение
    if 'metadata' in json_data and 'fps' in json_data['metadata']:
        fps = json_data['metadata']['fps']

    # Фиксируем начальные позиции и ориентации
    initial_positions = {joint: np.array([pos['x'], pos['y'], pos['z']])
                         for joint, pos in frames[0].items()}

    # Для каждого сустава вычисляем поворот относительно начального положения
    quaternions_per_frame = []

    for frame in frames:
        frame_rotations = {}
        for joint in initial_positions:
            if joint in frame:
                current_pos = np.array([frame[joint]['x'], frame[joint]['y'], frame[joint]['z']])
                # Вычисляем поворот от начальной позиции к текущей
                rot = calculate_relative_rotation(initial_positions[joint], current_pos)
                frame_rotations[joint] = rot.as_quat()
        quaternions_per_frame.append(frame_rotations)

    # Сохраняем результат с таймингом
    output = {
        "metadata": {
            "format": "fixed_reference_animation",
            "frames_count": len(frames),
            "fps": fps,
            "duration": len(frames) / fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "initial_positions": {k: list(v) for k, v in initial_positions.items()}
        },
        "frames": []
    }

    for i, frame_rot in enumerate(quaternions_per_frame):
        frame_time = i / fps  # Время в секундах от начала анимации
        frame_data = {
            "time": frame_time,
            "frame_index": start_frame + i,
            "joints": {}
        }

        for joint, quat in frame_rot.items():
            frame_data["joints"][joint] = {
                "quaternion": {
                    "w": float(quat[3]),
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2])
                }
            }
        output["frames"].append(frame_data)

    output_path = "fixed_reference_animation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Анимация сохранена в {output_path}")

    # Визуализация
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        current_positions = {}

        if frame_idx == 0:
            current_positions = initial_positions
        else:
            for joint in initial_positions:
                rot = R.from_quat(quaternions_per_frame[frame_idx][joint])
                current_positions[joint] = rot.apply(initial_positions[joint])

        # Отрисовка
        x_vals = [pos[0] for pos in current_positions.values()]
        y_vals = [pos[1] for pos in current_positions.values()]
        z_vals = [pos[2] for pos in current_positions.values()]

        ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
        ax.set_zlim(min(z_vals) - 1, max(z_vals) + 1)

        for joint, pos in current_positions.items():
            ax.scatter(*pos, s=50, label=joint)

        for conn in connections:
            start, end = conn['start_name'], conn['end_name']
            if start in current_positions and end in current_positions:
                ax.plot(*zip(current_positions[start], current_positions[end]), 'b-', linewidth=2)

        # Добавляем информацию о времени
        frame_time = frame_idx / fps
        ax.set_title(f'Frame {start_frame + frame_idx} (Time: {frame_time:.2f}s)')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=True)
    plt.tight_layout()
    plt.show()

    return output


# Параметры
data_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\clean_points_3d.json"
skeleton_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\skeleton_model.json"
start_frame = 0
end_frame = 260
fps = 30.0  # Можно указать FPS вручную или получить из JSON метаданных

# Запуск
json_data = load_json(data_path)
animation_data = process_animation_fixed_reference(
    json_data,
    skeleton_path,
    start_frame,
    end_frame,
    amplitude_scale=1.3,
    fps=fps
)