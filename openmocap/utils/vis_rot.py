import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


def load_skeleton_data(json_path):
    """Загружает данные скелета из JSON файла"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_base_skeleton():
    """Создает базовый скелет в T-позе (метры)"""
    skeleton = {
        'nose': {'pos': [0, 0.2, 0], 'children': ['left_eye', 'right_eye']},
        'left_eye': {'pos': [-0.05, 0.2, 0], 'children': ['left_ear']},
        'right_eye': {'pos': [0.05, 0.2, 0], 'children': ['right_ear']},
        'left_ear': {'pos': [-0.1, 0.2, 0], 'children': ['left_shoulder']},
        'right_ear': {'pos': [0.1, 0.2, 0], 'children': ['right_shoulder']},
        'left_shoulder': {'pos': [-0.2, 0.15, 0], 'children': ['left_elbow', 'left_hip']},
        'right_shoulder': {'pos': [0.2, 0.15, 0], 'children': ['right_elbow', 'right_hip']},
        'left_elbow': {'pos': [-0.4, 0.15, 0], 'children': ['left_wrist']},
        'right_elbow': {'pos': [0.4, 0.15, 0], 'children': ['right_wrist']},
        'left_wrist': {'pos': [-0.6, 0.15, 0], 'children': ['left_pinky', 'left_index', 'left_thumb']},
        'right_wrist': {'pos': [0.6, 0.15, 0], 'children': ['right_pinky', 'right_index', 'right_thumb']},
        'left_hip': {'pos': [-0.2, 0, 0], 'children': ['left_knee']},
        'right_hip': {'pos': [0.2, 0, 0], 'children': ['right_knee']},
        'left_knee': {'pos': [-0.2, -0.3, 0], 'children': ['left_ankle']},
        'right_knee': {'pos': [0.2, -0.3, 0], 'children': ['right_ankle']},
        'left_ankle': {'pos': [-0.2, -0.6, 0], 'children': ['left_heel', 'left_foot_index']},
        'right_ankle': {'pos': [0.2, -0.6, 0], 'children': ['right_heel', 'right_foot_index']},
        # Конечности (для полноты)
        'left_pinky': {'pos': [-0.62, 0.14, -0.02], 'children': []},
        'right_pinky': {'pos': [0.62, 0.14, -0.02], 'children': []},
        'left_index': {'pos': [-0.62, 0.16, 0.02], 'children': []},
        'right_index': {'pos': [0.62, 0.16, 0.02], 'children': []},
        'left_thumb': {'pos': [-0.58, 0.14, 0.04], 'children': []},
        'right_thumb': {'pos': [0.58, 0.14, 0.04], 'children': []},
        'left_heel': {'pos': [-0.18, -0.62, -0.05], 'children': []},
        'right_heel': {'pos': [0.18, -0.62, -0.05], 'children': []},
        'left_foot_index': {'pos': [-0.22, -0.62, 0.05], 'children': []},
        'right_foot_index': {'pos': [0.22, -0.62, 0.05], 'children': []},
    }
    return skeleton


def apply_rotations(skeleton, rotations):
    """Применяет повороты к скелету с коррекцией осей"""
    rotated_skeleton = {k: {'pos': v['pos'].copy(), 'children': v['children']}
                        for k, v in skeleton.items()}

    for joint, data in rotations.items():
        if joint not in rotated_skeleton:
            continue

        if 'quaternion' in data:
            q = data['quaternion']
            rot = R.from_quat([q['x'], q['y'], q['z'], q['w']])
        elif 'euler' in data:
            euler = data['euler']
            rot = R.from_euler('xyz', [euler['x'], euler['y'], euler['z']], degrees=True)
        else:
            continue

        # КОРРЕКЦИИ ДЛЯ РАЗНЫХ СУСТАВОВ
        if 'left' in joint:
            # Коррекция для левой стороны тела
            rot = rot * R.from_euler('z', 180, degrees=True)
        elif 'right' in joint:
            # Коррекция для правой стороны тела
            rot = rot * R.from_euler('x', 180, degrees=True)

        if joint in ['left_knee', 'right_knee']:
            # Дополнительная коррекция для коленей
            rot = rot * R.from_euler('x', 180, degrees=True)

        # Применяем поворот ко всем дочерним костям
        for child in rotated_skeleton[joint]['children']:
            if child not in rotated_skeleton:
                continue

            offset = np.array(rotated_skeleton[child]['pos']) - np.array(rotated_skeleton[joint]['pos'])
            rotated_offset = rot.apply(offset)
            rotated_skeleton[child]['pos'] = np.array(rotated_skeleton[joint]['pos']) + rotated_offset

            # Рекурсивно обновляем всех потомков
            stack = rotated_skeleton[child]['children']
            while stack:
                current = stack.pop()
                if current not in rotated_skeleton:
                    continue

                parent_pos = rotated_skeleton[joint]['pos']
                current_pos = rotated_skeleton[current]['pos']
                new_pos = parent_pos + rot.apply(np.array(current_pos) - np.array(parent_pos))
                rotated_skeleton[current]['pos'] = new_pos
                stack.extend(rotated_skeleton[current]['children'])

    return rotated_skeleton


def plot_skeleton(ax, skeleton):
    """Отрисовывает скелет на 3D графике"""
    ax.clear()

    # Настройки отображения
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Отрисовываем кости
    for bone, data in skeleton.items():
        pos = data['pos']
        ax.scatter(*pos, color='red', s=20)

        for child in data['children']:
            if child in skeleton:
                child_pos = skeleton[child]['pos']
                ax.plot([pos[0], child_pos[0]],
                        [pos[1], child_pos[1]],
                        [pos[2], child_pos[2]], 'b-', linewidth=2)


def animate_skeleton(json_path):
    """Создает анимацию скелета"""
    data = load_skeleton_data(json_path)
    base_skeleton = create_base_skeleton()

    # Конвертируем позиции в numpy массивы
    for bone in base_skeleton.values():
        bone['pos'] = np.array(bone['pos'])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        rotations = data['frames'][frame]
        rotated_skeleton = apply_rotations(base_skeleton, rotations)
        plot_skeleton(ax, rotated_skeleton)
        ax.set_title(f'Frame {frame + 1}/{len(data["frames"])} (FPS: {data["metadata"]["fps"]})')
        return ax,

    ani = FuncAnimation(fig, update, frames=len(data['frames']),
                        interval=1000 / data['metadata']['fps'], blit=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    json_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\joint_rotations.json"  # Укажите путь к вашему файлу
    animate_skeleton(json_path)