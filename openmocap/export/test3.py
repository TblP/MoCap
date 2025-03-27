import json
import os

# Путь к исходному JSON файлу
input_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\export\fixed_reference_animation.json"

# Путь для сохранения модифицированного JSON
output_directory = os.path.dirname(input_path)
output_filename = os.path.splitext(os.path.basename(input_path))[0] + "_scaled.json"
output_path = os.path.join(output_directory, output_filename)

# Загружаем исходный JSON
with open(input_path, 'r') as f:
    data = json.load(f)

# Множитель масштабирования
scale = 15.0

# Масштабируем initial_positions
if 'metadata' in data and 'initial_positions' in data['metadata']:
    for joint_name, position in data['metadata']['initial_positions'].items():
        if isinstance(position, list) and len(position) == 3:
            data['metadata']['initial_positions'][joint_name] = [pos * scale for pos in position]

# Масштабируем кватернионы x, y, z во всех кадрах
for frame in data['frames']:
    if 'joints' in frame:
        for joint_name, joint_data in frame['joints'].items():
            if 'quaternion' in joint_data:
                quat = joint_data['quaternion']
                if isinstance(quat, dict) and all(k in quat for k in ['w', 'x', 'y', 'z']):
                    # Масштабируем только компоненты x, y, z
                    quat['x'] *= scale
                    quat['y'] *= scale
                    quat['z'] *= scale
                    quat['w'] *= -1
# Сохраняем модифицированный JSON
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Файл обработан и сохранен как: {output_path}")