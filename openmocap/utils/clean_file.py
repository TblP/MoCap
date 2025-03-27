import json


def load_and_filter_json(input_path, output_path):
    # Загрузка JSON файла
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Получаем полный список всех точек скелета из metadata
    all_joints = set(data['metadata']['joint_mapping'].keys())

    # Фильтруем кадры
    filtered_frames = []
    for frame in data['frames']:
        # Получаем точки, присутствующие в текущем кадре
        present_joints = set(frame.keys())

        # Если все точки присутствуют, сохраняем кадр
        if present_joints == all_joints:
            filtered_frames.append(frame)

    # Заменяем кадры в данных на отфильтрованные
    data['frames'] = filtered_frames

    # Обновляем количество кадров в metadata
    data['metadata']['frame_count'] = len(filtered_frames)
    data['metadata']['duration'] = len(filtered_frames) / data['metadata']['fps']

    # Сохраняем результат
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Исходное количество кадров: {len(data['frames']) + len(filtered_frames) - len(filtered_frames)}")
    print(f"Оставшееся количество кадров: {len(filtered_frames)}")


# Пример использования
input_json_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\refined_points_3d.json"  # Замените на путь к вашему файлу
output_json_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\clean_points_3d.json"  # Куда сохранить результат

load_and_filter_json(input_json_path, output_json_path)