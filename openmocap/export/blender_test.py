import bpy
import json
from mathutils import Quaternion, Vector


def load_animation_from_json(json_path):
    # Загружаем данные из JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Вывод структуры JSON для отладки
    print("Структура JSON-файла:")
    if 'metadata' in data:
        print(f"Метаданные: {data['metadata']}")
    if 'frames' in data:
        print(f"Количество кадров: {len(data['frames'])}")
        print(f"Пример данных первого кадра: {list(data['frames'][0].keys())[:5]} ...")
    else:
        print("Формат без 'frames', проверяем альтернативную структуру...")
        print(f"Верхнеуровневые ключи: {list(data.keys())}")

    # Удаляем все объекты в сцене
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Создаем мета-риг Rigify
    bpy.ops.object.armature_human_metarig_add()
    metarig = bpy.context.object
    metarig.name = "Rigify_Metarig"

    # Переходим в режим позы для настройки костей
    bpy.ops.object.mode_set(mode='POSE')

    # Устанавливаем режим вращения для всех костей
    for bone in metarig.pose.bones:
        bone.rotation_mode = 'QUATERNION'

    # Возвращаемся в объектный режим
    bpy.ops.object.mode_set(mode='OBJECT')

    # Настройки анимации
    bpy.context.scene.render.fps = 30

    # Установка длительности анимации
    frames_count = len(data['frames']) if 'frames' in data else data['metadata'].get('frames_count', 100)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frames_count - 1

    # Создаем действие
    action = bpy.data.actions.new(name="MocapAnimation")
    metarig.animation_data_create()
    metarig.animation_data.action = action

    # Маппинг костей
    bone_map = {
        'pelvis': 'spine',
        'spine': 'spine',
        'spine.001': 'spine.001',
        'spine.002': 'spine.002',
        'spine.003': 'spine.003',
        'spine.004': 'spine.004',
        'spine.005': 'spine.005',
        'spine.006': 'spine.006',

        'shoulder.L': 'shoulder.L',
        'upper_arm.L': 'upper_arm.L',
        'forearm.L': 'forearm.L',
        'hand.L': 'hand.L',

        'shoulder.R': 'shoulder.R',
        'upper_arm.R': 'upper_arm.R',
        'forearm.R': 'forearm.R',
        'hand.R': 'hand.R',

        'thigh.L': 'thigh.L',
        'shin.L': 'shin.L',
        'foot.L': 'foot.L',
        'heel.L': 'heel.L',
        'toe.L': 'toe.L',

        'thigh.R': 'thigh.R',
        'shin.R': 'shin.R',
        'foot.R': 'foot.R',
        'heel.R': 'heel.R',
        'toe.R': 'toe.R',

        # Добавляем обратную совместимость с предыдущим форматом
        'left_shoulder': 'upper_arm.L',
        'left_elbow': 'forearm.L',
        'right_hip': 'thigh.R',
        'right_knee': 'shin.R',
        'left_hip': 'thigh.L',
        'hips': 'spine',
        'right_shoulder': 'upper_arm.R',
        'spine_001': 'spine.001',
        'right_elbow': 'forearm.R',
        'left_knee': 'shin.L'
    }

    # Проверка доступных костей
    available_bones = [bone.name for bone in metarig.pose.bones]
    print("\nДоступные кости в метариге:")
    for bone_name in available_bones:
        print(f" - {bone_name}")

    # Счетчики для отладки
    keyframes_added = 0
    bones_animated = set()

    # Обрабатываем каждый кадр
    for frame_idx in range(frames_count):
        bpy.context.scene.frame_set(frame_idx)

        # Получаем данные текущего кадра
        frame_data = {}
        if 'frames' in data and frame_idx < len(data['frames']):
            frame_data = data['frames'][frame_idx]

        # Проверка на наличие данных в кадре
        if not frame_data:
            print(f"Нет данных для кадра {frame_idx}")
            continue

        # Применяем анимацию к каждой кости
        for json_bone, blender_bone in bone_map.items():
            if json_bone not in frame_data:
                continue

            joint_data = frame_data[json_bone]

            # Проверяем наличие кватерниона
            if 'quaternion' not in joint_data:
                continue

            # Получаем кость в метариге
            pose_bone = metarig.pose.bones.get(blender_bone)
            if not pose_bone:
                if frame_idx == 0:  # Выводим предупреждение только один раз
                    print(f"Предупреждение: Кость {blender_bone} не найдена в метариге")
                continue

            # Получаем кватернион
            quat_data = joint_data['quaternion']
            quat = None

            if isinstance(quat_data, dict):
                if all(k in quat_data for k in ['w', 'x', 'y', 'z']):
                    quat = Quaternion((
                        quat_data['w'],
                        quat_data['x'],
                        quat_data['y'],
                        quat_data['z']
                    ))
            elif isinstance(quat_data, list) and len(quat_data) == 4:
                quat = Quaternion((quat_data[0], quat_data[1], quat_data[2], quat_data[3]))

            if quat is None:
                print(f"Не удалось получить кватернион для {json_bone} в кадре {frame_idx}")
                continue

            # Применяем вращение
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

            # Обновляем счетчики
            keyframes_added += 1
            bones_animated.add(blender_bone)

            # Для первого кадра выводим примененный кватернион
            if frame_idx == 0:
                print(f"Применен кватернион к кости {blender_bone}: {quat}")

    # Выводим итоги
    print(f"\nИтоги анимации:")
    print(f"Добавлено {keyframes_added} ключевых кадров")
    print(f"Анимировано {len(bones_animated)} костей: {bones_animated}")

    # Сглаживаем анимацию
    for fcurve in action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'BEZIER'

    return metarig


def export_to_fbx(filepath, armature):
    """Экспортирует анимацию в FBX"""
    # Выделяем армату для экспорта
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Экспортируем в FBX
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=True,
        global_scale=1.0,
        path_mode='AUTO',
        axis_forward='-Z',
        axis_up='Y'
    )


# Путь к JSON-файлу с анимацией
json_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\export\fixed_reference_animation.json"
fbx_path = r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\mocap_animation.fbx"

try:
    # Импортируем анимацию
    metarig = load_animation_from_json(json_path)

    # Экспортируем в FBX
    export_to_fbx(fbx_path, metarig)

    print(f"Анимация успешно экспортирована в {fbx_path}")
except Exception as e:
    import traceback

    print(f"Ошибка: {str(e)}")
    print(traceback.format_exc())