# OpenMoCap

Открытая система захвата движения без маркеров 🚀✨

## Описание

OpenMoCap — это система захвата движения без маркеров, которая позволяет отслеживать движения тела, рук и лица с использованием обычных камер. Система использует компьютерное зрение и машинное обучение для обнаружения ключевых точек тела и последующей реконструкции 3D-координат.

## Возможности

- 📸 Калибровка нескольких камер с использованием шахматной доски
- 🧍 Отслеживание положения тела, рук и лица на 2D-видео с помощью MediaPipe
- 🧮 Триангуляция 2D-точек в 3D-пространство
- 📊 Пост-обработка данных (фильтрация, заполнение пропусков)
- 📊 Расчет центра масс и других биомеханических параметров
- 🎨 Экспорт данных в различные форматы

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourorganization/openmocap.git
cd openmocap

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

## Быстрый старт

```python
import openmocap as om

# Создание сессии захвата движения
session = om.Session("my_session")

# Калибровка камер
calibrator = om.MultiCameraCalibrator()
calibrator.calibrate_from_videos(["camera1.mp4", "camera2.mp4", "camera3.mp4"])

# Отслеживание точек на видео
tracker = om.MediaPipeTracker()
points_2d = tracker.track_videos(["camera1.mp4", "camera2.mp4", "camera3.mp4"])

# Реконструкция 3D-координат
reconstructor = om.Reconstructor(calibrator.get_camera_parameters())
points_3d = reconstructor.triangulate(points_2d)

# Пост-обработка данных
processor = om.PostProcessor()
filtered_points_3d = processor.process(points_3d, 
                                     filter_type="butterworth", 
                                     cutoff_freq=7.0)

# Экспорт данных
exporter = om.Exporter()
exporter.to_csv(filtered_points_3d, "motion_capture_data.csv")
exporter.to_blender(filtered_points_3d, "motion_capture_data.blend")
```
