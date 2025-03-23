import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv(r"C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\refined_points_3d.csv",
                 comment='#')

# Вариант 1: Удаление строк, где слишком много NaN
# Например, если больше 50% точек отсутствует
threshold = 33 * 3 * 0.5  # 50% от всех координат
filtered_df = df.dropna(thresh=threshold)  # Удаляем строки, где меньше threshold не-NaN значений

# Сохранение результатов
filtered_df.to_csv(r'C:\Users\vczyp\PycharmProjects\MoCap\openmocap\core\output_data\filtered_no_nan_frames.csv', index=False)
