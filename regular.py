import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Имя файла с данными колебаний
data_file          = 'data_fixed.txt'
original_data_file = 'data.txt'

# Параметры для анализа
delta_time = 5  # Исключаемый диапазон времени в начале и конце, в секундах
zero_threshold = 0.05  # Порог для определения перехода через ноль
min_time_between_zero_crossings = 0.2 # Минимальное время между переходами через ноль

# ---------------------------------------------------------------------------------
# Блок 1: Загрузка и обработка данных
# ---------------------------------------------------------------------------------

try:
    with open(original_data_file, 'r', encoding='utf-8') as infile, open(data_file, 'w', encoding='utf-8') as outfile:
        header_ = infile.readline()
        for line in infile:
            fixed_line = line.replace(',', '.')
            fixed_line = fixed_line[:-2] + "\n"
            outfile.write(fixed_line)
    
    print(f"Файл успешно обработан.")
    # Загружаем данные из файла data.txt
    data = pd.read_csv(data_file, sep='\t', header=None, names=['Time', 'Value'])

    # Усредняем значения для повторяющихся моментов времени
    data = data.groupby('Time')['Value'].mean().reset_index()

    # Извлекаем данные из DataFrame
    time = data['Time'].values
    value = data['Value'].values
    

except FileNotFoundError:
    print(f"Ошибка: Файл {data_file} не найден.")
    exit()
except Exception as e:
    print(f"Ошибка при чтении файла {data_file}: {e}")
    exit()
# ---------------------------------------------------------------------------------
# Блок 2: Определение периода колебаний
# ---------------------------------------------------------------------------------

# Находим точки, где значение переходит через ноль
zero_crossings = np.where(np.diff(np.sign(value)))[0]
zero_fixed = []
last_crossing_time = 0  # Инициализируем время последнего перехода

for i in range(1, len(value)):
    if (value[i] <= 0.03) & (value[i] >= -0.03):
        crossing_time = time[i]
        # Проверяем, прошло ли достаточно времени с последнего перехода
        if (crossing_time - last_crossing_time >= min_time_between_zero_crossings):
            zero_fixed.append(crossing_time)
        last_crossing_time = crossing_time

# Если точек пересечения слишком мало
if len(zero_crossings) < 2:
    print("Ошибка: Недостаточно точек пересечения нуля для определения периода колебаний.")
    exit()

# Оцениваем период колебаний по первым нескольким точкам пересечения нуля
period = np.mean(np.diff(zero_fixed[:-10:2])) # Берем среднее по 5 точкам
print(f"Оценка периода колебаний: {period:.4f}")

# ---------------------------------------------------------------------------------
# Блок 3: Запись данных в файл zero_crossings.txt
# ---------------------------------------------------------------------------------

# Записываем данные о моментах перехода через ноль в файл zero_crossings.txt
with open('zero_crossings.txt', 'w') as f:
    for t in zero_fixed:
        f.write(f"{t:.6f}\n")

print(f"Файл zero_crossings.txt создан.")

# ---------------------------------------------------------------------------------
# Блок 4: Вывод результатов и запуск Gnuplot
# ---------------------------------------------------------------------------------

# Автоматически запускаем Gnuplot для построения графиков
gnuplot_command = [
    "gnuplot",
    "-e",
    f"set terminal pngcairo size 1200,800 enhanced font 'Verdana,10';"
    f"set output 'result.png';"
    f"set title 'Колебания';"
    f"set xlabel 'Время';"
    f"set ylabel 'Значение';"
    f"set key top right;"
    f"plot 'data_fixed.txt' using 1:2 with lines title 'Данные (колебания)', 'zero_crossings.txt' using 1:(0) with points pointtype 7 pointsize 1 lc rgb 'red' title 'Переходы через ноль';"
    f"exit;" # Для автоматического закрытия окна Gnuplot после создания графика (неинтерактивный режим)
    # f"pause -1;" # Если нужно, чтобы окно Gnuplot оставалось открытым после построения графиков
]

# Запускаем Gnuplot
try:
    subprocess.run(gnuplot_command, check=True)
    print("\nГрафик построен и сохранен в файле result.png")
except subprocess.CalledProcessError as e:
    print(f"\nОшибка при запуске Gnuplot: {e}")


