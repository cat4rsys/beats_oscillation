import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
import subprocess

# Имя файла с данными колебаний
data_file          = 'data_fixed.txt'
original_data_file = 'data.txt'

# Параметры для анализа
max_beta            = 0.2  # Максимальное значение beta_exp
peak_threshold      = 0.1  # Порог для определения пиков огибающей
distance_envelope   = 1    # Расстояние между пиками огибающей
delta_time          = 5    # Время в секундах, которе пропускается с начала и с конца колебаний для исключения выбросов

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
# Блок 2: Определение периода биений и a_exp
# ---------------------------------------------------------------------------------

# Находим огибающую с помощью преобразования Гильберта
analytic_signal = hilbert(value)
envelope = np.abs(analytic_signal)

# Записываем данные огибающей в файл env_data.txt
with open('env_data.txt', 'w') as f:
    for t, v in zip(time, envelope):
        f.write(f"{t:.6f} {v:.6f}\n")

start_oscillation  = time.min()
end_oscillation    = time.max()
oscillation_period = end_oscillation - start_oscillation

# ---------------------------------------------------------------------------------
# Блок 3: Нахождение максимумов огибающей и аппроксимация
# ---------------------------------------------------------------------------------

# Определяем функцию для аппроксимации огибающей (экспонента)
def exponential_decay(t, a, beta):
    return a * np.exp(-beta * t)  # a и beta теперь не фиксированы

# Находим пики огибающей с измененным параметром prominence и distance
peaks_envelope, _ = find_peaks(envelope, prominence=peak_threshold, distance=distance_envelope)

# Отбираем пики, попадающие в диапазон колебаний
peaks_envelope_in_range = [p for p in peaks_envelope if start_oscillation+delta_time <= time[p] <= end_oscillation-delta_time]

# Время и значения в точках пиков огибающей
time_peaks_envelope = time[peaks_envelope_in_range]
value_peaks_envelope = envelope[peaks_envelope_in_range]

# Находим впадины (минимумы) огибающей
invalleys_envelope, _ = find_peaks(-envelope, prominence=peak_threshold, distance=distance_envelope)

# Отбираем впадины, попадающие в диапазон колебаний
invalleys_envelope_in_range = [p for p in invalleys_envelope if start_oscillation+delta_time <= time[p] <= end_oscillation-delta_time]

time_invalleys_envelope = time[invalleys_envelope_in_range]
value_invalleys_envelope = envelope[invalleys_envelope_in_range]

# Если впадин слишком мало
if len(invalleys_envelope_in_range) < 2:
    print("Ошибка: Недостаточно впадин для определения периода биений.")
    period = -1
    with open('min_data.txt', 'w') as f:
        f.write(f"0 0\n")
else:
    period = np.mean(np.diff(time[invalleys_envelope_in_range[:4]]))
    with open('min_data.txt', 'w') as f:
        for t, v in zip(time_invalleys_envelope, value_invalleys_envelope):
            f.write(f"{t:.6f} {v:.6f}\n")

# Записываем данные о пиках огибающей в файл max_data.txt
with open('max_data.txt', 'w') as f:
    for t, v in zip(time_peaks_envelope, value_peaks_envelope):
        f.write(f"{t:.6f} {v:.6f}\n")

# Аппроксимируем огибающую по пикам экспонентой
try:
    popt_exp, pcov_exp = curve_fit(exponential_decay, time_peaks_envelope, value_peaks_envelope, p0=[0, 0.03], bounds=([0, 0], [2, max_beta]))
    a_exp, beta_exp = popt_exp
except Exception as e:
    print(f'Ошибка при подборе параметров: {e}')
    exit()

# ---------------------------------------------------------------------------------
# Блок 5: Генерация данных для файла exp_data.txt
# ---------------------------------------------------------------------------------

# Количество точек для экспоненты
num_points_exp = 100

# Создаем массив значений времени для экспоненты (от start_oscillation до end_oscillation)
time_exp = np.linspace(0, end_oscillation, num_points_exp)

# Вычисляем значения экспоненты, используя a_exp и beta_exp
value_exp = a_exp * np.exp(-beta_exp * time_exp)

# Записываем данные экспоненты в файл exp_data.txt
with open('exp_data.txt', 'w') as f:
    for t, v in zip(time_exp, value_exp):
        f.write(f"{t:.6f} {v:.6f}\n")

print(f"Файл exp_data.txt создан с {num_points_exp} точками.")


# ---------------------------------------------------------------------------------
# Блок 6: Вывод результатов в текстовом формате и запуск Gnuplot
# ---------------------------------------------------------------------------------

print("\nРезультаты:")
print(f"  Начальная амплитуда (a_exp): {a_exp:.4f}")
print(f"  Коэффициент затухания (β): {beta_exp:.4f}")
print(f"  Оценка периода биений: {period:.4f}")

# Автоматически запускаем Gnuplot для построения графиков
gnuplot_command = [
    "gnuplot",
    "-e",
    f"set terminal pngcairo size 2400,800 enhanced font 'Verdana,13';"
    f"set output 'result.png';"
    f"set title 'Биения и экспоненциальная огибающая';"
    f"set xlabel 'Время, c';"
    f"set ylabel 'Угловая скорость ω, рад/с';"
    f"set key top right;"
    f"plot 'data_fixed.txt' using 1:2 with lines linewidth 3 title 'График ω(t)', 'env_data.txt' using 1:2 with lines linewidth 0.2 title 'Огибающая', 'exp_data.txt' using 1:2 with lines linewidth 3 title 'Экспонента a = {a_exp:.3f} b = {beta_exp:.4f}', 'max_data.txt' using 1:2 with points pointtype 7 pointsize 2 lc rgb '#AA0000' title 'Максимумы огибающей', 'min_data.txt' using 1:2 with points pointtype 7 pointsize 2 lc rgb '#0000FF' title 'Минимумы огибающей';"
    f"exit;" # Для автоматического закрытия окна Gnuplot после создания графика (неинтерактивный режим)
    # f"pause -1;" # Если нужно, чтобы окно Gnuplot оставалось открытым после построения графиков
]

# Запускаем Gnuplot
try:
    subprocess.run(gnuplot_command, check=True)
    print("\nГрафик построен и сохранен в файле result.png")
except subprocess.CalledProcessError as e:
    print(f"\nОшибка при запуске Gnuplot: {e}")

# построение графиков в питоне
# Строим графики
plt.figure(figsize=(16, 8))
plt.plot(time, value, '-', label='Данные (биения)')
plt.plot(time, envelope, '-', label=f'Огибающая')

plt.plot(time_exp, value_exp, 'r-', linewidth=2, label=f'Экспонента для exp_data.txt (a={a_exp:.3f}, β={beta_exp:.3f})')
plt.plot(time_peaks_envelope, value_peaks_envelope, 'ro', label='Максимумы огибающей')

plt.xlabel('Время')
plt.ylabel('Значение')
plt.title('Биения и экспоненциальная огибающая')
plt.legend()
plt.grid(True)
plt.show()



