# Лабораторная работа 1

Реализация должна содержать 2 функции сложения элементов вектора: на CPU и на GPU с применением CUDA.<br><br>
Для реализации данной задачи использовалась следующая аппаратная база:<br>
Центральный процессор:Intel(R) Core(TM) i5-6300HQ CPU @ 2.30GHz, 2304 МГц, ядер: 4, логических процессоров: 4<br>
Оперативная память: _Kllisre DDR4, 2 × 8 GB, 1600 MHz, DualChannel._<br>
Графический процессор: NVIDIA GeForce GTX 950M 4GB

# Реализация 

Суть  алгоритма заключена в делении блока на пополам и сложение соответственных элементов из половин блока. Сумма будет записана в первый блок. Конечная сумма будет записана в первый элемент вектора.<br>

# Результаты
<br><br>
### Время работы и ускорение параллельного алгоритма
 Размерность вектора                                    | 1024 | 4096  | 16384   | 65536    |262144|1048576 
:----:|:----:|:----:|:----:|:----:|:----:|:----:
**Время работы <br /> алгоритма на CUDA, мс.**          | 0,052 | 0,047 | 0,087   | 0,141   |0,439 |1,627
**Время работы <br /> последовательного алгоритма, мс.**| 0,002 | 0,009 | 0,034   | 0,214   |0,6   |2,509
**Ускорение, раз**                                      | 0,045 | 0,193 | 0,394   |  1,519  |1,368 |1,541
<br>

<br>
Из таблицы можем видеть, что применение GPU не всегда оправдано. От части, это обясняется сравнительно малым размером массива и значительным времением затрачиваемым на распараллеливание алгоритма (включая процесс синхронизации). Однако с увеличением размерности массива, применение GPU становится оправданным, поскольку время затрачиваемое на распараллеливание и вычисления производимые на GPU становится меньшим посравнению со временем затрачиваемым CPU на работу последовательного алгоритма.
