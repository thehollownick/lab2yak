#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h> 
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define SIZE 512						// число потоков / размерность блока == числу потоков
#define SHMEM_SIZE SIZE * 4				// размер разделяемой памяти для работы

// Данная функция необходима для последней итерациии, не какая экономия, чтобы не выполнять бесполеную работу
// Использование volatile необходимо для предотвращения кэширования в регистрах (некая оптимизация компилятора)
// Нет необходимости в __syncthreads();
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

// функция ядра
// выполнение суммирования элементов массива на GPU
__global__ void sum_reduction(int* v, int* v_r) {
	// выделение shared памяти 
	__shared__ int partial_sum[SHMEM_SIZE];
	// расчет идентификатора потока
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// загрузка элементов
	// расчет масштабируемого индекса так как размер массива
	// в два раза больше чем имеющееся количество потоков
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// Сохранение перовой подсчитанной частичной суммы, помимо элементов
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();
	// цикл начинаем с половины блока и делим его на две части каждую итерацию
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Каждый поток выполняет свою работу если он не выходит за пределы шага
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}
	// Поток 0 для определенного блока результат будет писать в основную память
	// А результат индексируется этим блоком
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

//функция иницализации массива
// массив заполняется единицами для удобства проверки
void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
	}
}

// функция подсчета суммы элементов массива на CPU
int sumVectorCPU(int* v, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}
	return sum;
}

int main()
{
	//for (int i = 0; i < 10; i++) {
	// размерность тестируемого вектора
	int n = 1 << 10;
	size_t bytes = n * sizeof(int);

	// переменная для подсчета суммы на CPU
	int sumCPU = 0;

	// переенные события для отсчета времени работы алгоритма на GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// массивы для хранения и работы с данными
	// префикс h_ - работа на CPU
	// префикс d_ - работа на GPU
	// постфикс _r - вектор содержит результирующие данные 
	int* h_v, *h_v_r;
	int* d_v, *d_v_r;

	// выделение памяти под массивы
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// заполнение вектора
	initialize_vector(h_v, n);

	// замеры времение работы функции вычисляющей сумму элементов массива на CPU
	// high_resolution_clock для более тончого замера времени
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	sumCPU = sumVectorCPU(h_v, n);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;
	double cpu_time = time_span.count();
	printf("The time: %f milliseconds\n", cpu_time);

	// копирование данных в память GPU
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// определим рамзмеры сетки
	int TB_SIZE = SIZE;
	int GRID_SIZE = n / TB_SIZE / 2;

	// замеры времени работы функции на GPU
	cudaEventRecord(start, 0);
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f milliseconds\n", KernelTime);

	// расчет ускорения
	double S = cpu_time / KernelTime;
	printf("Acceleration: %f\n", S);

	// копирование данных в оперативную память
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// проверка результатов
	printf("Accumulated result is %d \n", h_v_r[0]);
	if(h_v_r[0] == sumCPU)
	printf("COMPLETED SUCCESSFULLY\n");
	cudaFree(d_v);
	cudaFree(d_v_r);
	free(h_v);
	free(h_v_r);
	//	}
	return 0;
}