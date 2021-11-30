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

#define SIZE 512						// ����� ������� / ����������� ����� == ����� �������
#define SHMEM_SIZE SIZE * 4				// ������ ����������� ������ ��� ������

// ������ ������� ���������� ��� ��������� ���������, �� ����� ��������, ����� �� ��������� ���������� ������
// ������������� volatile ���������� ��� �������������� ����������� � ��������� (����� ����������� �����������)
// ��� ������������� � __syncthreads();
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

// ������� ����
// ���������� ������������ ��������� ������� �� GPU
__global__ void sum_reduction(int* v, int* v_r) {
	// ��������� shared ������ 
	__shared__ int partial_sum[SHMEM_SIZE];
	// ������ �������������� ������
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// �������� ���������
	// ������ ��������������� ������� ��� ��� ������ �������
	// � ��� ���� ������ ��� ��������� ���������� �������
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	// ���������� ������� ������������ ��������� �����, ������ ���������
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();
	// ���� �������� � �������� ����� � ����� ��� �� ��� ����� ������ ��������
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// ������ ����� ��������� ���� ������ ���� �� �� ������� �� ������� ����
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}
	// ����� 0 ��� ������������� ����� ��������� ����� ������ � �������� ������
	// � ��������� ������������� ���� ������
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

//������� ������������ �������
// ������ ����������� ��������� ��� �������� ��������
void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
	}
}

// ������� �������� ����� ��������� ������� �� CPU
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
	// ����������� ������������ �������
	int n = 1 << 10;
	size_t bytes = n * sizeof(int);

	// ���������� ��� �������� ����� �� CPU
	int sumCPU = 0;

	// ��������� ������� ��� ������� ������� ������ ��������� �� GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ������� ��� �������� � ������ � �������
	// ������� h_ - ������ �� CPU
	// ������� d_ - ������ �� GPU
	// �������� _r - ������ �������� �������������� ������ 
	int* h_v, *h_v_r;
	int* d_v, *d_v_r;

	// ��������� ������ ��� �������
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// ���������� �������
	initialize_vector(h_v, n);

	// ������ �������� ������ ������� ����������� ����� ��������� ������� �� CPU
	// high_resolution_clock ��� ����� ������� ������ �������
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	sumCPU = sumVectorCPU(h_v, n);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;
	double cpu_time = time_span.count();
	printf("The time: %f milliseconds\n", cpu_time);

	// ����������� ������ � ������ GPU
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// ��������� �������� �����
	int TB_SIZE = SIZE;
	int GRID_SIZE = n / TB_SIZE / 2;

	// ������ ������� ������ ������� �� GPU
	cudaEventRecord(start, 0);
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);
	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f milliseconds\n", KernelTime);

	// ������ ���������
	double S = cpu_time / KernelTime;
	printf("Acceleration: %f\n", S);

	// ����������� ������ � ����������� ������
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// �������� �����������
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