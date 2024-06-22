#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

struct parametrs {
    int N, R, BS, K;
};

// Parametry dla liczb N, R, BS, K
parametrs Values[] = {

// wartości dla tab1 (R = 1, k = 1)
 
/*
    {64, 1, 8, 1},
    {64, 1, 16, 1},
    {64, 1, 32, 1},
    {128, 1, 8, 1},
    {128, 1, 16, 1},
    {128, 1, 32, 1},
    {256, 1, 8, 1},
    {256, 1, 16, 1},
    {256, 1, 32, 1},
    {512, 1, 8, 1},
    {512, 1, 16, 1},
    {512, 1, 32, 1},  
    {1024, 1, 8, 1},
    {1024, 1, 16, 1},
    {1024, 1, 32, 1},
    {2048, 1, 8, 1},
    {2048, 1, 16, 1},
    {2048, 1, 32, 1},
    {4096, 1, 8, 1},
    {4096, 1, 16, 1},
    {4096, 1, 32, 1},
    {8192, 1, 8, 1},
    {8192, 1, 16, 1},
    {8192, 1, 32, 1},
*/
//    {16384, 1, 8, 1},
//    {16384, 1, 16, 1},
//    {16384, 1, 32, 1},
    
/*
//Wartosci dla wykresu
{ 1024 , 1, 8, 1},
{ 1024 , 1, 16, 1},
{ 1024 , 1, 32, 1},
{ 1536 , 1, 8, 1},
{ 1536 , 1, 16, 1},
{ 1536 , 1, 32, 1},
{ 2048 , 1, 8, 1},
{ 2048 , 1, 16, 1},
{ 2048 , 1, 32, 1},
{ 2560 , 1, 8, 1},
{ 2560 , 1, 16, 1},
{ 2560 , 1, 32, 1},
{ 3072 , 1, 8, 1},
{ 3072 , 1, 16, 1},
{ 3072 , 1, 32, 1},
{ 3584 , 1, 8, 1},
{ 3584 , 1, 16, 1},
{ 3584 , 1, 32, 1},
{ 4096 , 1, 8, 1},
{ 4096 , 1, 16, 1},
{ 4096 , 1, 32, 1},
{ 4608 , 1, 8, 1},
{ 4608 , 1, 16, 1},
{ 4608 , 1, 32, 1},
{ 5120 , 1, 8, 1},
{ 5120 , 1, 16, 1},
{ 5120 , 1, 32, 1},
{ 5632 , 1, 8, 1},
{ 5632 , 1, 16, 1},
{ 5632 , 1, 32, 1},
{ 6144 , 1, 8, 1},
{ 6144 , 1, 16, 1},
{ 6144 , 1, 32, 1},
{ 6656 , 1, 8, 1},
{ 6656 , 1, 16, 1},
{ 6656 , 1, 32, 1},
{ 7168 , 1, 8, 1},
{ 7168 , 1, 16, 1},
{ 7168 , 1, 32, 1},
{ 7680 , 1, 8, 1},
{ 7680 , 1, 16, 1},
{ 7680 , 1, 32, 1},
{ 8192 , 1, 8, 1},
{ 8192 , 1, 16, 1},
{ 8192 , 1, 32, 1},
*/

// wartości dla tab2 (n = 2048)  
/*
{2048, 2, 32, 2},
{2048, 2, 32, 4},
{2048, 2, 32, 8},
{2048, 2, 32, 16},
{2048, 4, 32, 2},
{2048, 4, 32, 4},
{2048, 4, 32, 8},
{2048, 4, 32, 16},
{2048, 8, 32, 2},
{2048, 8, 32, 4},
{2048, 8, 32, 8},
{2048, 8, 32, 16},
{2048, 16, 32, 2},
{2048, 16, 32, 4},
{2048, 16, 32, 8},
{2048, 16, 32, 16},
*/

{1024, 2, 32, 2},

{1024, 16, 32, 8},

};

// Generowanie losowych liczb
void MatrixRandom(float* matrix, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = dis(gen);
        }
    }
}

// Sprawdzenie że 2 macierzy są takie same
void verifyMatrices(float* matrix1, float* matrix2, int N, int R) {
    int matrix_size = N - 2 * R;
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        if ((matrix1[i] != matrix2[i])) {
            fprintf(stderr, "Discrepancy at index [%d]: %f != %f\n", i, matrix2[i], matrix1[i]);
            exit(1);
        }
    }
}

// Obliczanie tablicy sekwencyjnie (CPU)
void SumMatrixCPU(int N, int R, float* input_matrix, float* output_matrix) {
    for (int i = R; i < N - R; i++) {
        for (int j = R; j < N - R; j++) {
            float sum = 0;
            for (int ii = i - R; ii <= i + R; ii++) {
                for (int jj = j - R; jj <= j + R; jj++) {
                    sum += input_matrix[ii * N + jj];
                }
            }
            output_matrix[(i - R) * (N - 2 * R) + j - R] = sum;
        }
    }
}

// Jądro do przetwarzania macierzy z wykorzystaniem pamięci globalnej na GPU
__global__ void kernelGlobalMemoii(float* input, float* output, int N, int R, int K) {
    int matrix_size = N - 2 * R;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = (threadIdx.y + blockIdx.y * blockDim.y) * K;

    for (int k = 0; k < K; k++) {
        float sum = 0;
        if (i < matrix_size && j + k < matrix_size) {
            for (int ii = -R; ii <= R; ii++) {
                for (int jj = -R; jj <= R; jj++) {
                    sum += input[(j + k + R + ii) * N + (i + R + jj)];
                }
            }
            output[(j + k) * matrix_size + i] = sum;
        }
    }
}

// Funkcja pomocnicza do konwersji indeksów macierzy dla pamięci współdzielonej
__device__ inline int getTranslatedIndex(int idx, int width, int xOffset, int yOffset, int N) {
    int matX = idx / width;
    int matY = idx % width;
    return (matY + xOffset) * N + (matX + yOffset);
}

// Jądro do przetwarzania macierzy z wykorzystaniem pamięci współdzielonej na GPU
__global__ void kernelSharedMemoii(float* input, float* output, int N, int R, int K) {
    extern __shared__ float sharedMemoii[];
    unsigned int i = ((blockIdx.x * blockDim.x) + threadIdx.x) + R;
    unsigned int j = ((blockIdx.y * blockDim.y * K) + threadIdx.y) + R;
    unsigned int width = (N - 2 * R - blockIdx.x * blockDim.x >= blockDim.x) ? blockDim.x + 2 * R : N - 2 * R - blockIdx.x * blockDim.x + 2 * R;
    unsigned int height = (N - 2 * R - blockIdx.y * blockDim.x >= blockDim.x) ? blockDim.x + 2 * R : N - 2 * R - blockIdx.y * blockDim.x + 2 * R;
    unsigned int size = height * width;
    unsigned int threadNumber = threadIdx.x * blockDim.x + threadIdx.y;
    unsigned int threadX = threadIdx.x;
    unsigned int threadY = threadIdx.y;

    if (N < blockDim.x + 2 * R) {
        width = N;
    }

    if (N < blockDim.x + 2 * R) {
        height = N;
    }

    for (int k = 0; k < K; k++) {
        for (unsigned int idx = threadNumber; idx < size; idx += blockDim.x * blockDim.x) {
            int translatedIndex = getTranslatedIndex(idx, width, blockIdx.x * blockDim.x, blockIdx.y * blockDim.y * K + k * blockDim.x, N);
            sharedMemoii[idx] = input[translatedIndex];
        }
        __syncthreads();

        if (i < N - R && j < N - R) {
            float total = 0;
            for (unsigned int x = threadX; x <= threadX + 2 * R; x++)
                for (unsigned int y = threadY; y <= threadY + 2 * R; y++)
                    total += sharedMemoii[y * width + x];

            output[(i - R) * (N - 2 * R) + (j - R)] = total;
            j += blockDim.x;
        }

        __syncthreads();
    }
}

// Obliczanie tablicy równolegle (GPU)
void SumMatrixGPU(int N, int R, int K, int matrix_size, const char* mode, float* input_matrix, float* output_matrix, dim3 threadGrid, dim3 blockGrid, int shared_memoii_size, float& avg_time, float& avg_speed) {
    float* input, * output;
    cudaEvent_t startEvent, stopEvent;
    auto* hostOutput = (float*)malloc(matrix_size * matrix_size * sizeof(float));
    float totalElapsedTime = 0.0f;

    cudaMalloc((void**)&input, N * N * sizeof(float));
    cudaMalloc((void**)&output, matrix_size * matrix_size * sizeof(float));

    for (int iter = 0; iter < 1; iter++) { // zmiana na iter = 1, żeby nie obliczało kilku razy (dla nvidia nsight)
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaMemcpyAsync(input, input_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(startEvent, nullptr);

        if (strcmp(mode, "global") == 0) {
            kernelGlobalMemoii<<<blockGrid, threadGrid>>>(input, output, N, R, K);
        } else if (strcmp(mode, "shared") == 0) {
            kernelSharedMemoii<<<blockGrid, threadGrid, shared_memoii_size>>>(input, output, N, R, K);
        }

        cudaEventRecord(stopEvent, nullptr);
        cudaMemcpyAsync(hostOutput, output, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stopEvent);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        totalElapsedTime += elapsedTime;
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    cudaFree(input);
    cudaFree(output);
   verifyMatrices(output_matrix, hostOutput, N, R);
    free(hostOutput);

    avg_time = totalElapsedTime / 1.0f; // zmiana na / 1.0f , żeby nie obliczało kilku razy (dla nvidia nsight)
    float processedElements = (N - 2 * R) * (N - 2 * R);
    float flops_per_element = (2 * R + 1) * (2 * R + 1); // liczba operacji dla każdego elementu wyjściowego
    avg_speed = processedElements * flops_per_element / (avg_time / 1000.0f); // FLOPS
}

int main() {
    ofstream resultsFile("performance_results.txt");
    resultsFile << "N, Average CPU Execution Time (ms), Average CPU Processing Speed (FLOPS), Average GPU Global Execution Time (ms), Average GPU Global Processing Speed (FLOPS), Average GPU Shared Execution Time (ms), Average GPU Shared Processing Speed (FLOPS)\n";

    for (auto& val : Values) {
        int N = val.N, R = val.R, BS = val.BS, K = val.K;
        //Dla zadania 3 e: N_nas = 2048
        N = (N-2)*K+2*R;
        
        int matrix_size = N - 2 * R;
        int shared_memoii_size = sizeof(float) * (BS + 2 * R) * (BS + 2 * R);
        auto* input_matrix = (float*)malloc(N * N * sizeof(float));
        auto* output_matrix = (float*)malloc(matrix_size * matrix_size * sizeof(float));
        dim3 threadGrid(BS, BS);
        dim3 blockGrid(ceil(matrix_size / (float)BS), ceil(matrix_size / (float)BS / K));

        // Wyświetlanie danych
        printf("Values - N: %d, R: %d, BS: %d, K: %d\n", N, R, BS, K);

        MatrixRandom(input_matrix, N);


        
        // Obliczanie czasu dla CPU [ms]
        float totalElapsedTime_CPU = 0.0f;
        for (int iter = 0; iter < 1; iter++) { // zmiana na iter = 1, żeby nie obliczało kilku razy (dla nvidia nsight)
            auto start_time_CPU = chrono::high_resolution_clock::now();
            SumMatrixCPU(N, R, input_matrix, output_matrix);
            auto end_time_CPU = chrono::high_resolution_clock::now();
            auto time_CPU = chrono::duration_cast<chrono::microseconds>(end_time_CPU - start_time_CPU);
            totalElapsedTime_CPU += time_CPU.count() / 1000.0f;
        }
        float avgElapsedTime_CPU = totalElapsedTime_CPU / 1.0f; // zmiana na /1.0f , żeby nie obliczało kilku razy (dla nvidia nsight)
        float processedElements_CPU = (N - 2 * R) * (N - 2 * R);
        float flops_per_element_CPU = (2 * R + 1) * (2 * R + 1); // liczba operacji dla każdego elementu wyjściowego
        float avgProcessingSpeed_CPU = processedElements_CPU * flops_per_element_CPU / (avgElapsedTime_CPU / 1000.0f); // FLOPS

        printf("Average CPU execution time: %f ms\n", avgElapsedTime_CPU);
        printf("Average CPU processing speed: %f FLOPS\n", avgProcessingSpeed_CPU);
        

        float avgElapsedTime_GPU_Global, avgProcessingSpeed_GPU_Global;
        SumMatrixGPU(N, R, K, matrix_size, "global", input_matrix, output_matrix, threadGrid, blockGrid, shared_memoii_size, avgElapsedTime_GPU_Global, avgProcessingSpeed_GPU_Global);
        printf("Average GPU execution time for global memory: %f ms\n", avgElapsedTime_GPU_Global);
        printf("Average GPU processing speed for global memory: %f FLOPS\n", avgProcessingSpeed_GPU_Global);

        
        float avgElapsedTime_GPU_Shared, avgProcessingSpeed_GPU_Shared;
        SumMatrixGPU(N, R, K, matrix_size, "shared", input_matrix, output_matrix, threadGrid, blockGrid, shared_memoii_size, avgElapsedTime_GPU_Shared, avgProcessingSpeed_GPU_Shared);
        printf("Average GPU execution time for shared memory: %f ms\n", avgElapsedTime_GPU_Shared);
        printf("Average GPU processing speed for shared memory: %f FLOPS\n", avgProcessingSpeed_GPU_Shared);
        

        // resultsFile << N << ", " << avgElapsedTime_CPU << ", " << avgProcessingSpeed_CPU << ", " << avgElapsedTime_GPU_Global << ", " << avgProcessingSpeed_GPU_Global << ", " << avgElapsedTime_GPU_Shared << ", " << avgProcessingSpeed_GPU_Shared << "\n";

        printf("\n");

        free(input_matrix);
        free(output_matrix);
    }

    resultsFile.close();

    return 0;
}
