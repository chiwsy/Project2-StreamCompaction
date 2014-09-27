
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include<chrono>
#include<ctime>
#include<stdlib.h>

#include "Macros.h"

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Serial Version On CPU///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
void CPU_PrefixSumEx(int* out, const int* in, const int& size){
	out[0] = 0;
	//for (int i = 0; i < INT_MAX>>4; i++){};
	for (int i = 1; i <= size; i++)
		out[i] = in[i - 1]+out[i-1];
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Naive Parrallel/////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Naive(int* out, const int* in, int size,int i){
	int id = (blockIdx.x*blockDim.x)+threadIdx.x;
	
	if (id > i&&id<=size){
			out[id] += in[id - i-1];
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Optimized Parrallel/////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Optimized(int* out, const int* in, const int& size,int i){
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (i == 0 && id < size) {
		out[id + 1] = in[id];
		//out[0] = 0;
	}
	else if (id >= i&&id <= size){
		out[id] =in[id]+in[id - i];
	}
	else if (id < i) out[id] = in[id];

	
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Optimized Parrallel with Scatter////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Optimized_WithScatter(int* out, const int* in, const int& size){

}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////Optimized Parrallel with Scatter and Bank Conflicts//////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Optimized_WithScatter_BankConflicts(int* out, const int* in, const int& size){

}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Helper functions////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
void printResult(const int* in, const int& size){
	for (int i = 0; i < size;i++){
		printf("%d ",in[i]);
	}
	printf("\n");
}

void RandArray(int* arr, const int& size){
	for (int i = 0; i < size; i++)
		arr[i] = rand() % 201-100;
}
bool GPU_MemHelper(const int* src, int* &dst, const int& size, const cudaMemcpyKind& type = cudaMemcpyHostToDevice){
	cudaError_t cudaStatus;
	if (type == cudaMemcpyHostToDevice)
		cudaStatus=cudaMalloc((void**)&dst, size*sizeof(int));
	//if (!cudaStatus) return false;
	cudaStatus = cudaMemcpy(dst, src, size*sizeof(int), type);
	return cudaStatus;
}
bool verifyResult(const int* a, const int* b,const int& size){
	for (int i = 0; i < size; i++)
		if (a[i] != b[i]) {
			printf("The %dth is expected %d, but got %d!\n" ,i, a[i], b[i]);
			return false;
		}
	return true;
}

template<class T>
void swap(T& a, T& b){
	T c = a;
	a = b;
	b = c;
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Main functions//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    //const int arraySize = 6;
	srand(time(NULL));
	int a[arraySize];// = { 3, 4, 6, 7, 9, 10 };
	RandArray(a, arraySize);
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize+1] = { 0 };
	
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	
	/////////////////////////////////////////////
	//////////////CPU calling////////////////////
	/////////////////////////////////////////////
	start = std::chrono::high_resolution_clock::now();
	CPU_PrefixSumEx(c, a, arraySize);
	end = std::chrono::high_resolution_clock::now();
	
	printf("CPU_Version:\t %f ms:\n", std::chrono::duration<float>(end - start).count()*1000.0f);
	//printResult(c, arraySize+1);

	/////////////////////////////////////////////
	//////////////GPU calling////////////////////
	/////////////////////////////////////////////
	cudaSetDevice(0);

	/////////////////////////////////////////////
	//////////////Optimized calling//////////////
	/////////////////////////////////////////////
	{

		int *src, *res, *buff;
		int *host_res = new int[arraySize + 1];
		memset(host_res, 0, (arraySize + 1)*sizeof(int));
		GPU_MemHelper(a, src, arraySize);



		GPU_MemHelper(host_res, buff, arraySize + 1);
		GPU_MemHelper(host_res, res, arraySize + 1);
		//cudaMalloc((void**)&res, (arraySize + 1)*sizeof(int));


		start = std::chrono::high_resolution_clock::now();
		//GPU_PrefixSumEx_Optimized << < (int)ceil((arraySize + 1) / (float)blockSize), blockSize >> >(res, src, arraySize, 0);
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);
		if (verifyResult(a, host_res, arraySize)) printf("GPU first step verifed OK!\n");


		for (int i = 1; i <= arraySize; i <<= 1){
			//GPU_PrefixSumEx_Optimized << < (int)ceil((arraySize + 1) / (float)blockSize), blockSize >> >(buff, res, arraySize, i);
			swap(res, buff);
		}
		end = std::chrono::high_resolution_clock::now();
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);
		if (verifyResult(c, host_res, arraySize + 1)) printf("GPU Naive verifed OK!\n");
		printf("GPU_Optimized_Version:\t %f ms:\n", std::chrono::duration<float>(end - start).count()*1000.0f);
		free(host_res);
		cudaFree(res);
		cudaFree(src);
		cudaDeviceReset();
	}

	/////////////////////////////////////////////
	//////////////Naive calling//////////////////
	/////////////////////////////////////////////
	{
		int *src, *res;
		int *host_res = new int[arraySize + 1];
		memset(host_res, 0, (arraySize + 1)*sizeof(int));
		GPU_MemHelper(a, src, arraySize);
		GPU_MemHelper(host_res, res, arraySize + 1);
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i <= arraySize; i++)
			GPU_PrefixSumEx_Naive << < (int)ceil((arraySize + 1) / (float)blockSize), blockSize >> >(res, src, arraySize, i);
		end = std::chrono::high_resolution_clock::now();
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);
		if (verifyResult(c, host_res, arraySize + 1)) printf("GPU Naive verifed OK!\n");
		printf("GPU_Naive_Version:\t %f ms:\n", std::chrono::duration<float>(end - start).count()*1000.0f);
		free(host_res);
		cudaFree(res);
		cudaFree(src);
		//cudaDeviceReset();
	}
	





	/////////////////////////////////////////////
	//////////////GPU Reset//////////////////////
	/////////////////////////////////////////////
	cudaDeviceReset();
	
	return 0;
}



