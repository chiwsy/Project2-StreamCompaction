
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<thrust\device_vector.h>
#include<thrust\host_vector.h>
#include<thrust\scatter.h>
#include<thrust\copy.h>

//print and file
#include<iostream>
#include<fstream>
#include<sstream>
#include <stdio.h>

//#include<chrono>
#include<ctime>
#include <windows.h>
#include<stdlib.h>

#include "Macros.h"



////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Helper functions////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
void printResult(const int* in, const int& size){
	for (int i = 0; i < size; i++){
		printf("%d ", in[i]);
	}
	printf("\n");
}

void RandArray(int* arr, const int& size){
	for (int i = 0; i < size; i++){
		arr[i] = rand() % 201-100;
		
	}

}

void RandLotsZeros(int* arr, const int& size){
	for (int i = 0; i < size; i++){
		arr[i] = rand() % 2;
		if (arr[i] != 0) arr[i] = rand() % 21 - 10;
	}
}

bool GPU_MemHelper(const int* src, int* &dst, const int& size, const cudaMemcpyKind& type = cudaMemcpyHostToDevice){
	cudaError_t cudaStatus;
	if (type == cudaMemcpyHostToDevice)
		cudaStatus = cudaMalloc((void**)&dst, size*sizeof(int));
	//if (!cudaStatus) return false;
	cudaStatus = cudaMemcpy(dst, src, size*sizeof(int), type);
	return cudaStatus;
}
template<class T>
bool verifyResult(const T& a, const T& b, const int& size){
	for (int i = 0; i < size; i++)
		if (a[i] != b[i]) {
			printf("The %dth is expected %d, but got %d!\n", i, a[i], b[i]);
			//return false;
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

void CPU_Scatter(int* out, int & outsize, const int* in, const int& size){
	//out[0] = 0;
	//for (int i = 0; i < INT_MAX>>4; i++){};
	//for (int i = 1; i <= size; i++)
		//out[i] = (in[i - 1]!=0) + out[i - 1];
	for (int i = 0; i < size; i++)
		if (*(in + i) == 0) continue;
		else {
			*(out+outsize) = *(in + i);
			outsize++;
		}
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Naive Parrallel/////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Naive(int* out, const int* in, int i,int size=arraySize){
	int id = (blockIdx.x*blockDim.x)+threadIdx.x;
	
	if (id > i&&id<size+1){
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
__global__ void GPU_PrefixSumEx_Optimized(int* out, int* src,int* multiBlock,int size,int dataSize=arraySize){
	extern __shared__ int cache[];
	int *cin = cache;
	//int *cout = &cache[blockSize];
	int blockid = (blockIdx.x*blockDim.x);
	int thid = threadIdx.x;
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (id < dataSize){
		cin[threadIdx.x] = src[id];
		//cin[2 * threadIdx.x+1] = src[blockid + 2 * threadIdx.x+1];
		__syncthreads();
		//swap(in, out);
		int lid = (1 + thid) << 1;
		int it = 1;

		for (it = 1; it < size; it *= 2){

			if (lid <= size + 1){

				int rid = lid - it;
				cin[lid - 1] += cin[rid - 1];
			}
			lid <<= 1;
			__syncthreads();
			//swap(cin, cout);
		}
		//it >>= 1;
		if (thid == 0){
			multiBlock[blockIdx.x] = cin[size - 1];
			//out[arraySize] = cin[it-1];
			cin[size - 1] = 0;
		}
		int upbound = it;
		while (it>1){
			it >>= 1;
			lid >>= 1;
			if (lid <= upbound){
				int rid = lid - it;
				int tmp = cin[lid - 1];
				cin[lid - 1] += cin[rid - 1];
				cin[rid - 1] = tmp;
			}
			__syncthreads();
		}

		//swap(cin, cout);
		out[id] = cin[threadIdx.x];

		__syncthreads();
	}
}

__global__ void GPU_final(int* out, int*in, int size){
	out[size] = out[size - 1] + in[size - 1];
}

__global__ void GPU_append(int *out, int *val, int loc,int pos=0){
	out[loc] = val[pos];
}

__global__ void GPU_add(int *out, int *src, int size = arraySize){
	if (blockIdx.x == 0) return;
	//int blockid = (blockIdx.x*blockDim.x);
	//int thid = threadIdx.x;
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (id < size){
		out[id] += src[blockIdx.x];
	}
}
void rec_GPU_PO(int* out, int* src, int size = arraySize, int* multiBlock = NULL){
	int blocks = (size-1) / blockSize+1;
	if (!multiBlock)
		cudaMalloc((void**)&multiBlock, (blocks+1)*sizeof(int));
	GPU_PrefixSumEx_Optimized << < blocks, blockSize, blockSize*sizeof(int) >> >(out, src, multiBlock,blockSize);
	cudaThreadSynchronize();
	if (blocks == 1){
		GPU_append << <1, 1 >> >(out, multiBlock, size);
		return;
	}
	else{
		rec_GPU_PO(multiBlock, multiBlock, blocks);
		GPU_add<<<blocks,blockSize>>>(out, multiBlock, size);
		cudaThreadSynchronize();
		GPU_append << <1, 1 >> >(out, multiBlock, size, blocks);
	}

	if (size == arraySize){
		GPU_final << <1, 1 >> >(out, src, size);
	}
	cudaFree(multiBlock);
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Optimized Parrallel for Scatter/////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Optimized_WithScatter(int* out, int* src, int* multiBlock, int size, int dataSize = scatterSize){
	extern __shared__ int cache[];
	int *cin = cache;
	//int *cout = &cache[blockSize];
	int blockid = (blockIdx.x*blockDim.x);
	int thid = threadIdx.x;
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (id < dataSize){
		cin[threadIdx.x] = (src[id]!=0);
		//cin[2 * threadIdx.x+1] = src[blockid + 2 * threadIdx.x+1];
		__syncthreads();
		//swap(in, out);
		int lid = (1 + thid) << 1;
		int it = 1;

		for (it = 1; it < size; it *= 2){

			if (lid <= size + 1){

				int rid = lid - it;
				cin[lid - 1] += cin[rid - 1];
			}
			lid <<= 1;
			__syncthreads();
			//swap(cin, cout);
		}
		//it >>= 1;
		if (thid == 0){
			multiBlock[blockIdx.x] = cin[size - 1];
			//out[arraySize] = cin[it-1];
			cin[size - 1] = 0;
		}
		int upbound = it;
		while (it>1){
			it >>= 1;
			lid >>= 1;
			if (lid <= upbound){
				int rid = lid - it;
				int tmp = cin[lid - 1];
				cin[lid - 1] += cin[rid - 1];
				cin[rid - 1] = tmp;
			}
			__syncthreads();
		}

		//swap(cin, cout);
		out[id] = cin[threadIdx.x];

		__syncthreads();
		/*for (int i = 1; i < blocks; i <<= 1)
		if (blockIdx.x >= i) cout[threadIdx.x] += multiBlock[blockIdx.x - i];*/
	}
}

void rec_GPU_POS(int* out, int* src, int size = scatterSize, int* multiBlock = NULL){
	int blocks = (size - 1) / blockSize + 1;
	if (!multiBlock)
		cudaMalloc((void**)&multiBlock, (blocks + 1)*sizeof(int));
	GPU_PrefixSumEx_Optimized_WithScatter << < blocks, blockSize, blockSize*sizeof(int) >> >(out, src, multiBlock, blockSize);
	cudaThreadSynchronize();
	if (blocks == 1){
		GPU_append << <1, 1 >> >(out, multiBlock, size);
		return;
	}
	else{
		rec_GPU_PO(multiBlock, multiBlock, blocks);
		GPU_add << <blocks, blockSize >> >(out, multiBlock, size);
		cudaThreadSynchronize();
		GPU_append << <1, 1 >> >(out, multiBlock, size, blocks);
	}

	if (size == scatterSize){
		GPU_final << <1, 1 >> >(out, src, size);
	}
	cudaFree(multiBlock);
}

__global__ void GPU_scatter_cp(int* out, int* aux, int* in, int size = scatterSize){
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (id < size&&in[id] != 0){
		out[aux[id]] = in[id];
	}
	__syncthreads();
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////Optimized Parrallel with Bank Conflicts resolving////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_PrefixSumEx_Optimized_BankConflicts(int* out, int* src, int* multiBlock, int size, int dataSize = arraySize){
	extern __shared__ int cache[];
	int *cin = cache;
	//int *cout = &cache[blockSize];
	int blockid = (blockIdx.x*blockDim.x);
	int thid = threadIdx.x;
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	int offset = 1;

	if (id < dataSize){

		cin[CONFLICT_FREE_OFFSET(thid)] = src[id];

		__syncthreads();
		//swap(in, out);
		int lid = (1 + thid) << 1;
		int it = 1;

		for (it = 1; it < size; it *= 2){

			if (lid <= size + 1){

				int rid = lid - it;
				cin[CONFLICT_FREE_OFFSET(lid - 1)] += cin[CONFLICT_FREE_OFFSET(rid - 1)];
			}
			lid <<= 1;
			__syncthreads();
			//swap(cin, cout);
		}
		//it >>= 1;
		if (thid == 0){
			multiBlock[blockIdx.x] = cin[CONFLICT_FREE_OFFSET(size - 1)];
			//out[arraySize] = cin[it-1];
			cin[CONFLICT_FREE_OFFSET(size - 1)] = 0;
		}
		int upbound = it;
		while (it>1){
			it >>= 1;
			lid >>= 1;
			if (lid <= upbound){
				int rid = lid - it;
				int tmp = cin[CONFLICT_FREE_OFFSET(lid - 1)];
				cin[CONFLICT_FREE_OFFSET(lid - 1)] += cin[CONFLICT_FREE_OFFSET(rid - 1)];
				cin[CONFLICT_FREE_OFFSET(rid - 1)] = tmp;
			}
			__syncthreads();
		}

		//swap(cin, cout);
		out[id] = cin[CONFLICT_FREE_OFFSET(thid)];

		__syncthreads();

	}
}

void rec_GPU_POBR(int* out, int* src, int size = arraySize, int* multiBlock = NULL){
	int blocks = (size - 1) / blockSize + 1;
	if (!multiBlock)
		cudaMalloc((void**)&multiBlock, (blocks + 1)*sizeof(int));
	GPU_PrefixSumEx_Optimized_BankConflicts << < blocks, blockSize, blockSize*sizeof(int) >> >(out, src, multiBlock, blockSize);
	cudaThreadSynchronize();
	if (blocks == 1){
		GPU_append << <1, 1 >> >(out, multiBlock, size);
		return;
	}
	else{
		rec_GPU_POBR(multiBlock, multiBlock, blocks);
		GPU_add << <blocks, blockSize >> >(out, multiBlock, size);
		cudaThreadSynchronize();
		GPU_append << <1, 1 >> >(out, multiBlock, size, blocks);
	}

	if (size == arraySize){
		GPU_final << <1, 1 >> >(out, src, size);
	}
	cudaFree(multiBlock);
}

__global__ void GPU_PrefixSumEx_Optimized_WithScatter_BankConflicts(int* out, int* src, int* multiBlock, int size, int dataSize = arraySize){
	extern __shared__ int cache[];
	int *cin = cache;
	//int *cout = &cache[blockSize];
	int blockid = (blockIdx.x*blockDim.x);
	int thid = threadIdx.x;
	int id = (blockIdx.x*blockDim.x) + threadIdx.x;
	int offset = 1;

	if (id < dataSize){

		cin[CONFLICT_FREE_OFFSET(thid)] = (src[id]!=0);

		__syncthreads();
		//swap(in, out);
		int lid = (1 + thid) << 1;
		int it = 1;

		for (it = 1; it < size; it *= 2){

			if (lid <= size + 1){

				int rid = lid - it;
				cin[CONFLICT_FREE_OFFSET(lid - 1)] += cin[CONFLICT_FREE_OFFSET(rid - 1)];
			}
			lid <<= 1;
			__syncthreads();
			//swap(cin, cout);
		}
		//it >>= 1;
		if (thid == 0){
			multiBlock[blockIdx.x] = cin[CONFLICT_FREE_OFFSET(size - 1)];
			//out[arraySize] = cin[it-1];
			cin[CONFLICT_FREE_OFFSET(size - 1)] = 0;
		}
		int upbound = it;
		while (it>1){
			it >>= 1;
			lid >>= 1;
			if (lid <= upbound){
				int rid = lid - it;
				int tmp = cin[CONFLICT_FREE_OFFSET(lid - 1)];
				cin[CONFLICT_FREE_OFFSET(lid - 1)] += cin[CONFLICT_FREE_OFFSET(rid - 1)];
				cin[CONFLICT_FREE_OFFSET(rid - 1)] = tmp;
			}
			__syncthreads();
		}

		//swap(cin, cout);
		out[id] = cin[CONFLICT_FREE_OFFSET(thid)];

		__syncthreads();

	}
}

void rec_GPU_POSBR(int* out, int* src, int size = scatterSize, int* multiBlock = NULL){
	int blocks = (size - 1) / blockSize + 1;
	if (!multiBlock)
		cudaMalloc((void**)&multiBlock, (blocks + 1)*sizeof(int));
	GPU_PrefixSumEx_Optimized_WithScatter_BankConflicts << < blocks, blockSize, blockSize*sizeof(int) >> >(out, src, multiBlock, blockSize);
	cudaThreadSynchronize();
	if (blocks == 1){
		GPU_append << <1, 1 >> >(out, multiBlock, size);
		return;
	}
	else{
		rec_GPU_POBR(multiBlock, multiBlock, blocks);
		GPU_add << <blocks, blockSize >> >(out, multiBlock, size);
		cudaThreadSynchronize();
		GPU_append << <1, 1 >> >(out, multiBlock, size, blocks);
	}

	if (size == scatterSize){
		GPU_final << <1, 1 >> >(out, src, size);
	}
	cudaFree(multiBlock);
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
	std::ifstream fin("config.cfg");
	char buff[80];
	while (fin.getline(buff, 80)){
		if (buff[0] == '#') continue;
		std::istringstream is(buff);
		is >> arraySize >> blockSize>>scatterSize;
	}
	fin.close();

	blkHost = (int)ceil((arraySize) / (float)blockSize);
    //const int arraySize = 6;
	srand(time(NULL));
	int *a=new int[arraySize];// = { 3, 4, 6, 7, 9, 10 };
	RandArray(a, arraySize);
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int *c=new int [arraySize+1];
	memset(c, 0, (arraySize + 1)*sizeof(int));
	
	//std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	LARGE_INTEGER  large_interger;
	__int64 start, end;
	double diff;
	QueryPerformanceFrequency(&large_interger);
	diff = large_interger.QuadPart;
	/////////////////////////////////////////////
	//////////////CPU calling////////////////////
	/////////////////////////////////////////////
	QueryPerformanceCounter(&large_interger);
	start = large_interger.QuadPart;

	CPU_PrefixSumEx(c, a, arraySize);
	QueryPerformanceCounter(&large_interger);
	end = large_interger.QuadPart;
	
	printf("CPU_Version:\t %f ms:\n", 1000 * (end - start)/diff);
	//printResult(c, arraySize+1);

	/////////////////////////////////////////////
	//////////////GPU calling////////////////////
	/////////////////////////////////////////////
	cudaSetDevice(0);

	/////////////////////////////////////////////
	//////////////Optimized calling//////////////
	/////////////////////////////////////////////
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		int *src, *res;
		int *host_res = new int[arraySize + 1];
		memset(host_res, 0, (arraySize + 1)*sizeof(int));
		GPU_MemHelper(a, src, arraySize);



		//GPU_MemHelper(host_res, buff, arraySize + 1);
		GPU_MemHelper(host_res, res, arraySize + 1);

		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;

		rec_GPU_POBR(res, src);
		

		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);
	
		if (verifyResult(c, host_res, arraySize + 1)) printf("GPU Second Step verifed OK!\n");
		printf("GPU_Optimized_Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaFree(res);
		cudaFree(src);
		cudaDeviceReset();
	}
	
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		int *src, *res;
		int *host_res = new int[arraySize + 1];
		memset(host_res, 0, (arraySize + 1)*sizeof(int));
		GPU_MemHelper(a, src, arraySize);



		//GPU_MemHelper(host_res, buff, arraySize + 1);
		GPU_MemHelper(host_res, res, arraySize + 1);

		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;

		rec_GPU_PO(res, src);


		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);

		if (verifyResult(c, host_res, arraySize + 1)) printf("GPU Second Step verifed OK!\n");
		printf("GPU_Optimized_Bank_Conflicts_Resolved_Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaFree(res);
		cudaFree(src);
		cudaDeviceReset();
	}

	/////////////////////////////////////////////
	//////////////CPU scatter calling////////////
	/////////////////////////////////////////////
	int *Host_before_Scatter = new int[scatterSize];
	RandLotsZeros(Host_before_Scatter, scatterSize);
	//printResult(Host_before_Scatter, scatterSize);
	int *Host_after_Scatter = new int[scatterSize];
	int sizeb=0;
	QueryPerformanceCounter(&large_interger);
	start = large_interger.QuadPart;
	CPU_Scatter(Host_after_Scatter, sizeb, Host_before_Scatter, scatterSize);
	QueryPerformanceCounter(&large_interger);
	end = large_interger.QuadPart;
	printf("CPU_Scatter:\t %f ms:\n", 1000 * (end - start) / diff);
	//printResult(Host_after_Scatter, sizeb);
	//printf("scatterSize-sizeb=%d\n", scatterSize - sizeb);

	/////////////////////////////////////////////
	//////////////GPU scatter calling////////////
	/////////////////////////////////////////////
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		int* Dev_before_Scatter;
		int* allzero = new int[scatterSize + 1];
		memset(allzero, 0, (scatterSize + 1)*sizeof(int));
		GPU_MemHelper(Host_before_Scatter, Dev_before_Scatter, scatterSize);
		int* Dev_aux_Scatter;
		GPU_MemHelper(allzero, Dev_aux_Scatter, scatterSize+1);
		int* Dev_after_Scatter;
		GPU_MemHelper(allzero, Dev_after_Scatter, scatterSize+1);
		
		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;

		rec_GPU_POS(Dev_aux_Scatter, Dev_before_Scatter);
		GPU_scatter_cp << <(int)ceil((scatterSize) / (float)blockSize) , blockSize >> >(Dev_after_Scatter, Dev_aux_Scatter, Dev_before_Scatter);

		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;

		int* host_res=new int[scatterSize];
		GPU_MemHelper(Dev_after_Scatter, host_res, scatterSize, cudaMemcpyDeviceToHost);
		if (verifyResult(Host_after_Scatter, host_res, sizeb)) printf("GPU Scatter verifed OK!\n");
		printf("GPU_Scatter_Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaFree(Dev_after_Scatter);
		cudaFree(Dev_aux_Scatter);
		cudaFree(Dev_before_Scatter);
		cudaDeviceReset();
	}

	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		int* Dev_before_Scatter;
		int* allzero = new int[scatterSize + 1];
		memset(allzero, 0, (scatterSize + 1)*sizeof(int));
		GPU_MemHelper(Host_before_Scatter, Dev_before_Scatter, scatterSize);
		int* Dev_aux_Scatter;
		GPU_MemHelper(allzero, Dev_aux_Scatter, scatterSize + 1);
		int* Dev_after_Scatter;
		GPU_MemHelper(allzero, Dev_after_Scatter, scatterSize + 1);

		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;

		rec_GPU_POSBR(Dev_aux_Scatter, Dev_before_Scatter);
		GPU_scatter_cp << <(int)ceil((scatterSize) / (float)blockSize), blockSize >> >(Dev_after_Scatter, Dev_aux_Scatter, Dev_before_Scatter);

		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;

		int* host_res = new int[scatterSize];
		GPU_MemHelper(Dev_after_Scatter, host_res, scatterSize, cudaMemcpyDeviceToHost);
		if (verifyResult(Host_after_Scatter, host_res, sizeb)) printf("GPU Scatter Bank Conflicts Resolved verifed OK!\n");
		printf("GPU Scatter Bank Conflicts Resolved Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaFree(Dev_after_Scatter);
		cudaFree(Dev_aux_Scatter);
		cudaFree(Dev_before_Scatter);
		cudaDeviceReset();
	}
	/////////////////////////////////////////////
	/////////GPU Thrust::scatter calling/////////
	/////////////////////////////////////////////
	{
		int* host_res = new int[scatterSize];
		struct notzero{
			__host__ __device__ bool operator()(const int x){
				return x != 0;
			}
		};

		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;
		thrust::copy_if(Host_before_Scatter, Host_before_Scatter + scatterSize, host_res, notzero());
		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;

		if (verifyResult(Host_after_Scatter, host_res, sizeb)) printf("Thrust Scatter verifed OK!\n");
		printf("Thrust::GPU_Scatter_Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaDeviceReset();
	}
	/////////////////////////////////////////////
	//////////////Naive calling//////////////////
	/////////////////////////////////////////////
	/*{
		int *src, *res;
		int *host_res = new int[arraySize + 1];
		memset(host_res, 0, (arraySize + 1)*sizeof(int));
		GPU_MemHelper(a, src, arraySize);
		GPU_MemHelper(host_res, res, arraySize + 1);
		QueryPerformanceCounter(&large_interger);
		start = large_interger.QuadPart;
		for (int i = 0; i <= arraySize; i++)
			GPU_PrefixSumEx_Naive << < (int)ceil((arraySize + 1) / (float)blockSize), blockSize >> >(res, src, i);
		QueryPerformanceCounter(&large_interger);
		end = large_interger.QuadPart;
		GPU_MemHelper(res, host_res, arraySize + 1, cudaMemcpyDeviceToHost);
		if (verifyResult(c, host_res, arraySize + 1)) printf("GPU Naive verifed OK!\n");
		printf("GPU_Naive_Version:\t %f ms:\n", 1000 * (end - start) / diff);
		free(host_res);
		cudaFree(res);
		cudaFree(src);
		cudaDeviceReset();
	}*/
	





	/////////////////////////////////////////////
	//////////////GPU Reset//////////////////////
	/////////////////////////////////////////////
	cudaDeviceReset();
	
	return 0;
}



