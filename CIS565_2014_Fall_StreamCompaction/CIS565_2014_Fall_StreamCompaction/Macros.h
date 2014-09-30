int arraySize = 512;
int scatterSize = 10;
//const __device__ int size = arraySize + 1;
int blockSize = 128;
int blkHost; 
//const __device__ int blocks = (arraySize)/blockSize+1;
//typedef std::chrono::duration<float, std::ratio<12096, 10000>> microSecond;

#define NUM_BANKS 32  
#define LOG_NUM_BANKS 5 
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))