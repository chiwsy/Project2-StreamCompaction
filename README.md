Questions and answers:
##You can use the performance.py to execute the program with ease. 
1, Please see the enclosed performance.xlsx to see all the result and chart;
	Based on chart in tab:GPU_Naive, we can see that the performance of this naive version is very bad. Much worse than the serial version. Because the computation threads of this naive version is very large and data transfer from the global memory to the SMs can consume large quantity of time. That is why the performance is so bad.

2, Also in the tab:GPU_Naive, we can see that the curve of optimized version is better than the naive version. That is because the latency of the shared memory is much shorter than global memory. But still, it does not exceeded the performance of CPU with little test data set. 

3, I don't really sure what is thrust do. But one thing, according to my performance statistics, the thrust performance is highly similar to my CPU version. You can see here I plot 10 plus chart and the thrust version and the CPU version is always almost the same line. And my GPU version exceeded them with block size=128 and data size larger than about 1MB. However, the thrust seems allocate memory on device with no time. Because If I count the time used for data transfer, it even exceeded the time thrust required for complete the whole computation. 