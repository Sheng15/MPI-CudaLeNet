#include <cstdio>

#define QUEENS (14)

__global__ void countQueens(int* frontQueensPos, int* data)
{

	int thisThread = ((blockIdx.x * gridDim.x + blockIdx.y) * gridDim.y + threadIdx.x)* blockDim.x + threadIdx.y;
//	printf("1_%d %d %d %d %d %d %d %d\n", thisThread, blockIdx.x, gridDim.x, blockIdx.y, gridDim.y, threadIdx.x, blockDim.x, threadIdx.y);
//	if (thisThread >= QUEENS * QUEENS * QUEENS * QUEENS)
//		return;
	
	int localResult = 0;

	int* queenPos = new int[QUEENS];
	for (int i = 0; i < QUEENS - 11; i++)
		queenPos[i] = frontQueensPos[i];

	queenPos[QUEENS - 11] = blockIdx.x;
	queenPos[QUEENS - 10] = blockIdx.y;
	queenPos[QUEENS - 9] = threadIdx.x;
	queenPos[QUEENS - 8] = threadIdx.y;

	for (int i = QUEENS - 11; i <= QUEENS - 8; i++) {
		for (int j = i - 1; j >= 0; j--) {
			if ((queenPos[i] - i) == (queenPos[j] - j) || (queenPos[i] + i) == (queenPos[j] + j) || queenPos[i] == queenPos[j]) {
				return;
			}
		}
	}

	printf("1_%d %d %d %d %d %d %d\n", thisThread, queenPos[2], blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
	//backtrace
	bool legal = true;
	int posNow = QUEENS - 7;
	queenPos[posNow] = 0;
	while (posNow > QUEENS - 8) {
		while (queenPos[posNow] < QUEENS) {
			legal = true;
			for (int j = posNow - 1; j >= 0; j--) {
				if ((queenPos[posNow] - posNow) == (queenPos[j] - j) || (queenPos[posNow] + posNow) == (queenPos[j] + j) || queenPos[posNow] == queenPos[j]) {
					legal = false;
					break;
				}
			}
			if (!legal)
				queenPos[posNow] ++;
			else
				break;
		}
		if (queenPos[posNow] < QUEENS) {
			if (posNow == (QUEENS - 1)) {
				localResult++;
				posNow--;
				queenPos[posNow]++;	
			}
			else {
				posNow++;
				queenPos[posNow] = -1;
			}
		}
		else
			posNow--;
	}	
	

	//atomicAdd(&result, 1);
	data[thisThread] += 1;
	printf("2_%d %d %d %d %d %d %d\n", thisThread, queenPos[2], blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
}

__host__ void initData(int* data) {
	for (int i = 0; i < QUEENS*QUEENS*QUEENS*QUEENS; i++)
		data[i] = 0;
}

int main(void)
{
	int resultHere = 0;
	int* d_FQP;
	int frontQueensPos[QUEENS - 11] = { 0, 2, 4 };
	int* d_data;
	int data[QUEENS*QUEENS*QUEENS*QUEENS];
	int totalResult = 0;
	initData(data);
	cudaMalloc((void**)&d_data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int));
	cudaMalloc((void**)&d_FQP, (QUEENS - 11) * sizeof(int));
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			for (int k = 0; k < 1; k++) {
				cudaMemcpy(d_data, data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_FQP, frontQueensPos, (QUEENS - 11) * sizeof(int), cudaMemcpyHostToDevice);

				dim3 blocksPerGrid(QUEENS, QUEENS, 1);
				dim3 threadsPerBlock(QUEENS, QUEENS, 1);

				countQueens << < blocksPerGrid, threadsPerBlock >> > (d_FQP, d_data);
				/*
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) {
					printf(cudaGetErrorString(error));
					exit(EXIT_FAILURE);
				}*/
				cudaThreadSynchronize();

				cudaMemcpy(data, d_data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int), cudaMemcpyDeviceToHost);

				for (int dNum = 0; dNum < QUEENS*QUEENS*QUEENS*QUEENS; dNum++) {
					resultHere += data[dNum];
				}
				printf("%d\n", resultHere);

				initData(data);
			}
		}
	}
	cudaFree(d_data);
	cudaFree(d_FQP);
	
}