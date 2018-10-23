#include <cstdio>
#include <time.h>
#include <vector>
#define QUEENS (16)

__global__ void countQueens(int* frontQueensPos, int* data, int* numFQP)
{
	int localResult = 0;
	//printf("%d\n", numFQP[0]);
	int thisThread = ((blockIdx.x * gridDim.x + blockIdx.y) * gridDim.y + threadIdx.x)* blockDim.x + threadIdx.y;
	//	printf("1_%d %d %d %d %d %d %d %d\n", thisThread, blockIdx.x, gridDim.x, blockIdx.y, gridDim.y, threadIdx.x, blockDim.x, threadIdx.y);
	//	if (thisThread >= QUEENS * QUEENS * QUEENS * QUEENS)
	//		return;
	if (blockIdx.x >= QUEENS || blockIdx.y >= QUEENS || threadIdx.x >= QUEENS || threadIdx.y >= QUEENS)
		return;

	int* queenPos = new int[QUEENS];

	queenPos[3] = blockIdx.x;
	queenPos[4] = blockIdx.y;
	queenPos[5] = threadIdx.x;
	queenPos[6] = threadIdx.y;

	for (int i = 4; i <= 6; i++) {
		for (int j = 3; j < i; j++) {
			if ((queenPos[i] - i) == (queenPos[j] - j) || (queenPos[i] + i) == (queenPos[j] + j) || queenPos[i] == queenPos[j]) {
				return;
			}
		}
	}

	int totalFQP = numFQP[0] / 3;

	for (int FQP_number = 0; FQP_number < totalFQP; FQP_number++) {
		//	printf("1_%d %d %d %d %d %d %d %d\n", thisThread, blockIdx.x, gridDim.x, blockIdx.y, gridDim.y, threadIdx.x, blockDim.x, threadIdx.y);
		//	if (thisThread >= QUEENS * QUEENS * QUEENS * QUEENS)
		//		return;
		
		for (int i = 0; i < 3; i++)
			queenPos[i] = frontQueensPos[(FQP_number * 3) + i];

		bool legal = true;

		//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
		//	printf("1_%d %d %d %d %d %d %d_%d\n", queenPos[0], queenPos[1], queenPos[2], queenPos[3], queenPos[4], queenPos[5], queenPos[6], totalFQP);

		for (int i = 3; i <= 6; i++) {
			for (int j = 0; j < 3; j++) {
				if ((queenPos[i] - i) == (queenPos[j] - j) || (queenPos[i] + i) == (queenPos[j] + j) || queenPos[i] == queenPos[j]) {
					legal = false;
					break;
				}
			}
			if (!legal)
				break;
		}
		if (!legal)
			continue;

		//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
		//	printf("1_%d %d %d %d %d %d %d_%d\n", queenPos[0], queenPos[1], queenPos[2], queenPos[3], queenPos[4], queenPos[5], queenPos[6], localResult);

		//printf("1_%d %d %d %d %d %d %d\n", thisThread, queenPos[2], blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
		//backtrace
		int posNow = 7;
		queenPos[posNow] = -1;
		while (posNow > 6) {
			queenPos[posNow] ++;
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
					//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
					//	printf("2_%d %d %d %d %d %d %d_%d\n", queenPos[7], queenPos[8], queenPos[9], queenPos[10], queenPos[11], queenPos[12], queenPos[13], localResult);
					posNow--;
				}
				else {
					posNow++;
					queenPos[posNow] = -1;
				}
			}
			else
				posNow--;
		}
	}
	//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
	//	printf("2.5_%d\n", localResult);
	data[thisThread] = localResult;
	//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
	//	printf("3_%d %d %d %d %d %d\n", thisThread, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
}

__host__ void initData(int* data) {
	for (int i = 0; i < QUEENS*QUEENS*QUEENS*QUEENS; i++)
		data[i] = 0;
}

int main(void)
{
	clock_t start, mid1, mid2, end;

	int resultHere = 0;
	int* d_FQP;
	std::vector <int> frontQueenPosV;
	int *frontQueenPos;
	int *tempFrontQueensPos = new int[3];
	int* d_data;
	int data[QUEENS*QUEENS*QUEENS*QUEENS];
	int totalResult = 0;

	initData(data);

	int seedFrom = 0;
	int seedTo = QUEENS * QUEENS * QUEENS;

	start = clock();

	if (seedTo > QUEENS * QUEENS * QUEENS || seedFrom < 0)
		return 0;
	
	for (int i = seedFrom; i < seedTo; i++) {
		tempFrontQueensPos[0] = i / QUEENS / QUEENS;
		tempFrontQueensPos[1] = i / QUEENS % QUEENS;
		tempFrontQueensPos[2] = i % QUEENS;
		if ((tempFrontQueensPos[0] - 0) == (tempFrontQueensPos[1] - 1) || (tempFrontQueensPos[0] + 0) == (tempFrontQueensPos[1] + 1) || tempFrontQueensPos[0] == tempFrontQueensPos[1])
				continue;
		if ((tempFrontQueensPos[2] - 2) == (tempFrontQueensPos[1] - 1) || (tempFrontQueensPos[2] + 2) == (tempFrontQueensPos[1] + 1) || tempFrontQueensPos[2] == tempFrontQueensPos[1])
				continue;
		if ((tempFrontQueensPos[0] - 0) == (tempFrontQueensPos[2] - 2) || (tempFrontQueensPos[0] + 0) == (tempFrontQueensPos[2] + 2) || tempFrontQueensPos[0] == tempFrontQueensPos[2])
				continue;
		frontQueenPosV.push_back(tempFrontQueensPos[0]);
		frontQueenPosV.push_back(tempFrontQueensPos[1]);
		frontQueenPosV.push_back(tempFrontQueensPos[2]);
	}
	//printf("%d\n", frontQueenPosV.size());

	frontQueenPos = new int[frontQueenPosV.size()];
	if (!frontQueenPosV.empty())
		memcpy(frontQueenPos, &frontQueenPosV[0], frontQueenPosV.size() * sizeof(int));
	else
		return 0;

	int numFQP = frontQueenPosV.size();
	int* d_numFQP;

	mid1 = clock();

	cudaMalloc((void**)&d_data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int));
	cudaMalloc((void**)&d_FQP, frontQueenPosV.size() * sizeof(int));
	cudaMalloc((void**)&d_numFQP, sizeof(int));
	cudaMemcpy(d_data, data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_FQP, frontQueenPos, frontQueenPosV.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_numFQP, &numFQP, sizeof(int), cudaMemcpyHostToDevice);

	dim3 blocksPerGrid(QUEENS, QUEENS, 1);
	dim3 threadsPerBlock(QUEENS, QUEENS, 1);

	mid1 = clock();
			//cudaMemcpy(d_FQP, frontQueensPos, (QUEENS - 11) * sizeof(int), cudaMemcpyHostToDevice);
				
	countQueens <<< blocksPerGrid, threadsPerBlock >>> (d_FQP, d_data, d_numFQP);
				
	/*
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf(cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}*/
	cudaThreadSynchronize();

	cudaMemcpy(data, d_data, QUEENS*QUEENS*QUEENS*QUEENS * sizeof(int), cudaMemcpyDeviceToHost);

	mid2 = clock();

	for (int dNum = 0; dNum < QUEENS*QUEENS*QUEENS*QUEENS; dNum++)
		totalResult += data[dNum];

	cudaFree(d_data);
	cudaFree(d_FQP);

	end = clock();

	printf("%d__%d, %d, %d\n", totalResult, mid1 - start, mid2 - mid1, end-mid2);
	
	
}