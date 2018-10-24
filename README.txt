How to compile: 	mpicxx NQueens_MPI.cpp -o NQueens_MPI
How to use: 		mpirun -np 8 ./NQueens_MPI 12 //12 is the size of the problem
Module required£º	GCC/5.4.0 and OpenMPI/3.1.2 

How to compile: 	nvcc NQueens_CUDA_12.cu -arch=sm_35 -o NQueens_CUDA_12
How to use:		NQueens_CUDA_12
Module required:	GCC/5.4.0 and CUDA/10.0

How to compile: 	nvcc NQueens_CUDA_16.cu -arch=sm_35 -o NQueens_CUDA_16
How to use:		NQueens_CUDA_16
Module required:	GCC/5.4.0 and CUDA/10.0

//NQueens_CUDA_16.cu & NQueens_CUDA_12.cu only changed the #define QUEENS from 12 to 16

How to compile:		gcc NQueens_sequential.cpp -o NQueens_sequential
How to use:		NQueens_sequential
Module required:	GCC/5.4.0