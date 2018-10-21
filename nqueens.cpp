#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <time.h>
//#include "mpi.h"

const int N = 26;
int queens[N];

enum messageType
{
	READY,
	FINISHED,
	NEW_TASK,
	TERMINATE
};

enum messageTage
{
	REQUEST,
	SEED,
	REPLY,
	NUM_SOLUTIONS,
};

bool collide(int row1, int col1, int row2, int col2){
    return (col1==col2 ||row1-row2==col2-col1||row1+col1 == row2+col2);
}


//check is （i，k）safe to place
int valid(int i, int k) {
	int j = 1;
	while (j < i) {
		if(collide(i,k,j,queens[j])){
			return 0;
		}
		j++;
	}
	return 1;
}

void shuffle(int size){
	for (int i = 0; i < size; ++i)
	{
		queens[i] = 0;
	}
}

//place queen, befor that row has been initialed
int place(int size,int row) {
	int solutions = 0;
	int col;
	if (row >=size) {
		solutions++;
	}
	else {
		for (col = 1; col <= size; col++) {
			if (valid(row, col)) {
				queens[row] = col;
				place(row + 1, size);  //recursive
			}
		}
	} 
	return solutions;
}


int check(int size,int col0,int col1,int col2){
	return (collide(0,col0,1,col1)&&collide(0,col0,2,col2)&&collide(1,col1,2,col2));
}

//可以 openmp
std::vector<int> generate(int size) {
	std::vector<int> vector(size*size*size*3);
	int count = 0;
	for (int i = 0; i < size; i++){
		for (int j = 0; i < size; j++){
			for (int k = 0; k < size; k++){
				if(check(size,i,j,k)){
					printf("working now for %d queens problem!\n",size);	
					vector[count*3] = i;
					vector[count*3+1] = j;
					vector[count*3+2] = k;
					count++;
				}

			}
		}
	}
	vector(size*size*size*3) = count;
	return vector;	
}
/*
int main(int argc, char  *argv[]){	
	MPI_Status status;
	int solutions = 0;	// number of solutions
	int size = 8;	    // init size of problem as 8
	int reply;	
	int child;
	int seeds;


	int newTask = NEW_TASK;
	int terminate = TERMINATE;
	int ready = READY;
	int finished = FINISHED;



	double startTime,endTime;
    startTime = MPI_Wtime();

	//  Give MPI it's command arguments  
	int rank, MPIsize;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

    if(rank == 0) {
    	int salves = MPIsize -1;
    	int num_solutions;
    	while(salves){
    		MPI_Recv(&reply, 1, MPI_INT, MPI_ANY_SOURCE, REPLY, MPI_COMM_WORLD, &status);
    		child = status.MPI_SOURCE;

    		if(reply == FINISHED){
    			MPI_Recv(&num_solutions, 1, MPI_INT, child, NUM_SOLUTIONS, MPI_COMM_WORLD, &status);

    			if(num_solutions >0){
    				solutions +=num_solutions;
    			}   		
    		}

    		seeds = generate(size);

    		if (seeds){
    			MPI_Send(&newTask, 1, MPI_INT, child, REQUEST, MPI_COMM_WORLD);

    			MPI_Send(&seeds, 1, MPI_INT, child, SEED, MPI_COMM_WORLD);
    		}else{

    			MPI_Send(&terminate, 1, MPI_INT, child, REQUEST, MPI_COMM_WORLD);
    			salves --;
    		}

    	}

    }else{

    	bool done = false;
    	int my_solutions = 0;
    	int request;
    	int seed;

    	MPI_Send(&ready, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);

    	while(!done){
    		MPI_Recv(&request, 1, MPI_INT, 0, REQUEST, MPI_COMM_WORLD, &status);

    		if(request == NEW_TASK){
    			MPI_Recv(&seed, 1, MPI_INT, child, SEED, MPI_COMM_WORLD, &status);

    			memset(queens,0,sizeof(queens));

    			queens[0] = seed/size;
    			queens[1] = seed%size;

    			my_solutions = place(2,size);

    			MPI_Send(&finished, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);


    			MPI_Send(&my_solutions, 1, MPI_INT, 0, NUM_SOLUTIONS, MPI_COMM_WORLD);
			}else{
				done = true;
			}

    	}

    }


    MPI_Finalize();
    return 0;
}*/

int main(int argc, char *argv[])
{
	clock_t start,finish;
	int size;
	int my_solutions = 0;
	printf("num of prob1em size: ",&size);
	scanf("%d",&size);
	start = clock();
	printf("working now for %d queens problem!\n",size);
	std::vector<int> vector =  generate(size);
	int count = vector[size*size*size*3];
	printf("seeds are %d !\n",count);
	for (int i = 0; i < count; i++)
	{
		shuffle(size);
		printf("working for the %d board\n",i);
		queens[0] = vector[3*i];
		queens[1] = vector[3*i+1];
		queens[2] = vector[3*i+2];
		my_solutions += place(size,4);
	}
	printf("num of solutions are %d\n",my_solutions );

	return 0;
}









