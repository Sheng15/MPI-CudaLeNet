#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

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
    return (col1==col2 ||abs(row1-row2)==abs(col2-col1));
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



//place queen
int place(int k, int n) {
	int solutions = 0;
	int j;
	if (k > n) {
		solutions++;
	}
	else {
		for (j = 1; j <= n; j++) {
			if (valid(k, j)) {
				queens[k] = j;
				place(k + 1, n);  //recursive
			}
		}
	}
	return solutions;
}



int generate(int size){
	int seed = 0;
	do{
		seed++;
	}while(
		seed<=size*(size-1)&&collide(0,seed/size,1,seed%size));

	if(seed >size*(size-1)){
		return 0;
	}else{
		return seed;
	}
}

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

	/*  Give MPI it's command arguments  */
	int rank, MPIsize;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

    if(rank == 0) {
    	int salves = size-1;
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
}












