#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

const int N = 26;//max of size if 26
//int queens[N]; // board of a game 
int solutions = 0;

enum messageType
{
	READY,
	FINISHED,
	NEW_TASK,
	TERMINATE,
	RESULT,
};

enum messageTage
{
	REQUEST,
	SEED,
	REPLY,
	NUM_SOLUTIONS,
};

bool collide(int row1, int col1, int row2, int col2){
    return (col1==col2 ||(row1-row2==col1-col2)||(row1+col1 == row2+col2));
}


//check is （i，k）safe to place
int valid(int i, int k, int* queens) {
	for (int j = 0; j < i;j++){
		if (collide(i,k,j,queens[j])){
			return 0;
		}
	}
	return 1;
}

void shuffle(int size,int* queens){
	for (int i = 0; i < size; ++i)
	{
		queens[i] = 0;
	}
}

//place queen, befor that row has been initialed
void place(int size,int row,int* queens) {
	int col;
	if (row >= size) {
		solutions++;
	}
	else {
		for (col = 0; col < size; col++) {
			if (valid(row, col,queens)) {
				queens[row] = col;
				place(size,row + 1,queens);  //recursive
			}
		}
	} 
}


int check(int size,int col0,int col1,int col2){
	return (collide(0,col0,1,col1)&&collide(0,col0,2,col2)&&collide(1,col1,2,col2));
}


int generate(int size){
	static int seed = 0;
	do{
		seed++;
	}while(seed <= size*size -1 && collide(0,seed/size,1,seed%size));

	if (seed > size*size -1){
		return 0;
	}else{
		return seed;
	}

}




int main(int argc, char  *argv[]){	
	//int solutions = 0;	// number of solutions
	int size = 8;	        // init size of problem as 8
	int reply;	
	int slave;
	int seeds = size * size * size -1;
	int solutionCount = 0;
	int slaveResult = 0;

	//mpi message type
	int ready = READY;
	int finished = FINISHED;
	int newTask = NEW_TASK;
	int terminate = TERMINATE;
	int result = RESULT;

	double startTime,endTime;
    startTime = MPI_Wtime();

	int rank, MPIsize;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

    if(rank == 0) {
    	MPI_Status masterStatus;
    	int slaves = MPIsize -1;
    	while(slaves){
    		MPI_Recv(&reply, 1, MPI_INT, MPI_ANY_SOURCE, REPLY, MPI_COMM_WORLD, &masterStatus);
    		slave = masterStatus.MPI_SOURCE;
    		//printf("receive notice from slave %d\n", slave );

    		if(reply == FINISHED || reply == READY){
	    		if (seeds){
	    			MPI_Send(&newTask, 1, MPI_INT, slave, REQUEST, MPI_COMM_WORLD);

	    			MPI_Send(&seeds, 1, MPI_INT, slave, SEED, MPI_COMM_WORLD);
	    			//printf("send seed to salve %d\n", slave );
	    			seeds --;
	    		}else{
	    			MPI_Send(&terminate, 1, MPI_INT, slave, REQUEST, MPI_COMM_WORLD);
	    			//printf("message to terminate slave %d\n", slave );
	    			slaves --;
	    		}			  		
    		}

    		if (reply == RESULT)
    		{
    			MPI_Recv(&slaveResult, 1, MPI_INT, slave, NUM_SOLUTIONS, MPI_COMM_WORLD, &masterStatus);
    			solutionCount +=slaveResult;
    			printf("from slave %d ,num of solutions are %d\n",rank,slaveResult);
    		}
    	}
    	printf("from slave %d ,num of solutions are %d\n",rank,solutionCount);
    }else{
    	MPI_Status slaveStatus;
    	bool done = false;
    	int request;
    	int seed;

    	MPI_Send(&ready, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);

    	while(!done){
    		MPI_Recv(&request, 1, MPI_INT, 0, REQUEST, MPI_COMM_WORLD, &slaveStatus);

    		if(request == NEW_TASK){
    			MPI_Recv(&seed, 1, MPI_INT, 0, SEED, MPI_COMM_WORLD, &slaveStatus);
    			//printf("%d receive seed message\n", slave);
    			int queens[N];

    			if(!collide(0,seed/(size*size),1,(seed/size)%size)&&!collide(0,seed/(size*size),2,seed%size)&&!collide(1,(seed/size)%size,2,seed%size)){
					queens[0] = seed/(size*size);
					queens[1] = (seed/size)%size;
					queens[2] = seed%size;
					place(size,3,queens);
		    	}

    			MPI_Send(&finished, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);

			}else{//receive terminate from master, stop then
				MPI_Send(&result, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);
				MPI_Send(&solutions, 1, MPI_INT, 0, NUM_SOLUTIONS, MPI_COMM_WORLD);
				done = true;
			}

    	}

    }

    MPI_Finalize();
    //printf("from slave %d ,num of solutions are %d\n",rank,solutions);
    return 0;
}



/*
int main(int argc, char *argv[]){
	int size;
	clock_t start,finish;
	start = clock();
	int queens[N];
	printf("num of prob1em size: ",&size);
	scanf("%d",&size);
	int seed = size * size *  size -1;
	for (int i = seed; i>0 ; i--){
		if(!collide(0,i/(size*size),1,(i/size)%size)&&!collide(0,i/(size*size),2,i%size)&&!collide(1,(i/size)%size,2,i%size)){
			printf("seed : %d    ,",i);
			//solutions = 0;
			queens[0] = i/(size*size);
			queens[1] = (i/size)%size;
			queens[2] = i%size;
			place(size,3,queens);
			printf("rows (%d,%d,%d),",queens[0],queens[1],queens[2]);
    	}
	}
	printf("num of solutions are %d\n",solutions);
	finish = clock();
	printf("finish the work in %d seconds\n",(finish - start)/1000	);

	return 0;
}

*/



