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
int place(int size,int row,int* queens) {
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
	return solutions;
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

/*
//可以 openmp
std::vector<int> generate(int size) {
	std::vector<int> result(size*size*size*3);
	int count = 0;
	for (int i = 0; i < size; i++){
		for (int j = 0; i < size; j++){
			for (int k = 0; k < size; k++){
				if(check(size,i,j,k)){
					printf("working now for %d queens problem!\n",size);	
					result[count*3] = i;
					result[count*3+1] = j;
					result[count*3+2] = k;
					count++;
				}

			}
		}
	}
	result(size*size*size*3) = count;
	return result;	
}*/


int main(int argc, char  *argv[]){	
	//int solutions = 0;	// number of solutions
	int size = 8;	        // init size of problem as 8
	int reply;	
	int slave;
	int seeds = size * size * size -1;

	//mpi message type
	int ready = READY;
	int finished = FINISHED;
	int newTask = NEW_TASK;
	int terminate = TERMINATE;
	int solutionCount = 0;
	



	double startTime,endTime;
    startTime = MPI_Wtime();

	int rank, MPIsize;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

    if(rank == 0) {
    	MPI_Status masterStatus;
    	int salves = MPIsize -1;
    	int num_solutions;
    	while(salves){
    		MPI_Recv(&reply, 1, MPI_INT, MPI_ANY_SOURCE, REPLY, MPI_COMM_WORLD, &masterStatus);
    		slave = masterStatus.MPI_SOURCE;
    		printf("receive notice from slave %d\n", slave );

    		if(reply == FINISHED){
    			MPI_Recv(&num_solutions, 1, MPI_INT, slave, NUM_SOLUTIONS, MPI_COMM_WORLD, &masterStatus);
    			solutionCount += num_solutions;  
    			printf("%d slave said it finished its work\n",slave ); 			  		
    		}

    		if (seeds){
    			MPI_Send(&newTask, 1, MPI_INT, slave, REQUEST, MPI_COMM_WORLD);

    			MPI_Send(&seeds, 1, MPI_INT, slave, SEED, MPI_COMM_WORLD);
    			printf("send seed to salve %d\n", slave );
    			seeds --;
    		}else{

    			MPI_Send(&terminate, 1, MPI_INT, slave, REQUEST, MPI_COMM_WORLD);
    			printf("message to terminate slave %d\n", slave );
    			salves --;
    		}

    	}

    }else{
    	MPI_Status slaveStatus;
    	bool done = false;
    	int my_solutions = 0;
    	int request;
    	int seed;

    	MPI_Send(&ready, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);

    	while(!done){
    		MPI_Recv(&request, 1, MPI_INT, 0, REQUEST, MPI_COMM_WORLD, &slaveStatus);

    		if(request == NEW_TASK){
    			MPI_Recv(&seed, 1, MPI_INT, 0, SEED, MPI_COMM_WORLD, &slaveStatus);
    			printf("%d receive seed message\n", slave);
    			int queens[N];

    			if(!collide(0,seed/(size*size),1,(seed/size)%size)&&!collide(0,seed/(size*size),2,seed%size)&&!collide(1,(seed/size)%size,2,seed%size)){
					queens[0] = seed/(size*size);
					queens[1] = (seed/size)%size;
					queens[2] = seed%size;
					my_solutions = place(size,3,queens);
		    	}

    			MPI_Send(&finished, 1, MPI_INT, 0, REPLY, MPI_COMM_WORLD);


    			MPI_Send(&my_solutions, 1, MPI_INT, 0, NUM_SOLUTIONS, MPI_COMM_WORLD);
    			//printf("for seed %d ,slave %d find %d solutions.\n",seed,rank,my_solutions);
			}else{//receive terminate from master, stop then
				done = true;
			}

    	}

    }

    MPI_Finalize();
    printf("num of solutions are %d\n",solutionCount);
    return 0;
}

/*
int main(int argc, char *argv[])
{
	clock_t start,finish;
	int size;
	//int my_solutions = 0;
	printf("num of prob1em size: ",&size);
	scanf("%d",&size);
	start = clock();
	printf("working now for %d queens problem!\n",size);
	//int seed = generate(size);
	//printf("num of seeds are %d\n",seed);
	/*std::vector<int> result =  generate(size);
	int count = result[size*size*size*3];
	printf("seeds are %d !\n",count);
	for (int i = 0; i < count; i++)
	{
		shuffle(size);
		printf("working for the %d board\n",i);
		queens[0] = result[3*i];
		queens[1] = result[3*i+1];
		queens[2] = result[3*i+2];
		my_solutions += place(size,4);
	}
	place(size,0);
	printf("num of solutions are %d\n",solutions);
	finish = clock();
	printf("finish the work in %d seconds\n",finish - start);

	return 0;
}*/

/*
int main(int argc, char *argv[]){
	int size;
	clock_t start,finish;
	start = clock();
	//int my_solutions = 0;
	int queens[N];
	printf("num of prob1em size: ",&size);
	scanf("%d",&size);
	int seed = size * size *  size -1;
	for (int i = seed; i>0 ; i--){
		if(!collide(0,i/(size*size),1,(i/size)%size)&&!collide(0,i/(size*size),2,i%size)&&!collide(1,(i/size)%size,2,i%size)){
			printf("%d    ,",i);
			queens[0] = i/(size*size);
			queens[1] = (i/size)%size;
			queens[2] = i%size;
			place(size,3,queens);
			printf("(%d,%d,%d),",queens[0],queens[1],queens[2]);
			printf("%d\n",solutions);
    	}
	}
	printf("num of solutions are %d\n",solutions);
	finish = clock();
	printf("finish the work in %d seconds\n",(finish - start)/1000	);

	return 0;
}*/





