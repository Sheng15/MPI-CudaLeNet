#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <time.h>

const int N = 26;//max of size if 26
//int queens[N]; // board of a game 
int solutions = 0;

enum messageType
{
	READY,
	FINISHED,
	NEW_TASK,
	TERMINATE,
	RESULT
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





int main(int argc, char *argv[]){
	int size;
	sscanf(argv[1], "%d", &size); 
	clock_t start,finish;
	start = clock();
	int queens[N];

	place(size,0,queens);
	printf("num of solutions are %d\n",solutions);
	finish = clock();
	printf("finish the work in %ld seconds\n",(finish - start)/1000	);

	return 0;
}

