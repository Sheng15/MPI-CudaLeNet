#include <iostream>
#include <vector>
#include <stdint.h>

using std::cout;
using std::endl;
using std::cin;

bool collide(int row1, int col1, int row2, int col2){
    return (row1==row2)||(col1==col2)||(row1+col1 == col1+col2)||(row1-row2 == col1-col2);
}


class ChessBoard{
public:

    int* _queens;

    
    ChessBoard(int size){
        _queens = new int[size];

        for(int i=0; i<size; i++){
            _queens[i] = -1;
        }
    }

    ~ChessBoard(){}

    int* getBoard(){
        return _queens;
    }
        
};


int generateBoard(int size){
    
}
















int main(int argc, char const *argv[]){

    int* _queens;
    int size = 10;
    _queens = new int[20];

    for(int i=0; i<10; i++){
            _queens[i] = 0;
        }

    for(int i=0; i<10; i++){
    cout<<"init:"<<endl;
    cout<<"init:"<<_queens[i]<<endl;
    }
    
}