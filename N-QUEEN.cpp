#include <iostream>

using std::cout;
using std::endl;
using std::cin;

//Prints board
void print_board(int ** board, int n)
{
  cout << endl;
  for (int y = 0;y < n;y++)
  {
    for (int x = 0;x < n;x++)
    {
      cout << '\t' <<  board[y][x];
    }
    cout << endl << endl << endl << endl;
  }
  cout << endl;
}

//Checks if valid location to place a queen
bool valid(int ** board, int x, int y, int n)
{
  bool room_right = false;
  bool room_left = false;
  bool room_up = false;
  bool room_down = false;
  for (int i = 1;i < n;i++)
  {
    //RIGHT
    if ((x + i) < n)
    {
      room_right = true;
      if (board[y][x + i] == 1)
        return false;
    }
    //LEFT
    if ((x - i) >= 0)
    {
      room_left = true;
      if (board[y][x - i] == 1)
        return false;
    }
    //UP
    if ((y - i) >= 0)
    {
      room_up = true;
      if (board[y - i][x] == 1)
        return false;
    }
    //DOWN
    if ((y + i) < n)
    {
      room_down = true;
      if (board[y + i][x] == 1)
        return false;
    }
    //UP-RIGHT
    if (room_up && room_right)
    {
      if (board[y - i][x + i] == 1)
        return false;
    }
    //UP-LEFT
    if (room_up && room_left)
    {
      if (board[y - i][x - i] == 1)
        return false;
    }
    //DOWN-RIGHT
    if (room_down && room_right)
    {
      if (board[y + i][x + i] == 1)
        return false;
    }
    //DOWN-LEFT
    if (room_down && room_left)
    {
      if (board[y + i][x - i] == 1)
        return false;
    }
    room_up = false;
    room_down = false;
    room_left = false;
    room_right = false;
  }
  return true;
}

//Recursive backtracking function
//Does the heavy lifting
bool solve(int ** board, int x, int y, int n)
{
  static int num_queens = 0;
  //If N queens are on the board, it has been solved.
  if (num_queens == n)
    return true;

  //For every row in the board
  for (int i = 0;i < n;i++)
  {
    //If current row and col is a valid position
    if (valid(board, x, i, n))
    {
      //Place a queen at current position
      board[i][x] = 1;
      num_queens ++;

      //If we were unable to solve, remove the queen
      if (!solve(board, x + 1, i, n))
      {
        board[i][x] = 0;
        num_queens --;
      }
      else
        return true;
    }
  }
  return false;
}



int main()
{
  int n;
  bool solved = false;

  cout << "How many queens? : ";
  cin >> n;

  //Dynamically allocating a 2D array
  int ** board = new int * [n];

  for (int y = 0;y < n;y++)
  {
    board[y] = new int[n];
    for (int x = 0;x < n;x++)
      board[y][x] = 0;
  }

  //Entering recursive function
  solved = solve(board, 0, 0, n);

  if (solved)
  {
    cout << endl << "Solution Found:" << endl;
    print_board(board, n);
  }
  else
    cout << "No Solution." << endl;

  for (int y = 0;y < n;y++)
    delete [] board[y];

  delete [] board;
  return 0;
}