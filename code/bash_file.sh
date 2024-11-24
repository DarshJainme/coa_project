#!/bin/bash

# Variables
SOURCE_FILE="main_code.cpp"      
EXECUTABLE="main_code.exe"      

echo "Compiling $SOURCE_FILE with OpenMP support..."
g++ -std=c++14 -mavx2 -fopenmp -o $EXECUTABLE $SOURCE_FILE

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  read -p "Press Enter to exit..."
fi

echo "Compilation successful. Running the executable..."

# Run the executable
./$EXECUTABLE

echo "Execution completed."

read -p "Press Enter to exit..."