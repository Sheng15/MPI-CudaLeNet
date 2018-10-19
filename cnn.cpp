#include "pch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>


//convCalculate(float ***inputArray,int *** kernelarray, float *** outputArray, int stride, float bias ){

float*** getPatch3D(float*** inputArray, int i, int j,int filterWidth, int filterHeight,int stride){
    int start_i = i * stride;
    int start_j = j * stride;
    int sizeOfInput = sizeof(inputArray);
    int depthOfInput = sizeof(inputArray[0]);
    int heightOfInput = sizeof(inputArray[1]);
    int widthOfInput = sizeof(inputArray[2]);

    int*** outputArray;

    for (int k = 0; k < depthOfInput; k++) {
		for (int i = 0; i < filterHeight; i++) {
			outputArray[k][i] = new float[filterWidth];
		}
	}

    for (int k = 0; k < depthOfInput; k++) {
		for (int i = start_i; i < start_i+filterHeight; i++) {
			for (int j = start_j; j < start_j+filterWidth; j++){
				outputArray[k][i-start_i][j-start_j] = inputArray[k][i][j]
			}
		}
	}

	return outputArray
}


float** getPatch2D(float*** inputArray, int i, int j,int filterWidth, int filterHeight,int stride){
    int start_i = i * stride;
    int start_j = j * stride;
    int sizeOfInput = sizeof(inputArray);
    int heightOfInput = sizeof(inputArray[0]);
    int widthOfInput = sizeof(inputArray[1]);

    int** outputArray;


	for (int i = 0; i < filterHeight; i++) {
		outputArray[k][i] = new float[filterWidth];
	}


	for (int i = start_i; i < start_i+filterHeight; i++) {
		for (int j = start_j; j < start_j+filterWidth; j++){
			outputArray[k][i-start_i][j-start_j] = inputArray[k][i][j]
		}
	}
	return outputArray
}




float max(float a, float b) {
	return (a < b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}



//a very small random float for initialization
float randomFloat()
{  
	int rand1 = rand() % 2;
	int rand2 = rand() % 2018 + 20182018;
	float rand3 = rand1 / rand2;
	return rand3;
};



class Activator
{
public:
	float forward(float weightedInput) {
		float result = max(0, weightedInput);
		return result;
	}

	float backward(float output) {
		if (output > 0) {
			return 1;
		}
		else {
			return 0;
		}
	}
};



class Filter
{
public:
	
	Filter(int width, int height, int depth) {
		filterWidth = width;
		filterHeight = height;
		filterDepth = depth;

		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < height; j++) {
				weights[i][j] = new float[width];
				weights_grad[i][j] = new float[width];
			}
		}
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < height; k++) {
					weights[i][j][k] = randomFloat();
					weights_grad[i][j][k] = 0;
				}
			}
		}

		bias = 0;
		bias_grad = 0;		
	}

	float *** getWeights() {
		return weights;
	}

	float getBias() {
		return bias;
	}

	void update(float learningRate) {
		for (int i = 0; i < filterDepth; i++) {
			for (int j = 0; j < filterWidth; j++) {
				for (int k = 0; k < filterHeight; k++) {
					weights[i][j][k] -= learningRate * weights_grad[i][j][k];
				}
			}
		}

		bias -= learningRate * bias_grad;
	}
	

private:
	float bias;
	float bias_grad;
	float *** weights;
	float *** weights_grad;
	int filterWidth;
	int filterHeight;
	int filterDepth;
};



int outputSize(int inputSize, int filterSize, int zeroPadding, int stride) {
	int size = (inputSize - filterSize + 2 * zeroPadding) / stride + 1;
	return size;
}

class ConvLayer
{
public:

	ConvLayer(int inputWidth, int inputHeight, int channelNumber, int filterWidth, int filterHeight, int filterNumber, int zeroPadding, int stride, Activator activator, float learningRate) {

		convInputWidth = inputWidth;
		convInputHeight = inputHeight;
		convChannelNumber = channelNumber;
		convFilterWidth = filterWidth;
		convFliterHeight = filterHeight;
		convFilterNumber = filterNumber;
		convZeroPadding = zeroPadding;
		convStride = stride;
		convActivator = activator;
		convLearningRate = learningRate;

		convOutputWidth = outputSize(inputWidth, filterWidth, zeroPadding, stride);
		convOutputHeight = outputSize(inputHeight, filterHeight, zeroPadding, stride);

		for (int i = 0; i < filterNumber; i++) {
			for (int j = 0; j < outputHeight; j++) {
				outputArray[i][j] = new float[outputWidth];
			}
		}

		for (int i = 0; i < filterNumber; i++) {
			for (int j = 0; j < outputHeight; j++) {
				for (int k = 0; k < outputWidth; k++) {
					outputArray[i][j][k] = 0;
				}
			}
		}


		for (int i = 0; i < filterNumber; i++) {
			Filter filter = Filter(filterWidth, filterHeight, filterNumber);
			filters[i] = filter;
		}
		
		void forward(this,float *** input) {
			inputArray = padding(input,this.zeroPadding);
			for(int i = 0; i< this.convFilterNumber; i++){
				Filter filter = this.filters[i];
				for(int j = 0; j< this.convOutputHeight; j++){
					for(int k = 0; k < this.convOutputWidth; k++){

					}
				}

			}


		def conv(input_array, 
         kernel_array,
         output_array, 
         stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (    
                get_patch(input_array, i, j, kernel_width, 
                    kernel_height, stride) * kernel_array
                ).sum() + bias



        conv(self.padded_input_array, 
                filter.get_weights(), self.output_array[f],
                self.stride, filter.get_bias())	



		}

	}




private:
	int convInputWidth,convInputHeight,convChannelNumber,convFilterWidth,convFliterHeight,convFilterNumber,convZeroPadding,convStride,convOutputWidth,convOutputHeight;
	Activator convActivator;
	float convLearningRate;
	float *** inputArray;
	float *** outputArray;
	Filter filters[100];
};




















int main()
{
	Filter f = Filter(3, 3, 3);
	float b = f.getBias;
	printf("%4.2f", b);
}

