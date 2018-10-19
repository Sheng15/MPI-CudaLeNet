#include "pch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>


//convCalculate(float **inputArray,int *** kernelarray, )

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
			for (int j = 0; j < width; j++) {
				weights[i][j] = new float[height];
				weights_grad[i][j] = new float[height];
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

		int outputWidth = outputSize(inputWidth, filterWidth, zeroPadding, stride);
		int outputHeight = outputSize(inputHeight, filterHeight, zeroPadding, stride);

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
		
		void forward() {
			



		}

	}




private:
	int convInputWidth,convInputHeight,convChannelNumber,convFilterWidth,convFliterHeight,convFilterNumber,convZeroPadding,convStride;
	Activator convActivator;
	float convLearningRate;
	float 
	float *** outputArray;
	Filter filters[100];
};




















int main()
{
	Filter f = Filter(3, 3, 3);
	float b = f.getBias;
	printf("%4.2f", b);
}

