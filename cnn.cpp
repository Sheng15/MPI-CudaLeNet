#include "pch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>



float*** getPatch3D(float*** inputArray, int i, int j,int filterWidth, int filterHeight,int stride){
    int start_i = i * stride;
    int start_j = j * stride;
    int*** patchedArray;

    for (int k = 0; k < depthOfInput; k++) {
		for (int i = 0; i < filterHeight; i++) {
			patchedArray[k][i] = new float[filterWidth];
		}
	}

    for (int k = 0; k < depthOfInput; k++) {
		for (int i = start_i; i < start_i+filterHeight; i++) {
			for (int j = start_j; j < start_j+filterWidth; j++){
				patchedArray[k][i-start_i][j-start_j] = inputArray[k][i][j]
			}
		}
	}

	return patchedArray
}


float** getPatch2D(float** inputArray, int i, int j,int filterWidth, int filterHeight,int stride){
    int start_i = i * stride;
    int start_j = j * stride;


    int** patchedArray;


	for (int i = 0; i < filterHeight; i++) {
		patchedArray[k][i] = new float[filterWidth];
	}


	for (int i = start_i; i < start_i+filterHeight; i++) {
		for (int j = start_j; j < start_j+filterWidth; j++){
			patchedArray[k][i-start_i][j-start_j] = inputArray[k][i][j]
		}
	}
	return patchedArray
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
 bn
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
		
		void forward(this,float*** input) {
			this.inputArray = padding(input,this.zeroPadding);
			this.convChannelNumber = 3;
			for(int i = 0; i< this.convFilterNumber; i++){
				Filter filter = this.filters[i];
				for(int j = 0; j< this.convOutputHeight; j++){
					for(int k = 0; k < this.convOutputWidth; k++){
						float*** patchedArray =  getPatch3D(input,p,q,this.convFliterHeight,this.convFliterHeight,this.stride);
						int depthOfPatchedArray = sizeof(patchArray) / sizeof(patchArray[0]);
						int heightOfPatchedArray = sizeof(patchedArray[0]) / sizeof(patchedArray[0][0]);
						int widthOfPatchedArray = sizeof(patchedArray[0][0]) / sizeof(patchedArray[0][0][0]);
						for(int x=0; x < depthOfPatchedArray;x++){
							for(int y=0; y < depthOfPatchedArray;y++){
								for(int z=0; z < depthOfPatchedArray;z++){
									outputArray[x][y][z] += patchedArray[x][y][z]*filter[y][z];
								}
							}

						}
						

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


class MaxPoolingLayer {

	private int MPLInputWidth;
	private int MPLInputHeight;
	private int MPLChannelNumber;
	private int MPLFilterWidth;
	private int MPLFilterHeight;
	private int MPLStride;
	private int MPLOutputWidth;
	private int MPLOutputHeight;
	public float*** MPLOutputArray;
	public float*** MPLDeltaArray;

	MaxPoolingLayer(int inputWidth, int inputHeight, int channelNumber,
		int filterWidth, int filterHeight, int stride) {
		MPLInputWidth = inputWidth;
		MPLInputHeight = inputHeight;
		MPLChannelNumber = channelNumber;
		MPLFilterWidth = filterWidth;
		MPLFilterHeight = filterHeight;
		MPLStride = stride;
		MPLOutputWidth = (MPLInputWidth - MPLFilterWidth) / MPLStride + 1;
		MPLOutputHeight = (MPLInputHeight - MPLFilterHeight) / MPLSride + 1;
		for (int i = 0; i < MPLChannelNumber; i++) {
			for (int j = 0; j < MPLOutputHeight; j++) {
				for (int k = 0; k < MPLStride; k++) {
					MPLOutputArray[i] = 0;
				}
			}
		}
	}

	private float max(float **a) {
		int x = sizeof(a) / sizeof(a[0]);
		int y = sizeof(a[0]) / sizeof(a[0][0]);
		float max = 0.0;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (max < a[i][j])
					max = a[i][j];
			}
		}
		return max;
	}

	private int* getMaxIndex(float **a) {
		int x = sizeof(a) / sizeof(a[0]);
		int y = sizeof(a[0]) / sizeof(a[0][0]);
		int* coor = { 0,0 };
		float max = 0.0;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (max < a[i][j]) {
					max = a[i][j];
					coor[0] = i;
					coor[1] = j;
				}
			}
		}
		return coor;
	}

	void forward(float*** inputArray) {
		for (int d = 0; d < MPLChannelNumber; d++) {
			for (int i = 0; i < MPLOutputHeight; i++) {
				for (int j = 0; j < MPLOutputWidth; j++) {
					MPLOutputArray[d][i][j] = max(
						getPatch2D(inputArray[d], i, j,
							MPLFilterWidth, MPLFilterHeight, MPLStride));
				}
			}
		}
	}

	void backward(float*** inputArray, float*** sensitivityArray) {
		for (int d = 0; d < MPLChannelNumber; d++) {
			for (int i = 0; i < MPLOutputHeight; i++) {
				for (int j = 0; j < MPLOutputWidth; j++) {
					int** patchArray = getPatch2D(inputArray[d], i, j,
						MPLFilterWidth, MPLFilterHeight, MPLStride);
					int* MPLCoor= getMaxIndex(patchArray);
					MPLDeltaArray[d][i * MPLStride + MPLCoor[0]][j * MPLStride + MPLCoor[1]]
						= sensitivityArray[d][i][j];
				}
			}
		}
	}

}




















int main()
{
	Filter f = Filter(3, 3, 3);
	float b = f.getBias;
	printf("%4.2f", b);
}

