#include "grayscale.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
* \brief CUDA kernel that creates a grayscale image using the average rgb values, each block is a line
*
* \param src Source Image
* \param dest Destination
* \param w Width
* \param h Height
*
* \details IMPORTANT: This kernel should be called with a 1D block, each block is one line of the image
*/
__global__ void g_grayscale_avg_1d(const unsigned char *src, unsigned char *dest, int w, int h)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	if (x >= w || y >= h)
	{
			return;
	}

	int pos = (y * w + x) * 3;
	int avg = (src[pos] + src[pos + 1] + src[pos + 2]) / 3;
	dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char)avg;
}

/**
* \brief CUDA kernel that creates a grayscale image using the average rgb values, each block is a rectangle
*
* \param src Source Image
* \param dest Destination
* \param w Width
* \param h Height
*
* \details IMPORTANT: This kernel should be called with a 2D block, each block is a square of the image
*/
__global__ void g_grayscale_avg_2d(const unsigned char *src, unsigned char *dest, int w, int h)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
			return;
	}

	int pos = (y * w + x) * 3;
	int avg = (src[pos] + src[pos + 1] + src[pos + 2]) / 3;
	dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char)avg;
}

/**
* \brief Launches a CUDA kernel to grayscale an image
*
* \param srch Source Image
* \param dest_h Destination
* \param w Width
* \param h Height
*
*/
void gpu::grayscale_avg(const unsigned char *src_h, unsigned char *dest_h, int h, int w)
{

	unsigned char *src_d;
	unsigned char *dest_d;

	size_t size = h * w * 3 * sizeof(unsigned char);

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	dim3 block_size = dim3(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
	dim3 grid_size(GRID_SIZE_X, GRID_SIZE_Y);
	g_grayscale_avg_2d<<<grid_size, block_size>>>(src_d, dest_d, w, h);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(dest_d);
	cudaFree(src_d);
}

/**
* \brief Launches a CUDA kernel to grayscale an image and reduce the color channels from 3 to 1
*
* \param srch Source Image
* \param dest_h Destination
* \param w Width
* \param h Height
*
*/
__global__ void g_grayscale_avg_3ch_1ch(const unsigned char *src_h, unsigned char *dest_h, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h)
		return;

	int pos = y * w + x;

	dest_h[pos] = (src_h[pos*3] + src_h[pos*3+1] + src_h[pos*3+2]) / 3;
}

/**
* \brief Launches a CUDA kernel to grayscale an image and reduce the channels from 3 to 1
*
* \param srch Source Image
* \param dest_h Destination
* \param w Width
* \param h Height
*
*/
void gpu::grayscale_avg_3ch_1ch(const unsigned char *src_h, unsigned char *dest_h, int w, int h)
{
	unsigned char *src_d;
	unsigned char *dest_d;

	size_t size = h * w * sizeof(unsigned char);

	cudaMalloc((void **) &src_d, size * 3);
	cudaMalloc((void **) &dest_d, size);

	cudaMemcpy(src_d, src_h, size * 3, cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	dim3 block_size = dim3(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int) ceil(((float) w) / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil(((float) h) / NUM_OF_THREADS);
	dim3 grid_size(GRID_SIZE_X, GRID_SIZE_Y);
	g_grayscale_avg_3ch_1ch<<<grid_size, block_size>>>(src_d, dest_d, w, h);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(dest_d);
	cudaFree(src_d);
}
