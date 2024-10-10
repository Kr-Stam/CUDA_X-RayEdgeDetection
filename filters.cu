#include "kernels.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.hpp"
#include "filters.cuh"

///used for the bilateral filter
__constant__ double gaus_kernel_10x10_gpu[100];

__global__ void g_bilinear_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaB)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < 0 || y < 0 || x >= w || y >= h){
			return;
	}

	int hwh = wh >> 1;
	int hww = ww >> 1;

	const 
	double* gaus_mask = gaus_kernel_10x10_gpu;

	int pos = y * w + x;
	double wsb = 0;

	int start_y = y - hwh;
	int start_x = x - hww;

	double f_ij = gray[pos * 3];

	double tmp[3] = {0, 0, 0};
	for (int m = 0; m < wh; m++)
	{
			int c_y = start_y + m;
			if (c_y < 0 || c_y >= h)
			{
					continue;
			}
			for (int n = 0; n < ww; n++)
			{
					double sigmaB2 = sigmaB * sigmaB;

					int c_x = start_x + n;

					if (c_x < 0 || c_x >= w)
					{
							continue;
					}

					int c_pos = c_y * w + c_x;

					double f_mn = gray[c_pos * 3];
					double k = f_mn - f_ij;
					double k2 = k * k;

					double n_b = 1.0 / (2.0 * M_PI * sigmaB2) * pow(M_E, -0.5 * (k2) / sigmaB2);
					double n_s = gaus_mask[m * ww + n];

					wsb += n_b * n_s;
					tmp[0] += src[c_pos * 3] * n_b * n_s;
					tmp[1] += src[c_pos * 3 + 1] * n_b * n_s;
					tmp[2] += src[c_pos * 3 + 2] * n_b * n_s;
			}
	}
	tmp[0] /= wsb;
	tmp[1] /= wsb;
	tmp[2] /= wsb;

	dest[pos * 3] = (unsigned char)tmp[0];
	dest[pos * 3 + 1] = (unsigned char)tmp[1];
	dest[pos * 3 + 2] = (unsigned char)tmp[2];
}

void gpu::bilateral_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaS, double sigmaB)
{
	double* gaus_mask = (double*) malloc(ww * wh * sizeof(double)); 
	utils::generate_gaussian_kernel(sigmaS, ww, gaus_mask);

	unsigned char* src_d;
	unsigned char* gray_d;
	unsigned char* dest_d;
	
	cudaMalloc((void**) &src_d, w * h * 3 * sizeof(unsigned char));
	cudaMalloc((void**) &gray_d, w * h * 3 * sizeof(unsigned char));
	cudaMalloc((void**) &dest_d, w * h * 3 * sizeof(unsigned char));

	cudaMemcpyToSymbol(gaus_kernel_10x10_gpu, gaus_mask, ww * wh * sizeof(double));

	cudaMemcpy(src_d, src, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(gray_d, gray, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int) ceil((float) w / (float) NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil((float) h / (float) NUM_OF_THREADS);
	dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

	g_bilinear_filter<<<blockSize, gridSize>>>(src_d, gray_d, dest_d, w, h, ww, wh, sigmaB);

	cudaMemcpy(dest, dest_d, w * h * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(gray_d);
	cudaFree(dest_d);

	free(gaus_mask);
}
