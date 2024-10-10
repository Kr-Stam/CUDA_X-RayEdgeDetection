#include "convolutions.cuh"
#include "utils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

///constant memory used for mask convolution
__constant__ double c_mask[25];

__constant__ double GAUS_KERNEL_3x3_d[9] = {
	0.0625, 0.125, 0.0625,
	0.125, 0.25, 0.125,
	0.0625, 0.125, 0.0625
};


//----------------CONVOLUTIONS--------------------

//-------------------3CHANNEL---------------------
/**
* \brief Unoptimized CUDA kernel for 2D convolution
*
* \param src Source Matrix
* \param mask Mask Matrix
* \param dest Destination Matrix
* \param w Width
* \param h Heigth
* \param mw Mask Width
* \param mh Mask Height
*/
__global__ void g_conv_3ch_2d(const unsigned char *src, const double *mask, unsigned char *dest, int w, int h, int mw, int mh)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h)
			return;

	int pos = y * w + x;

	int tmp[3] = {0, 0, 0};

	int hmw = mw >> 1;
	int hmh = mh >> 1;
	int start_x = x - hmw;
	int start_y = y - hmh;
	int tmp_pos, mask_pos, tmp_x, tmp_y;

	for (int i = 0; i < mh; i++)
	{
			for (int j = 0; j < mw; j++)
			{
					tmp_x = start_x + j;
					tmp_y = start_y + i;
					if (tmp_x >= 0 && tmp_x < w && tmp_y >= 0 && tmp_y < h)
					{
							tmp_pos = tmp_y * w + tmp_x;
							mask_pos = i * mw + j;
							tmp[0] += src[tmp_pos * 3 + 0] * mask[mask_pos];
							tmp[1] += src[tmp_pos * 3 + 1] * mask[mask_pos];
							tmp[2] += src[tmp_pos * 3 + 2] * mask[mask_pos];
					}
			}
	}
	dest[pos * 3 + 0] = (unsigned char)tmp[0];
	dest[pos * 3 + 1] = (unsigned char)tmp[1];
	dest[pos * 3 + 2] = (unsigned char)tmp[2];
}

/**
* \brief Launch a CUDA kernel to perform 2D convolution
*
* \param src Source Matrix
* \param dest Destination Matrix
* \param w Width
* \param h Height
* \param mask_t Mask Matrix
* \param mw Mask Width <=5
* \param mh Mask Height <=5
*/
void gpu::conv_3ch_2d(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh)
{

	size_t size = w * h * 3 * sizeof(unsigned char);
	size_t mask_size = mw * mh * sizeof(double);

	unsigned char *src_d;
	unsigned char *dest_d;
	double *mask_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);
	cudaMalloc((void **)&mask_d, mask_size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask_t, mask_size, cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
	dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_3ch_2d<<<blockSize, gridSize>>>(src_d, mask_d, dest_d, w, h, mw, mh);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
	cudaFree(mask_d);
}

/**
* \brief A more optimized 2D convolution where the mask is loaded into constant GPU memory before execution
*
* \param src Source Matrix
* \param dest Destination Matrix
* \param w Width
* \param h Height
* \param mw Mask Width
* \param mh Mask Height
*/ 
__global__ void g_conv_3ch_2d_constant(const unsigned char *src, unsigned char *dest, int w, int h, int mw, int mh)
{

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h)
			return;

	int pos = y * w + x;

	int tmp[3] = {0, 0, 0};

	int hmw = mw >> 1;
	int hmh = mh >> 1;
	int start_x = x - hmw;
	int start_y = y - hmh;
	int tmp_pos, mask_pos, tmp_x, tmp_y;

	for (int i = 0; i < mh; i++)
	{
			for (int j = 0; j < mw; j++)
			{
					tmp_x = start_x + j;
					tmp_y = start_y + i;
					if (tmp_x >= 0 && tmp_x < w && tmp_y >= 0 && tmp_y < h)
					{
							tmp_pos = tmp_y * w + tmp_x;
							mask_pos = i * mw + j;
							tmp[0] += src[tmp_pos * 3 + 0] * c_mask[mask_pos];
							tmp[1] += src[tmp_pos * 3 + 1] * c_mask[mask_pos];
							tmp[2] += src[tmp_pos * 3 + 2] * c_mask[mask_pos];
					}
			}
	}
	dest[pos * 3 + 0] = (unsigned char)tmp[0];
	dest[pos * 3 + 1] = (unsigned char)tmp[1];
	dest[pos * 3 + 2] = (unsigned char)tmp[2];
}

/**
* \brief Launch a CUDA kernel to perform a 2D convolution with constant memory
*
* \param src Source Matrix
* \param dest Destination Matrix
* \param w Width
* \param h Height
* \param mask_t Mask Matrix
* \param mw Mask Width <=5
* \param mh Mask Height <=5
*/
void gpu::conv_3ch_2d_constant(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh)
{
	size_t size = w * h * 3 * sizeof(unsigned char);
	size_t mask_size = mw * mh * sizeof(double);

	unsigned char *src_d;
	unsigned char *dest_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_mask, mask_t, mask_size);

	int NUM_OF_THREADS = 32;
	dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
	dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_3ch_2d_constant<<<blockSize, gridSize>>>(src_d, dest_d, w, h, mw, mh);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
}

__global__ void g_conv_3ch_tiled(const unsigned char *src, unsigned char *dest, int w, int h, int mw, int mh, int TILE_SIZE_X, int TILE_SIZE_Y){
	//load all data
	//Objasnuvanje za kako raboti, povekje e ova za licna upotreba
	//Se upotrebuva maksimalniot mozhen blockSize shto e 32x32
	//Se loadiraat site vrednosti vnatre vo toj blockSize
	//Se koristi TILE_SIZE shto e 32-mw+1;
	//Za da se loadiraat vrednosti nadvor od src mora da se napravat input indeksi i output indeksi
	//Mapiranjeto na nivo na thread e out(0,0) e na TILE_SIZE, in(0,0) e na BLOCK_SIZE
	//Site threads loadiraat, ama ako threadot e nadvor od TILE_SIZE togash ne e output thread 

	extern __shared__ unsigned char tile[];    

	int hmh = mh >> 1;
	int hmw = mw >> 1;

	int x_o = threadIdx.x + blockIdx.x * TILE_SIZE_X;
	int y_o = threadIdx.y + blockIdx.y * TILE_SIZE_Y;
	int pos_o = x_o + y_o * w; 
	int x_i = x_o - hmw;
	int y_i = y_o - hmh;

	int tile_pos = threadIdx.x + threadIdx.y * blockDim.x;
	if(x_i < 0 || x_i >= w || y_i < 0 || y_i >= h){
			tile[tile_pos * 3] = tile[tile_pos * 3 + 1] = tile[tile_pos * 3 + 2] = 0;
	}else{
			int pos_i = x_i + y_i * w;
			tile[tile_pos * 3 + 0] = src[pos_i * 3];
			tile[tile_pos * 3 + 1] = src[pos_i * 3 + 1];
			tile[tile_pos * 3 + 2] = src[pos_i * 3 + 2];
	}

	__syncthreads();

	if(x_o >= w || y_o >= h)
			return;

	if(threadIdx.x >= TILE_SIZE_X || threadIdx.y >= TILE_SIZE_Y){
			return;
	}

	int tmp_x, tmp_y, tmp_pos, mask_pos;
	double tmp[] = {0, 0, 0};
	for(int i = 0; i < mh; i++){
			tmp_y = threadIdx.y + i;
			for(int j = 0; j < mw; j++){
					tmp_x = threadIdx.x + j;
					tmp_pos = tmp_x + tmp_y * blockDim.x;
					mask_pos = j + i * mw;
					tmp[0] += tile[tmp_pos * 3 + 0] * c_mask[mask_pos];
					tmp[1] += tile[tmp_pos * 3 + 1] * c_mask[mask_pos];
					tmp[2] += tile[tmp_pos * 3 + 2] * c_mask[mask_pos];
			}
	}
	dest[pos_o * 3] = (unsigned char) tmp[0]; 
	dest[pos_o * 3 + 1] = (unsigned char) tmp[1]; 
	dest[pos_o * 3 + 2] = (unsigned char) tmp[2]; 

	//Tile e indeksiran na nivo na block
	//Odma gi isfrlame site outputs shto se out of bounds na src    
	//
}

void gpu::conv_3ch_tiled(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const double *mask_t, int mw, int mh)
{
	size_t size = w * h * 3 * sizeof(unsigned char);

	unsigned char *src_d;
	unsigned char *dest_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_mask, mask_t, mw * mh * sizeof(double));

	int NUM_OF_THREADS = 32;
	int TILE_SIZE_X = NUM_OF_THREADS - mw + 1;
	int TILE_SIZE_Y = NUM_OF_THREADS - mh + 1;
	dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
	//? Mozhe da se optimizira ova
	int GRID_SIZE_X = (int)ceil((float)w / TILE_SIZE_X);
	int GRID_SIZE_Y = (int)ceil((float)h / TILE_SIZE_Y);
	dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_3ch_tiled<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(unsigned char) * 3>>>(src_d, dest_d, w, h, mw, mh, TILE_SIZE_X, TILE_SIZE_Y);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
}

//-------------------1CHANNEL---------------------
__global__ void g_conv(const unsigned char* src, unsigned char* dest, int w, int h, const double* mask, int mw, int mh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h)
		return;

	int pos = y * w + x;

	int hmw = mw >> 1;
	int hmh = mh >> 1;

	int start_x = x - hmw;
	int start_y = y - hmh;

	int tmp_y, tmp_x, tmp_pos, mask_pos;
	double tmp = 0;
	for(int i = 0; i < mh; i++)
	{
		tmp_y = start_y + i;
		if(tmp_y < 0 || tmp_y >= h)
			continue;

		for(int j = 0; j < mw; j++)
		{
			tmp_x = start_x + j;
			if(tmp_x < 0 || tmp_x >= w)
				continue;

			tmp_pos = tmp_y * w + tmp_x;
			mask_pos = i * mw + j;
			tmp += src[tmp_pos] * mask[mask_pos];
		}
	}

	dest[pos] = (unsigned char) tmp;
}

void gpu::conv(const unsigned char* src, unsigned char* dest, int w, int h, const double* mask, int mw, int mh)
{
	unsigned char* src_d;
	unsigned char* dest_d;
	double* mask_d;

	size_t size = w * h * sizeof(unsigned char);
	size_t mask_size = mw * mh * sizeof(double);

	cudaMalloc((void**) &src_d, size);
	cudaMalloc((void**) &dest_d, size);
	cudaMalloc((void**) &mask_d, mask_size);

	cudaMemcpy(src_d, src, size, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, mask_size, cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	dim3 BLOCK_SIZE(NUM_OF_THREADS, NUM_OF_THREADS); 
	int GRID_SIZE_X = (int) ceil(((float) w) / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil(((float) h) / NUM_OF_THREADS);
	dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);

	g_conv<<<GRID_SIZE, BLOCK_SIZE>>>(src_d, dest_d, w, h, mask_d, mw, mh);

	cudaMemcpy(dest, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
	cudaFree(mask_d);
}

__global__ void g_conv_constant(const unsigned char* src, unsigned char* dest, int w, int h, int mw, int mh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h)
		return;

	int pos = y * w + x;

	int hmw = mw >> 1;
	int hmh = mh >> 1;

	int start_x = x - hmw;
	int start_y = y - hmh;

	int tmp_y, tmp_x, tmp_pos, mask_pos;
	double tmp = 0;
	for(int i = 0; i < mh; i++)
	{
		tmp_y = start_y + i;
		if(tmp_y < 0 || tmp_y >= h)
			continue;

		for(int j = 0; j < mw; j++)
		{
			tmp_x = start_x + j;
			if(tmp_x < 0 || tmp_x >= w)
				continue;

			tmp_pos = tmp_y * w + tmp_x;
			mask_pos = i * mw + j;
			tmp += src[tmp_pos] * c_mask[mask_pos];
		}
	}

	dest[pos] = (unsigned char) tmp;
}

void gpu::conv_constant(const unsigned char* src_h, unsigned char* dest_h, int w, int h, const double* mask, int mw, int mh)
{
	size_t size = w * h * sizeof(unsigned char);
	size_t mask_size = mw * mh * sizeof(double);

	unsigned char *src_d;
	unsigned char *dest_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_mask, mask, mask_size);

	int NUM_OF_THREADS = 32;
	const dim3 BLOCK_SIZE(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int) ceil(((float) w) / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil(((float) h) / NUM_OF_THREADS);
	const dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_constant<<<BLOCK_SIZE, GRID_SIZE>>>(src_d, dest_d, w, h, mw, mh);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
}

__global__ void g_conv_tiled(const unsigned char* src, unsigned char* dest, int w, int h, int mw, int mh) 
{
	extern __shared__ unsigned char tile[];

	//position that the tile has to load into memory
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h)
		return;

	int hmw = mw >> 1;
	int hmh = mh >> 1;
	//position in the shared memory tile
	int x_t = threadIdx.x + hmw;
	int y_t = threadIdx.y + hmh;

	//offsets of additional loads
	int x_offset = 0;
	int y_offset = 0;

	//top and bottom excess
	if(x_t / hmw == 0)
	{
		x_offset = - x_t % hmw;
	}
	else if((blockDim.x - x_t) / hmw == 0)
	{
		x_offset = (blockDim.x - x_t) % hmw;
	}

	//left and right excess
	if(y_t / hmh == 0)
	{
		y_offset = - y_t % hmh;
	}
	else if((blockDim.y - y_t) / hmh == 0)
	{
		y_offset = (blockDim.y - y_t) % hmh;
	}

	//corner offsets
	if(x_t / hmw == 1)
	{
		x_offset = - hmw - x_t % hmw;
	}
	else if((blockDim.x - x_t) / hmw == 1)
	{
		x_offset = hmw + (blockDim.x - x_t) % hmw;
	}
	if(y_t / hmh == 1)
	{
		y_offset = - hmh - y_t % hmh;
	}
	else if((blockDim.y - y_t) / hmh == 1)
	{
		y_offset = hmh + (blockDim.y - y_t) % hmh;
	}

	//load regular tile
	int tile_pos = y_t * blockDim.x + x_t;
	int src_pos  = y * w + x;
	tile[tile_pos] = src[src_pos];

	//deka ne se odnesuva na statichna memorija mozhe ova slobodno da ne se proveri
	int offset_tile_pos = (y_t + y_offset) * blockDim.x + x_t + x_offset;

	int offset_x = x + x_offset;
	int offset_y = y + y_offset;
	int offset_src_pos = offset_y * w + offset_x;
	if(offset_x < 0 || offset_x >= w || offset_y < 0 || offset_y >= h)
		tile[offset_tile_pos] = 0;
	else
		tile[offset_tile_pos] = src[offset_src_pos];

	__syncthreads();
	
	int pos = src_pos;
	int start_x = x_t - hmw;
	int start_y = y_t - hmh;
	//isto e so gornoto
	//int start_x = threadIdx.x;
	//int start_y = threadIdx.y;
	double tmp = 0;

	//ne mora checks deka znam deka sekogash se validni poziciite vo tile
	for(int i = 0; i < mh; i++)
	{
		for(int j = 0; j < mw; j++)
		{
			if(start_x < hmw || start_x >= blockDim.x + hmw || 
			   start_y < hmh || start_y >= blockDim.y + hmh)
				continue;
			int mask_pos = i * mw + j;
			int tmp_tile_pos = (start_y + i) * blockDim.x + start_x + j;
			tmp += c_mask[mask_pos] * tmp_tile_pos;
		}
	}

	dest[pos] = (unsigned char) tmp;
}

//TODO: Ova ne e celosno testirano, bi trebalo da raboti ama ne sum siguren
void gpu::conv_tiled(const unsigned char* src_h, unsigned char* dest_h, int w, int h, const double* mask, int mw, int mh)
{
	size_t size = w * h * sizeof(unsigned char);
	size_t mask_size = mw * mh * sizeof(double);

	unsigned char *src_d;
	unsigned char *dest_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_mask, mask, mask_size);

	int NUM_OF_THREADS = 32;
	const dim3 BLOCK_SIZE(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int) ceil(((float) w) / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil(((float) h) / NUM_OF_THREADS);
	const dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_tiled<<<BLOCK_SIZE, GRID_SIZE, (BLOCK_SIZE.x + mw) * (BLOCK_SIZE.y + mh) * sizeof(unsigned char)>>>(src_d, dest_d, w, h, mw, mh);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
}

__global__ void g_conv_range(const unsigned char* src, unsigned char* dest, int w, int h, int mw, int mh)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h)
		return;

	int pos = y * w + x;

	int hmw = mw >> 1;
	int hmh = mh >> 1;

	int start_x = x - hmw;
	int start_y = y - hmh;

	int tmp_y, tmp_x, tmp_pos;

	int min = UCHAR_MAX;
	int max = 0;
	for(int i = 0; i < mh; i++)
	{
		tmp_y = start_y + i;
		if(tmp_y < 0 || tmp_y >= h)
		{
			//min = 0;
			continue;
		}

		for(int j = 0; j < mw; j++)
		{
			tmp_x = start_x + j;
			if(tmp_x < 0 || tmp_x >= w)
			{
				//min = 0;
				continue;
			}

			tmp_pos = tmp_y * w + tmp_x;
			if(src[tmp_pos] < min)
				min = src[tmp_pos];
			if(src[tmp_pos] > max)
				max = src[tmp_pos];
		}
	}

	int range = max - min;
	if(range != 0)
		dest[pos] = 0;
	else
		dest[pos] = 255;
}

void gpu::conv_range(const unsigned char* src_h, unsigned char* dest_h, int w, int h, int mw, int mh)
{
	size_t size = w * h * sizeof(unsigned char);

	unsigned char *src_d;
	unsigned char *dest_d;

	cudaMalloc((void **)&src_d, size);
	cudaMalloc((void **)&dest_d, size);

	cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);

	int NUM_OF_THREADS = 32;
	const dim3 BLOCK_SIZE(NUM_OF_THREADS, NUM_OF_THREADS);
	int GRID_SIZE_X = (int) ceil(((float) w) / NUM_OF_THREADS);
	int GRID_SIZE_Y = (int) ceil(((float) h) / NUM_OF_THREADS);
	const dim3 GRID_SIZE(GRID_SIZE_X, GRID_SIZE_Y);
	g_conv_range<<<BLOCK_SIZE, GRID_SIZE>>>(src_d, dest_d, w, h, mw, mh);

	cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

	cudaFree(src_d);
	cudaFree(dest_d);
}
