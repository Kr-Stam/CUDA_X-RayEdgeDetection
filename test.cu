#include "test.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add_vec(int *a, int *b, int *c, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    c[idx] = a[idx] + b[idx];
}

int *generate_arr(int size) {
  int *a = (int*) malloc(size * sizeof(int));

  for (int i = 0; i < size; i++) {
    a[i] = i;
  }

  return a;
}

bool valid_size(int size) 
{
	return size > 0; 
}

void print_arr(int* arr, int size)
{
	for(int i = 0; i < size; i++)
	{
		printf("%d ", arr[i]);
		if(i != 0 && i % 20 == 0)
			printf("\n");
	}
	printf("\n");
}

void test::vector_add_test()
{
  int *a;
  int *b;
  int *c;

  int size;

  do {
    printf("Enter an arr size: ");
    scanf("%d", &size);

    if (!valid_size(size))
      printf("Please enter a valid size!\n");
  } while (!valid_size(size));

  a = generate_arr(size);
  b = generate_arr(size);
	c = (int*) malloc(size * sizeof(int));

  int *d_a;
  int *d_b;
  int *d_c;
  cudaMalloc((void**) &d_a, size * sizeof(int));
  cudaMalloc((void**) &d_b, size * sizeof(int));
  cudaMalloc((void**) &d_c, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block_size(1024);
	dim3 grid_size(size / block_size.x + 1);
	add_vec<<<grid_size, block_size>>>(d_a, d_b, d_c, size);

	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	printf("arr a: \n");
	print_arr(a, size);
	printf("arr b: \n");
	print_arr(b, size);
	printf("arr c: \n");
	print_arr(c, size);

  free(a);
  free(b);
  free(c);
}
