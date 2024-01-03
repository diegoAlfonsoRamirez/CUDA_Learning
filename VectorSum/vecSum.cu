#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N (1000000)
#define threads_per_block 1024 

__global__ void vecSum(int *a, int *b, int *c, int n){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(id < n){
      c[id] = a[id] + b[id];
  }
}

void vecFill(int *a, int n){
  for(int i = 0; i < n; i++){
      a[i] = rand() % 100000;
  }
}

int main(){
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;
  int size = N*sizeof(int);

  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  h_a = (int*)malloc(size);
  h_b = (int*)malloc(size);
  h_c = (int*)malloc(size);

  vecFill(h_a, N);
  vecFill(h_b, N);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  clock_t gputime = clock();

  vecSum<<<N / threads_per_block, threads_per_block>>>(d_a, d_b, d_c, N);

  cudaDeviceSynchronize();

  printf("Time of GPU vector sum for a %d sized vector: %f\n", N, ((double)clock() - gputime) / CLOCKS_PER_SEC);

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}