#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __managed__ int  n;
__device__ __managed__ int range;

__global__ void bucket_sort(int *key, int *bucket, int n, int range) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) atomicAdd(&bucket[key[id]], 1);
  __syncthreads();
  int cum = 0;
  for(int i=0; i<range; i++)
    if (id<(cum+=bucket[i])) {
      key[id] = i;
      return;
    }
}

int main() {
  n = 50;
  range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  //-- Bucket Sort on GPU --
  const int NUM_THREADS = 7;
  const int NUM_BLOCKS = (n+NUM_THREADS-1)/NUM_THREADS;
  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  bucket_sort<<<NUM_BLOCKS, NUM_THREADS>>>(key, bucket, n, range);
  cudaDeviceSynchronize();
  //-- Bucket Sort on GPU --

  for (int i=0; i<n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
}
