 #ifdef __CDT_PARSER__
 #define __global__
 #define __device__
 #define __shared__
 #endif

//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#include <stdint.h>
#include <limits.h>


__device__
int scan_op(int x, int y){
  return x+y;
}

__global__
void scan_exclusive(unsigned int* const filter, const size_t num)
{
  extern __shared__ unsigned int sdata[];

  unsigned int pos = threadIdx.x;
  unsigned int abs_pos = threadIdx.x + blockIdx.x*blockDim.x;

  if (abs_pos < num)
    sdata[pos] = filter[abs_pos];
  __syncthreads();

  for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1){
    bool use = (pos & (shift | shift-1)) == (shift | shift-1); // reset top bits, check that all lower bits are 1s.
    if(use && (abs_pos < num) && (pos - shift >= 0)) {
//      printf("%u [%02u]+[%02u]: (%u)+(%u) -> (%u)\n", shift, pos - shift, pos, sdata[pos - shift], sdata[pos], sdata[pos] + sdata[pos - shift]);
      sdata[pos] = scan_op(sdata[pos], sdata[pos - shift]);
    }
    __syncthreads();
  }

  if(pos == blockDim.x-1)
    sdata[pos] = 0;
  __syncthreads();

  if ((blockDim.x & blockDim.x-1) != 0) printf("ALLERT");

  unsigned int tmp;
  for (unsigned int shift = blockDim.x/2; shift > 0; shift >>= 1){
    bool use = (pos & (shift | shift-1)) == (shift | shift-1); // reset top bits, check that all lower bits are 1s.
    if(use && (abs_pos < num) && (pos - shift >= 0)) {
//      printf("%u [%02u],[%02u]:  %u,%u -> %u,%u\n", shift, pos - shift, pos, sdata[pos-shift], sdata[pos], sdata[pos], sdata[pos] + sdata[pos - shift]);
      tmp = sdata[pos-shift]+sdata[pos];
      sdata[pos-shift] = sdata[pos];
      sdata[pos] = tmp;
    }
    __syncthreads();
  }
//  printf("tid:%u %u\n", pos, sdata[pos]);

  if (abs_pos < num)
    filter[abs_pos] = sdata[pos];
}


__global__
void scatter(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
  int pos = threadIdx.x + blockIdx.x*blockDim.x;
  if(pos < numElems){
    d_outputPos[pos] = d_inputPos[pos];
    d_outputVals[pos] = d_inputVals[pos];
  }
}

void swap_arrays(unsigned int** a, unsigned int** b)
{
  unsigned int** tmp;
  tmp = a;
  a = b;
  b = tmp;
}

void round_up(size_t in, size_t* out){
  *out = 1;
  int i = 200;
  while (*out < in)
    *out <<= 1;
}

__global__
void map_offset(unsigned int* in, size_t count, size_t offset){
  unsigned int pos = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int source = blockIdx.x*blockDim.x;

  if(threadIdx.x == 0)
    in[pos] += offset;
  __syncthreads();
  if(pos < count && threadIdx.x != 0)
    in[pos] += in[source];
}

__global__
void reduce_sum(unsigned int* filter, size_t count, unsigned int* out){
  extern __shared__ unsigned int sdata[];
  int tid = threadIdx.x;
  int pos = threadIdx.x + blockIdx.x*blockDim.x;

  if(pos < count)
    sdata[tid] = filter[pos];
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s && pos+s < count)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if(tid == 0) {
    atomicAdd(&out[0], sdata[tid]);
//    printf("%d -> %d\n", pos, sdata[tid]);
  }
}

__global__
void map_equal(unsigned int* in, unsigned int mask, bool mask_match, unsigned int* filter, size_t count){
  unsigned int pos = threadIdx.x + blockDim.x*blockIdx.x;

  if (pos < count) {
    filter[pos] = (in[pos] & mask) == mask_match;
  }

}

__global__
void scatter_filtered(unsigned int* in, unsigned int* inVal,
                            unsigned int* filter, unsigned int* indexes,
                            size_t count,
                            unsigned int* out, unsigned int* outVal)
{
  unsigned int pos = threadIdx.x + blockDim.x*blockIdx.x;
  if (pos < count && filter[pos] > 0){
    unsigned int tar = indexes[pos];
    out[tar] = in[pos];
    outVal[tar] = inVal[pos];
  }
}

void compact(unsigned int* in, unsigned int* inVal, size_t count,
             unsigned int mask, bool mask_match, unsigned int offset,
             unsigned int** out_p, unsigned int** outVal_p,
             size_t* outCount)
{
  checkCudaErrors(cudaGetLastError());
  unsigned int* filter;
  unsigned int* indexes;
  unsigned int* sum;
  unsigned int* sumout;
  size_t round_count;
  round_up(count, &round_count);
  dim3 block(1024);
  dim3 grid(count/block.x+1);
  dim3 rgrid(round_count/block.x);

//  dim3 block(16);
//  dim3 grid(1);
//  dim3 rgrid(1);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMallocManaged(&filter,  count*sizeof(unsigned int)));
  checkCudaErrors(cudaMallocManaged(&indexes, round_count*sizeof(unsigned int)));
  checkCudaErrors(cudaMallocManaged(&sum, count*sizeof(unsigned int)));
  checkCudaErrors(cudaMallocManaged(&sumout, sizeof(unsigned int)));

  map_equal<<<grid, block>>>(in, mask, mask_match, filter, count);
  checkCudaErrors(cudaDeviceSynchronize());

  reduce_sum<<<grid, block, block.x*sizeof(unsigned int)>>>(filter, count, sumout);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  cudaMemcpy(indexes, filter, count*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaGetLastError());

  printf("line %u (filtred:%u/%u)\n", 216, sumout[0], count);
  scan_exclusive<<<grid, block, block.x*sizeof(unsigned int)>>>(indexes, round_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  printf("LINE %u scan_fist_block: %6u -b0- %6u ... %6u -b-1- %6u\n", 230, indexes[0], indexes[block.x-1], indexes[count-count%block.x], indexes[count-1]);

  // scan blocks
  int running_sum = 0;
  for(int i = 1; i < grid.x; i++)
  {
    running_sum += indexes[i*block.x-1]+filter[i*block.x-1];
    indexes[i*block.x] += running_sum;
  }
  printf("LINE %u scan_fist_block: %6u -b0- %6u ... %6u -b-1- %6u\n", 240, indexes[0], indexes[block.x-1], indexes[count-count%block.x], indexes[count-1]);

  // add offset per block
  map_offset<<<grid, block>>>(indexes, round_count, offset);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  printf("LINE %u scan_fist_block: %6u -b0- %6u ... %6u -b-1- %6u\n", 246, indexes[0], indexes[block.x-1], indexes[count-count%block.x], indexes[count-1]);


  unsigned int num_remaining = sumout[0];
  *outCount = num_remaining;
  checkCudaErrors(cudaMallocManaged(out_p,    num_remaining*sizeof(unsigned int)));
  checkCudaErrors(cudaMallocManaged(outVal_p, num_remaining*sizeof(unsigned int)));
  printf("LINE %u\n", 253, sumout[0], count);

  scatter_filtered<<<grid, block>>>(in, inVal, filter, indexes, count, *out_p, *outVal_p);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  printf("LINE %u\n", 258, sumout[0], count);
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
  checkCudaErrors(cudaGetLastError());
  size_t num_zero;
  size_t num_ones;
  unsigned int* out_zero;
  unsigned int* out_zero_values;
  unsigned int* out_ones;
  unsigned int* out_ones_values;


  int mask = 1;
  for(unsigned int i = UINT_MAX; i > 0; i >>= 1){
    printf("REACHED %u\n", i);
    checkCudaErrors(cudaGetLastError());
    compact(d_inputVals, d_inputPos, numElems, mask, 0, 0,        &out_zero, &out_zero_values, &num_zero);
    compact(d_inputVals, d_inputPos, numElems, mask, 1, num_zero-1, &out_ones, &out_ones_values, &num_ones);
    // merge
    cudaMemcpy(d_inputVals,              out_zero,        num_zero, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inputPos,               out_zero_values, num_zero, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(d_inputVals[num_zero]), out_ones,        num_ones, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&(d_inputPos[num_zero]),  out_ones_values, num_ones, cudaMemcpyDeviceToDevice);
    // scatter
//    scatter<<<numElems/1024, 1024>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

    swap_arrays(&d_inputVals, &d_outputVals);
    swap_arrays(&d_inputPos,  &d_outputPos);

    mask <<= 1;
  }
}
