/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>
#include <functional>


__global__
void __reduce_max(const float* const din, float* gdata, int count)
{
  extern __shared__ float sdata[];

  int mid = threadIdx.x + blockIdx.x*blockDim.x;
  int tid = threadIdx.x;

  if (mid < count)
    sdata[tid] = din[mid];
  __syncthreads();
//  return;

  for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if(tid < s && mid+s < count)
    {
      sdata[tid] = sdata[tid+s] > sdata[tid] ? sdata[tid+s] : sdata[tid];
    }
    __syncthreads();
  }

  if (tid == 0)
    gdata[blockIdx.x] = sdata[tid];
}


float reduce_max(const float* const d_logLuminance,
            const size_t numRows,
            const size_t numCols
            )
{
  const int blockSize = 1024;
  const int gridSize  = (numCols*numRows)/blockSize+1;
  int size = numCols*numRows*sizeof(float);

  float* gdata;
  float* din;
  cudaMallocManaged(&gdata, gridSize*sizeof(float));
  checkCudaErrors(cudaMallocManaged(&din, size));
  checkCudaErrors(cudaMemcpyAsync(din, d_logLuminance, size, cudaMemcpyDeviceToDevice));

  __reduce_max<<<gridSize, blockSize, blockSize*sizeof(float)>>>(din, gdata, numRows*numCols);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  cudaMemPrefetchAsync(gdata, gridSize*sizeof(float), cudaCpuDeviceId);
  float max = gdata[0];
  for(int i = 0; i < gridSize; i++)
    max = max > gdata[i] ? max : gdata[i];

  cudaFree(gdata);
  cudaFree(din);
  return max;
}


__global__
void __reduce_min(const float* const din, float* gdata, int count)
{
  extern __shared__ float sdata[];

  int mid = threadIdx.x + blockIdx.x*blockDim.x;
  int tid = threadIdx.x;

  if (mid < count)
    sdata[tid] = din[mid];
  __syncthreads();
//  return;

  for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if(tid < s && mid+s < count)
    {
      sdata[tid] = sdata[tid+s] < sdata[tid] ? sdata[tid+s] : sdata[tid];
    }
    __syncthreads();
  }

  if (tid == 0)
    gdata[blockIdx.x] = sdata[tid];
}


float reduce_min(const float* const d_logLuminance,
            const size_t numRows,
            const size_t numCols
            )
{
  const int blockSize = 1024;
  const int gridSize  = (numCols*numRows)/blockSize+1;
  int size = numCols*numRows*sizeof(float);

  float* gdata;
  float* din;
  cudaMallocManaged(&gdata, gridSize*sizeof(float));
  checkCudaErrors(cudaMallocManaged(&din, size));
  checkCudaErrors(cudaMemcpyAsync(din, d_logLuminance, size, cudaMemcpyDeviceToDevice));

  __reduce_min<<<gridSize, blockSize, blockSize*sizeof(float)>>>(din, gdata, numRows*numCols);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  cudaMemPrefetchAsync(gdata, gridSize*sizeof(float), cudaCpuDeviceId);
  float max = gdata[0];
  for(int i = 0; i < gridSize; i++)
    max = max < gdata[i] ? max : gdata[i];

  cudaFree(gdata);
  cudaFree(din);
  return max;
}

__global__
void histogram(const float* const din, unsigned int* const cdf, int count, float min, float max, const size_t numBins)
{
  int ps = threadIdx.x + blockDim.x*blockIdx.x;


  if (ps > count)
    return;

  int index = (int)(din[ps]-min)/(max-min)*numBins;
//  if (index != 0)
//    printf("%3d %f\n", index, din[ps]);
  if (index>numBins)
    printf("%f min:%f max:%f\n", din[ps], min, max);
  atomicAdd(&cdf[index], 1);

}

__global__
void prefix_scan(unsigned int* const cdf)
{

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  max_logLum = reduce_max(d_logLuminance, numRows, numCols);
  min_logLum = reduce_min(d_logLuminance, numRows, numCols);

  printf(" MAX: %f\n", max_logLum);
  printf(" MIN: %f\n", min_logLum);
  printf(" NB:  %d\n", (int)numBins);
  const int blockSize = 1024;
  const int gridSize = numCols*numRows/blockSize+1;

  histogram<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, numCols*numRows, min_logLum, max_logLum, numBins);
  checkCudaErrors(cudaDeviceSynchronize());

  unsigned int* h_cdf;
  cudaMallocManaged(&h_cdf, numBins*sizeof(unsigned int));
  cudaMemcpyAsync(h_cdf, d_cdf, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for(unsigned i = 0; i < numBins; i++){
    if (i==0 || h_cdf[i] != 0)
      printf("%03d %d\n", i, h_cdf[i]);
  }

  int x = 0;
  for(int i = 0; i < numBins; i ++)
  {
    int tmp = h_cdf[i];
    h_cdf[i] = x;
    x += tmp;
  }

  for(unsigned i = 0; i < numBins; i++){
    if (i==0 || h_cdf[i] != 0)
    {
//      printf("%03d %d\n", i, h_cdf[i]);
    }
  }

  cudaMemcpyAsync(d_cdf, h_cdf, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


}
