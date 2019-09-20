#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int px = threadIdx.x + blockIdx.x*blockDim.x;
  int py = threadIdx.y + blockIdx.y*blockDim.y;
  int pos = py+px*numCols;

  if (px < numRows && py < numCols)
  {
    greyImage[pos] = (int)(.299f * rgbaImage[pos].x + .587f * rgbaImage[pos].y + .114f * rgbaImage[pos].z);
  }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 blockSize(16, 16, 1);  //TODO
  const dim3 gridSize( numRows/blockSize.x+1, numCols/blockSize.y+1, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

}
