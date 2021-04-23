/***********************************************************************
* Greg Talotta
* 441 HW 4
* modified from
      * sobel-cpu.cu
      *
      * Implements a Sobel filter on the image that is hard-coded in main.
      * You might add the image name as a command line option if you were
      * to use this more than as a one-off assignment.
      *
      * See https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
      * or https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
      * for info on how the filter is implemented.
      *
      * Compile/run with:  nvcc sobel-cpu.cu -lfreeimage
      *
 ***********************************************************************/
#include "FreeImage.h"
#include "stdio.h"
#include "math.h"

// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
__device__ int pixelIndex(int x, int y, int width)
{
  return (y * width + x);
}

// Returns the sobel value for pixel x,y
// ** convert to void with CUDA mem transfer
__global__ void sobel(int width, char *pixels, int *c)
{
  //** tricky
  __shared__ int cache[12]; 
  int x = blockIdx.x;
  int y = blockIdx.y;
  int retIndex = pixelIndex(x, y, width);
  if(x <1 || y <1 || x > blockDim.x -1 || y > blockDim.y -1){
      return;
  }
  int cacheIndex = pixelIndex(threadIdx.x, threadIdx.y, 12);
  
  // ** individual thread logic
  int xOffset = 0;
  int pixValue;
  int yOffset = 0;
  int primaryOffset = 0;
  int secondaryOffset = 0;
  int tx = threadIdx.x;
  if(threadIdx.y == 0){
    if(tx == 0){
      pixValue = -1;
      pixValue *= pixels[pixelIndex(x - 1, y - 1, width)];
    }else if(tx==1){
      pixValue = 1;
      pixValue*= pixels[pixelIndex(x - 1, y, width)];
    }else if(tx==2){
      pixValue = -2;
      pixValue*= pixels[pixelIndex(x - 1, y + 1, width)];
    }else if(tx==3){
      pixValue = 2;
      pixValue*= pixels[pixelIndex(x + 1, y - 1, width)];
    }else if(tx==4){
      pixValue = -1;
      pixValue*= pixels[pixelIndex(x + 1, y, width)];
    }else if(tx==5){
      pixValue = -1;
      pixValue *=pixels[pixelIndex(x + 1, y + 1, width)];
    }
  }
  else if (){
    if(tx == 0){
      pixValue = -1;
      pixValue*= pixels[pixelIndex(x - 1, y - 1, width)];
    }else if(tx==1){
      pixValue = -2;
      pixValue*= pixels[pixelIndex(x, y - 1, width)];
    }else if(tx==2){
      pixValue = -1;
      pixValue*= pixels[pixelIndex(x + 1, y - 1, width)];
    }else if(tx==3){
      pixValue = 1;
      pixValue*= pixels[pixelIndex(x - 1, y + 1, width)];
    }else if(tx==4){
      pixValue = 2;
      pixValue*= pixels[pixelIndex(x, y + 1, width)];
    }else if(tx==5){
      pixValue = 1;
      pixValue*= pixels[pixelIndex(x + 1, y + 1, width)];
    }
  }
  // store in shared mem
  cache[cacheIndex] = pixValue;

  // **do this in thread 0 and save sqrt in mem not return
  if((threadIdx.x == 0) && (threadIdx.y == 0)){
    __syncthreads();
    int px = 0;
    for(int i = 0; i < 6; ++i){
      px += cache[i];
    }
    int py = 0;
    for(int i = 6; i < 12; ++i){
      py += cache[i];
    }
    c[retIndex] = (int) sqrtf(px * px + py * py); // store in mem
  }
}

int main()
{
  printf("start\n");
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);
  printf("preimg load\n");
  // Load image and get the width and height
  FIBITMAP *image;
  image = FreeImage_Load(FIF_PNG, "coins.png", 0);
  if (image == NULL)
  {
    printf("Image Load Problem\n");
    exit(0);
  }
  int imgWidth;
  int imgHeight;
  imgWidth = FreeImage_GetWidth(image);
  imgHeight = FreeImage_GetHeight(image);
  printf("premaloc 1\n");
  // // 
  int * c;
  int * dev_c;
  c = (int *)malloc(sizeof(int) * imgWidth * imgHeight);
  cudaMalloc((void**)&dev_c, sizeof(int) * imgWidth * imgHeight);
  printf("preconvert\n");
  // Convert image into a flat array of chars with the value 0-255 of the
  // greyscale intensity
  RGBQUAD aPixel;
  char *pixels;
  int pixIndex = 0;
  pixels = (char *)malloc(sizeof(char) * imgWidth * imgHeight);
  for (int i = 0; i < imgHeight; i++)
    for (int j = 0; j < imgWidth; j++)
    {
      FreeImage_GetPixelColor(image, j, i, &aPixel);
      char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue) / 3);
      pixels[pixIndex++] = grey;
    }
  char *dev_pixels;
  cudaMalloc((void**)&dev_pixels, sizeof(char) * imgWidth * imgHeight);
  cudaMemcpy(dev_pixels, pixels,sizeof(char) * imgWidth * imgHeight, cudaMemcpyHostToDevice);
  printf("precopy\n");

  //** find dimesntion for blocks
  dim3 threadsPerBlock(2, 6); //one sorbo area
  dim3 numBlocks(imgWidth, imgHeight); //probably block per pixel
  sobel<<<numBlocks, threadsPerBlock>>>(imgWidth, dev_pixels, dev_c);
  cudaMemcpy(c, dev_c, sizeof(int) * imgWidth * imgHeight, cudaMemcpyDeviceToHost);
  printf("postcopy\n");
  
  // Apply sobel operator to pixels, ignoring the borders
  // ** chnage this nested for loop to kernal calls
  FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
  for (int i = 1; i < imgWidth - 1; i++)
  {
    for (int j = 1; j < imgHeight - 1; j++)
    {
      int sVal = c[j*imgWidth + i];
      aPixel.rgbRed = sVal;
      aPixel.rgbGreen = sVal;
      aPixel.rgbBlue = sVal;
      FreeImage_SetPixelColor(bitmap, i, j, &aPixel);
    }
  }
  FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);
  printf("postbit\n");

  // ** free all my pointers/arrays
  free(c);
  cudaFree(dev_c);
  cudaFree(dev_pixels);
  free(pixels);
  FreeImage_Unload(bitmap);
  FreeImage_Unload(image);
  return 0;
}
