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
  int x = blockIdx.x +1;
  int y = blockIdx.y +1;
  int retIndex = pixelIndex(x, y, width);
  
  int cacheIndex = pixelIndex(threadIdx.x, threadIdx.y, 12);
  
  // ** individual thread logic
  int xOffset = 0;
  int yOffset = 0;
  int primaryOffset = 0;
  int secondaryOffset = 0;
  if(threadIdx.x <3){
    primaryOffset = -1;
  }
  else{
    primaryOffset = 1;
  }
  // secondary offset 
  if((threadIdx.x == 0)||(threadIdx.x == 3)){
    secondaryOffset = -1;
  }
  else if((threadIdx.x == 2)||(threadIdx.x == 5)){
    secondaryOffset = 1;
  }
  // midify the actual offsets
  if(threadIdx.y ==0){
    xOffset = primaryOffset;
    yOffset = secondaryOffset;
  }
  else{
    yOffset = primaryOffset;
    xOffset = secondaryOffset;
  }
  int pixValue = 0;
  if(threadIdx.y ==0){
    if((threadIdx.x == 0)||(threadIdx.x == 4)){
      pixValue = -1;
    }
    else if((threadIdx.x == 1)||(threadIdx.x == 5)){
      pixValue = 1;
    }
    else if(threadIdx.x == 2){
      pixValue = -2;
    }
    else if(threadIdx.x == 3){
      pixValue = 2;
    }
  }else{
    if((threadIdx.x == 0)||(threadIdx.x == 2)){
      pixValue = -1;
    }
    else if((threadIdx.x == 3)||(threadIdx.x == 5)){
      pixValue = 1;
    }
    else if(threadIdx.x == 1){
      pixValue = -2;
    }
    else if(threadIdx.x == 4){
      pixValue = 2;
    }
  }
  // store in shared mem
  cache[cacheIndex] = pixValue * pixels[x + xOffset, y + yOffset, width];

  // //*logic probably in a while loop
  // int temp = 0;

  // int x00 = -1;
  // int x20 = 1;
  // int x01 = -2;
  // int x21 = 2;
  // int x02 = -1;
  // int x22 = 1;
  // x00 *= pixels[pixelIndex(x - 1, y - 1, width)];
  // x01 *= pixels[pixelIndex(x - 1, y, width)];
  // x02 *= pixels[pixelIndex(x - 1, y + 1, width)];
  // x20 *= pixels[pixelIndex(x + 1, y - 1, width)];
  // x21 *= pixels[pixelIndex(x + 1, y, width)];
  // x22 *= pixels[pixelIndex(x + 1, y + 1, width)];

  // int y00 = -1;
  // int y10 = -2;
  // int y20 = -1;
  // int y02 = 1;
  // int y12 = 2;
  // int y22 = 1;
  // y00 *= pixels[pixelIndex(x - 1, y - 1, width)];
  // y10 *= pixels[pixelIndex(x, y - 1, width)];
  // y20 *= pixels[pixelIndex(x + 1, y - 1, width)];
  // y02 *= pixels[pixelIndex(x - 1, y + 1, width)];
  // y12 *= pixels[pixelIndex(x, y + 1, width)];
  // y22 *= pixels[pixelIndex(x + 1, y + 1, width)];

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
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);

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

  // 
  int * c;
  int * dev_c;
  c = (int *)malloc(sizeof(int) * imgWidth * imgHeight);
  cudaMalloc((void**)&dev_c, sizeof(int) * imgWidth * imgHeight);

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

  //** find dimesntion for blocks
  dim3 threadsPerBlock(2, 6); //one sorbo area
  dim3 numBlocks(imgHeight, imgWidth); //probably block per pixel
  sobel<<<numBlocks, threadsPerBlock>>>(imgWidth, dev_pixels, dev_c);
  cudaMemcpy(c, dev_c, sizeof(int) * imgWidth * imgHeight, cudaMemcpyDeviceToHost);

  
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


  // ** free all my pointers/arrays
  free(c);
  free(dev_c);
  free(dev_pixels);
  free(pixels);
  FreeImage_Unload(bitmap);
  FreeImage_Unload(image);
  return 0;
}
