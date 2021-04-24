# 441_hw4

sobel-cpu.cu is an example from class, I did not write it. gjtalotta_prob_N1 is
sobel-cpu.cu edited to use the GPU. take1_gjtalotta_prob_N1.cu is the GPU
version done wrong. 
gitHub for this assignment: https://github.com/GregTalotta/441_hw4

#1 It works. I find the maximum number of threads I can have per block, then
calculate the number of blocks in each direction by deviding the amount of
pixels by the amount of threads per block.
