# ChatGPT usage 

I used OpenAI ChatGPT to assist with this assignment. My programming experience
is primarily in python, and I have very limited experience writing C++ or CUDA
C++. I used ChatGPT as a programming and learning aid to help translate the
assignment requirements and concepts, understanding and translating the provided code in 
`register_memeory` and `shared_constant_memory`, implementing the constant/global comparisons, and adding the cpu validation and timing. 

# Starter code and changes

Repository: https://github.com/JHU-EP-Intro2GPU/EN605.617/tree/main/module5

| Course file | Used in the solution |
| --- | --- |
| shared_constant_memory/shared_memory2.cu | `dynamicReverse`: dynamic shared array, one load/thread, barrier, reversed cross-thread read |
| register_memory/register.cu | `test_gpu_register`: private `d_tmp` scalar, multiply, global output; allocation/copy workflow |
| shared_constant_memory/constant_memory2.cu | Constant symbol and `cudaMemcpyToSymbol` initialization pattern; global/constant comparison idea |
| register_memory/assignment.c | `totalThreads`, `blockSize`, ceiling block count and argument order |



