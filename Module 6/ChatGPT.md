# ChatGPT usage

I used OpenAI ChatGPT to assist with this assignment. My programming experience is primarily in Python, and I have limited experience writing C++ or CUDA C++. I used ChatGPT as a programming and learning aid to interpret the assignment requirements, understand the instructor's CUDA stream example, extend the kernel, and implement the stream comparison, timing, and CPU validation. I reviewed the resulting code and used the Colab test results to assess its correctness and performance.

# Starter code and changes

Course repository: https://github.com/JHU-EP-Intro2GPU/EN605.617/tree/main/module6

| Course file | Used in the solution |
| --- | --- |
| `stream_example.cu` | Starting pattern for the two-input array-addition kernel, `cudaHostAlloc` pinned memory, `cudaMemcpyAsync` transfers, `cudaStreamCreate`, `cudaStreamSynchronize`, and CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`). |
| `multiGPU.cu` | Reviewed for context on device selection and GPU work, but its multi-GPU/CPU-thread approach was not reused. |
| `multi_gpu_example.cu` | Reviewed for its GPU workload and launch concepts, but its multi-GPU design and dependent utility headers were not used. |

# Extensions and testing

I adapted the single-stream array-addition example into an add-and-transform kernel with 24 iterations of floating point calculations. The same kernel is tested in one-stream and two-stream modes, with two executions per combination of dataset size (65,536; 1,048,576; 4,194,304 elements) and block size (128, 256, 512 threads). CUDA events measure individual kernel durations and a host steady clock measures the full pipeline including transfers and synchronization. Sampled CPU reference checks validate the outputs. The program reports measured values rather than assuming that two streams are faster.

These files were developed with ChatGPT assistance with benchmark results obtained through working it through Google Colab. 