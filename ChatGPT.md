# AI Assistance Disclosure

I used OpenAI ChatGPT to assist with this assignment. My programming experience
is primarily in python, and I have very limited experience writing C++ or CUDA
C++. I used ChatGPT as a programming and learning aid to help translate the
assignment requirements and concepts, especially in the `assignment.cu` from github into CUDA C++ code.

# Where I Used AI

I provided ChatGPT with the `assignment.cu` starter file supplied by the course. ChatGPT assisted with:

- extending the professor-provided command-line and thread-configuration code
- writing equivalent CPU and CUDA implementations of the algorithms
- creating the branchless and conditional-branching CUDA kernels
- implementing a grid-stride loop so the requested CUDA threads could process
  a dataset containing millions of values
- allocating GPU memory and copying data between host and device memory
- measuring CPU execution time with C++ timing utilities
- measuring GPU kernel execution time with CUDA events and synchronization
- adding CUDA error checking and CPU/GPU result validation
- explaining unfamiliar C++ and CUDA syntax so I could run and interpret the
  program in Google Colab.