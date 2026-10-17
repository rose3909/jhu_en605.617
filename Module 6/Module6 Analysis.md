# Module 6 Performance Results

## 1. Performance results

The program was executed in Google Colab on an*NVIDIA Tesla T4 GPU, which reported 3 asynchronous engines and support for concurrent kernels. It compared the same mathematical workload using one CUDA stream and two CUDA streams.

The workload consisted of asynchronous host-to-device transfers of two arrays, array addition followed by a 24-step mathematical transformation, and asynchronous device-to-host transfers. The full pipeline was measured with a CPU steady clock while individual kernels were timed using CUDA events.

Three dataset sizes (65,536; 1,048,576; and 4,194,304 elements) and three thread configurations (128, 256, and 512 threads per block) were tested. Each configuration was run twice per stream mode, resulting in 36 timed runs. All runs passed the program's validation checks.

### Average end-to-end pipeline time

For each dataset size, the table averages both trials across all three thread configurations (six measurements per stream mode). Improvement is calculated as `(one-stream time - two-stream time) / one-stream time × 100`.

| Dataset size (elements) | One stream (ms) | Two streams (ms) | Pipeline time reduction |
|---|---:|---:|---:|
| 65,536 | 0.1544 | 0.1114 | 27.9% |
| 1,048,576 | 1.3514 | 1.0598 | 21.6% |
| 4,194,304 | 4.9567 | 4.0041 | 19.2% |

The two-stream configuration had lower average end-to-end pipeline times at every tested dataset size. The measured reduction ranged from approximately 19% to 28%.

## 2. Interpretation of the results

### Multiple streams improved overall pipeline time

The two-stream mode completed the same workload faster than the one-stream mode across all tested dataset sizes and block configurations. 

### Individual kernel times stayed relatively similar

Although end-to-end execution improved, the recorded durations of individual kernels were generally similar between the one-stream and two-stream modes. 

This showed me that the reduction in overall time was associated more with execution scheduling and possible operation overlap than with a substantial decrease in the time needed for each kernel's calculations. 

### Thread configuration affected performance but less consistently

Changing the number of threads per block affected execution time but larger blocks did not always produce faster results. 

### Conclusion

On the tested Tesla T4 using two CUDA streams improved end-to-end pipeline time for this workload while individual kernel times remained broadly similar. All 36 timed runs passed validation. This experiment demonstrated why CUDA performance analysis should distinguish kernel duration from total elapsed pipeline time including data movement and synchronization.