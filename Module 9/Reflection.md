# GPU-Accelerated Customer Entity Resolution

Customer data can exist across CRM, website and loyalty systems where the same person can appear with slightly different details. For Module 9, I built the matching stage of a GPU-accelerated entity-resolution system. I am also using this assignment to test options for my final project based on dataset size, noise, and matching thresholds.

First, I generated synthetic customer records with known duplicates and deliberately similar but unrelated customers. Because I have the original customer IDs I could measure accuracy. cuBLAS is used to answer 'How similar are these records?' using GPU matrix multiplication to calculate cosine similarity. Thrust is used to answer 'Which pairs should we investigate?' by removing self comparisons and repeated pairs, filtering with a threshold and ranking the likely duplicates. I chose this because cuBLAS's output can feed directly into Thrust on the GPU.

I ran five scenarios on a Google Colab Tesla T4 comparing GPU and CPU results with all five passing validation. At 256 records, the GPU took 30.512 ms versus 1.152 ms on CPU. At 2,048 records, it took 7.568 ms versus 274.389 ms, a 36.254x speedup. This suggests GPU overhead outweighs benefits at small scales although I should repeat the timing trials to fully confirm. 

Threshold tuning also showed to make an impact. With the same 1,024-record dataset, raising the threshold from 0.40 to 0.45 improved precision from 79.3% to 92.8%, but recall fell from 69.3% to 49.8%. Being more selective reduced false matches but it missed more genuine duplicates.

My biggest challenges included understanding cuBLAS's matrix layout and choosing thresholds for the noisy data. My biggest success was getting both libraries to work together and passing validation. For my final project, I think I'm going to try with more realistic customer attributes, narrow candidate pairs before comparison and evaluate the speed-accuracy tradeoff as data scales.
