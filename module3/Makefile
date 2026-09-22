NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17

.PHONY: all clean

all: assignment.exe

assignment.exe: assignment.cu
	$(NVCC) $(NVCCFLAGS) assignment.cu -o assignment.exe

clean:
	rm -f assignment.exe results.csv performance_comparison.png \
		results_*.csv performance_comparison_*.png
