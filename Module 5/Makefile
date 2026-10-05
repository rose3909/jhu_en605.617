NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo -Xptxas=-v

.PHONY: all clean
all: assignment.exe

assignment.exe: assignment.cu
	$(NVCC) $(NVCCFLAGS) assignment.cu -o assignment.exe

clean:
	rm -f assignment.exe
