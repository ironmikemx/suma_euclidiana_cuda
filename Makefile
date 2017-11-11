all:	recommender recommender_optimized recommender_memory_optimized

recommender:	data.o recommender.o
	nvcc data.o recommender.o -o recommender

recommender.o:	recommender.cu
	nvcc -c recommender.cu -o recommender.o

recommender_optimized:	data.o  recommender_optimized.o
	nvcc data.o recommender_optimized.o -o recommender_optimized

recommender_optimized.o:  recommender_optimized.cu
	nvcc -c recommender_optimized.cu -o recommender_optimized.o

recommender_memory_optimized:       data.o  recommender_memory_optimized.o
	nvcc data.o recommender_memory_optimized.o -o recommender_memory_optimized

recommender_memory_optimized.o:  recommender_memory_optimized.cu
	nvcc -c recommender_memory_optimized.cu -o recommender_memory_optimized.o


data.o: data.cu
	nvcc -c data.cu -o data.o

clean:
	rm -f *.o recommender recommender_optimized recommender_memory_optimized


