CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
EXECSEQ = k_meansSeq
#THREADS = 
CFLAGS = -O3 -fopenmp

.DEFAULT_GOAL = K_means

K_means: $(SRC)K_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)K_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/k_means
	rm -r bin/utils.o
runseq:
	./$(BIN)$(EXECSEQ) 10000000 $(CP_CLUSTERS)
runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREAD)
