CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
EXECSEQ = k_meansSeq
#THREADS = 40
CP_CLUSTERS = 32
CFLAGS = -O3 -fopenmp

.DEFAULT_GOAL = K_means

K_means: $(SRC)k_means.c $(BIN)utils.o k_meansSeq
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

k_meansSeq: $(SRC)k_meansSeq.c $(BIN)utils.o
	$(CC) -O3 $(SRC)k_meansSeq.c $(BIN)utils.o -o $(BIN)$(EXECSEQ)

clean:
	rm -r bin/*
#	rm -r bin/k_means
#	rm -r bin/utils.o
runseq:
	./$(BIN)$(EXECSEQ) 2120490 $(CP_CLUSTERS)
runpar:
	./$(BIN)$(EXEC) 2120490 $(CP_CLUSTERS) $(THREADS)
