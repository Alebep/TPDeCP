CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means

CFLAGS = -O3 -O -funroll-loops -march=native -mfpmath=sse

.DEFAULT_GOAL = OptimizedKmeans

OptimizedKmeans: $(SRC)OptimizedKmeans.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)OptimizedKmeans.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC)
