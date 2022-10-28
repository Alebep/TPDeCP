CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means

CFLAGS = 

.DEFAULT_GOAL = Tp1kmeans

Tp1kmeans: $(SRC)Tp1kmeans.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)Tp1kmeans.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC)
