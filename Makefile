all: main

main: main.cc neural-network.hh
	g++ -std=c++11 -g -Ofast -march=native main.cc -o main
