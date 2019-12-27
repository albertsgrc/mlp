all: cli test

cli: cli.cc mlp.hh
	g++ -std=c++11 -g -Ofast -march=native cli.cc -o cli

test: test.cc mlp.hh
	g++ -std=c++11 -g -Ofast -march=native test.cc -o test

clean:
	rm -rf main
