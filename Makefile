all:
	g++ -std=c++11 -fopenmp -pthread -DUSE_OMP ./main.cc `pkg-config --libs --cflags opencv` -o main
	g++ -std=c++11 ./test_dynamics.cc `pkg-config --libs --cflags opencv` -o test_dynamics
