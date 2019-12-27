#include "mlp.hh"
#include <fstream>
#include <iostream>

int main() {
    int n_inputs = 2;
    int n_hidden_layers = 1;
    int n_hidden_neurons = 2;
    int n_outputs = 1;
    double learning_rate = 0.01;

    // Instantiate
    MLP nn(n_inputs, n_hidden_layers, n_hidden_neurons, n_outputs, learning_rate);

    // Modify learning rate
    nn.set_learning_rate(0.1);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> samples = {
        {{0,0}, {0}},
        {{0,1}, {1}},
        {{1,0}, {1}},
        {{1,1}, {0}}
    };

    for (int i = 0; i < 10000; ++i) {
        double mse = 0.0;
        // Train and get mean squared error
        mse += nn.train(&samples[0].first[0], &samples[0].second[0]);
        mse += nn.train(&samples[1].first[0], &samples[1].second[0]);
        mse += nn.train(&samples[2].first[0], &samples[2].second[0]);
        mse += nn.train(&samples[3].first[0], &samples[3].second[0]);
    }

    // Recall
    std::cout << nn.recall(&samples[0].first[0])[0] << std::endl
              << nn.recall(&samples[1].first[0])[0] << std::endl
              << nn.recall(&samples[2].first[0])[0] << std::endl
              << nn.recall(&samples[3].first[0])[0] << std::endl;  

    // Writing to file
    std::ofstream file_out;
    file_out.open("./nn.txt");
    file_out << nn;
    file_out.close();

    // Reading from file
    std::ifstream file_in;
    file_in.open("./nn.txt");
    file_in >> nn;
    file_in.close();
}