#include <iostream>
#include "mlp.hh"
using namespace std;

template <typename T>
void prompt(const string& s, T& x) {
    std::cout << s << "> ";
    std::cin >> x;
    std::cout << std::endl;
}

int main() {
    uint n_inputs, n_hidden_neurons, n_hidden_layers, n_outputs, n_samples, n_epochs;
    double learning_rate;

    prompt("# of inputs", n_inputs);
    prompt("# of hidden neurons", n_hidden_neurons);
    prompt("# of hidden layers", n_hidden_layers);
    prompt("# of outputs", n_outputs);
    prompt("# of samples", n_samples);
    prompt("# of epochs", n_epochs);
    prompt("Learning rate", learning_rate);

    MLP nn(n_inputs, n_hidden_layers, n_hidden_neurons, n_outputs, learning_rate);

    std::vector<std::vector<double>> samples(n_samples, std::vector<double>(n_inputs));
    std::vector<std::vector<double>> expected_output(n_samples, std::vector<double>(n_outputs));

    for (int i = 0; i < n_samples; ++i) {
        std::cout << "Training sample #" << i + 1 << ":" << std::endl;
        std::cout << "Input " << n_inputs << " numbers for the input features, space-separated:" << std::endl;
        for (int j = 0; j < n_inputs; ++j) std::cin >> samples[i][j];
        std::cout << "Input " << n_outputs << " numbers for the expected output, space-separated:" << std::endl;
        for (int j = 0; j < n_outputs; ++j) std::cin >> expected_output[i][j];

        std::cout << std::endl << std::endl;
    }

    std::cout << "Training..." << std::endl;

    for (int i = 0; i < n_epochs; ++i) {
        double mse = 0;
        for (int j = 0; j < n_samples; ++j)
            mse += nn.train(&samples[j][0], &expected_output[j][0]);
        std::cout << "\rMSE: " << mse/n_samples;
        std::cout.flush();
    }

    std::cout << std::endl 
         << "Enter the testing samples: " << n_inputs << 
         "-dimensional vectors separated by spaces." << std::endl;

    double x;
    while (std::cin >> x) {
        std::vector<double> input(n_inputs);
        input[0] = x;
        for (int i = 1; i < n_inputs; ++i) std::cin >> input[i];

        double* output = nn.recall(&input[0]);

        for (int i = 0; i < n_outputs; ++i) std::cout << output[i] << ' ';
        std::cout << std::endl;
    }
}
