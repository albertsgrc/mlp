#include <iostream>
#include "neural-network.hh"
using namespace std;

template <typename T>
void prompt(const string& s, T& x) {
    cout << s << "> ";
    cin >> x;
    cout << endl;
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

    cout << "Enter " << n_samples << " " << n_inputs << "-dimensional samples and "
         << "their corresponding outputs" << endl;

    Neural_Network nn(n_inputs, n_hidden_layers, n_hidden_neurons, n_outputs, learning_rate);

    vector<vector<double>> samples(n_samples, vector<double>(n_inputs));
    vector<vector<double>> expected_output(n_samples, vector<double>(n_outputs));

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_inputs; ++j) cin >> samples[i][j];
        for (int j = 0; j < n_outputs; ++j) cin >> expected_output[i][j];
    }

    cout << "Training..." << endl;

    for (int i = 0; i < n_epochs; ++i) {
        double mse = 0;
        for (int j = 0; j < n_samples; ++j)
            mse += nn.train(&samples[j][0], &expected_output[j][0]);
        cout << "\rMSE: " << mse/n_samples;
        cout.flush();
    }

    cout << endl << "Enter the testing samples (end with 'end'):" << endl;

    double x;
    while (cin >> x) {
        vector<double> input(n_inputs);
        input[0] = x;
        for (int i = 1; i < n_inputs; ++i) cin >> input[i];

        double* output = nn.recall(&input[0]);

        for (int i = 0; i < n_outputs; ++i) cout << output[i] << ' ';
        cout << endl;
    }
}
