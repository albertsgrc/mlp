//
// Created by Albert Segarra Roca on 16/12/16.
//

// 2 2 2 1 4 10000 0.3 1 1 0 0 1 1 1 0 1 0 0 0

#ifndef SRC_MLP_HH
#define SRC_MLP_HH

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

typedef unsigned int uint;

const float DFL_MIN_WEIGHT_INIT = -0.3;
const float DFL_MAX_WEIGHT_INIT = +0.3;

#define hiddenat(layer, neuron) hidden[(layer)*n_hidden_neurons + (neuron)]
#define inputweightat(neuron_to, neuron_from) weights[(neuron_to)*n_inputs + (neuron_from)]
#define hiddenweightat(layer_to, neuron_to, neuron_from) weights[n_inputs*n_hidden_neurons + ((layer_to) - 1)*n_hidden_neurons*n_hidden_neurons + (neuron_to)*n_hidden_neurons + (neuron_from)]
#define outputweightat(neuron_to, neuron_from) weights[n_hidden_neurons*(n_inputs + (n_hidden_layers - 1)*n_hidden_neurons) + n_hidden_neurons*(neuron_to) + (neuron_from)]
#define hiddenbiasat(layer, neuron) bias[(layer)*n_hidden_neurons + (neuron)]
#define outputbiasat(neuron) bias[n_hidden_layers*n_hidden_neurons + (neuron)]
#define hiddendeltaat(layer, neuron) delta_hidden[(layer)*n_hidden_neurons + (neuron)]

void error(const std::string& s) {
    perror(s.c_str());
    exit(1);
}

class MLP {
private:
    void reserve() {
        n_weights = n_hidden_neurons*(n_inputs + (n_hidden_layers - 1)*n_hidden_neurons + n_outputs);
        n_bias = n_hidden_neurons*n_hidden_layers + n_outputs;

        weights = (double*) malloc(sizeof(double)*n_weights);
        bias = (double*) malloc(sizeof(double)*n_bias);

        hidden = (double*) malloc(sizeof(double)*n_hidden_layers*n_hidden_neurons);
        output = (double*) malloc(sizeof(double)*n_outputs);
        delta_output = (double*) malloc(sizeof(double)*n_outputs);
        delta_hidden = (double*) malloc(sizeof(double)*n_hidden_layers*n_hidden_neurons);

        reserved = true;
    }

    void free_mem() {
        if (reserved) {
            free(weights);
            free(bias);
            free(hidden);
            free(output);
            free(delta_output);
            free(delta_hidden);
            reserved = false;
        }
    }

    static inline double sigmoid(double x) {
        return (1.0/(1.0 + exp(-x)));
    }

    static inline double derivative_sigmoid(double x) {
        return x*(1.0 - x);
    }

    double backpropagate(double* given, double* expected) {
        double mse = 0;

        // Delta for outputs
        for (uint neuron_output = 0; neuron_output < n_outputs; ++neuron_output) {
            delta_output[neuron_output] = expected[neuron_output] - given[neuron_output];
            mse += delta_output[neuron_output]*delta_output[neuron_output];
        }

        // Delta for last layer
        uint last_hidden_layer = n_hidden_layers - 1;
        for (uint hidden_to = 0; hidden_to < n_hidden_neurons; ++hidden_to) {
            double delta = 0;

            for (uint output_from = 0; output_from < n_outputs; ++output_from)
                delta += delta_output[output_from]*outputweightat(output_from, hidden_to);

            hiddendeltaat(last_hidden_layer, hidden_to) = derivative_sigmoid(hiddenat(last_hidden_layer, hidden_to))*delta;
        }

        // Delta for layer layer_to+1 to layer_to
        for (int layer_to = n_hidden_layers - 2; layer_to >= 0; layer_to--) {
            for (uint neuron_to = 0; neuron_to < n_hidden_neurons; ++neuron_to) {
                double delta = 0;

                for (uint neuron_from = 0; neuron_from < n_hidden_neurons; ++neuron_from) {
                    delta += hiddendeltaat(layer_to+1, neuron_from)*
                             hiddenweightat(layer_to+1, neuron_from, neuron_to);
                }

                hiddendeltaat(layer_to, neuron_to) = derivative_sigmoid(hiddenat(layer_to, neuron_to))*delta;
            }
        }

        // Weight update inputs
        for (uint hidden_to = 0; hidden_to < n_hidden_neurons; ++hidden_to) {
            hiddenbiasat(0, hidden_to) += learning_rate*hiddendeltaat(0, hidden_to);
            for (uint input_from = 0; input_from < n_inputs; ++input_from)
                inputweightat(hidden_to, input_from) += learning_rate * hiddendeltaat(0, hidden_to) * input[input_from];
        }

        // Weight update for hidden layers
        for (uint layer_to = 1; layer_to < n_hidden_layers; ++layer_to) {
            for (uint neuron_to = 0; neuron_to < n_hidden_neurons; ++neuron_to) {
                hiddenbiasat(layer_to, neuron_to) += learning_rate*hiddendeltaat(layer_to, neuron_to);
                for (uint neuron_previous = 0; neuron_previous < n_hidden_neurons; ++neuron_previous)
                    hiddenweightat(layer_to, neuron_to, neuron_previous) +=
                            learning_rate*hiddendeltaat(layer_to, neuron_to)*hiddenat(layer_to - 1, neuron_previous);
            }
        }

        // Weight update for outputs
        for (uint neuron_output = 0; neuron_output < n_outputs; ++neuron_output) {
            outputbiasat(neuron_output) += learning_rate*delta_output[neuron_output];
            for (uint neuron_hidden = 0; neuron_hidden < n_hidden_neurons; ++neuron_hidden)
                outputweightat(neuron_output, neuron_hidden) +=
                        learning_rate*delta_output[neuron_output]*hiddenat(n_hidden_layers - 1, neuron_hidden);
        }

        return mse/n_outputs;
    }
public:
    bool reserved;

    double* input;
    double* hidden; // Matrix
    double* output;

    double* weights;
    double* bias;

    double* delta_output;
    double* delta_hidden;

    uint n_inputs;
    uint n_hidden_layers;
    uint n_hidden_neurons;
    uint n_outputs;
    uint n_weights;
    uint n_bias;

    double learning_rate;

    MLP() : reserved(false) {}

    MLP(uint n_inputs, uint n_hidden_layers, uint n_hidden_neurons, uint n_outputs, double learning_rate,
                   double min_weight_init = DFL_MIN_WEIGHT_INIT, double max_weight_init = DFL_MAX_WEIGHT_INIT)
            : n_inputs(n_inputs), n_hidden_layers(n_hidden_layers),
              n_hidden_neurons(n_hidden_neurons), n_outputs(n_outputs), learning_rate(learning_rate) {

        reserve();

        std::mt19937_64 generator(time(0));

        std::normal_distribution<double> distribution_inputs(0, sqrt(2.0/n_inputs));
        std::normal_distribution<double> distribution_hidden(0, sqrt(2.0/n_hidden_neurons));

        uint i;
        for (i = 0; i < n_inputs*n_hidden_neurons; ++i)
            weights[i] = distribution_inputs(generator);
        for (; i < n_weights; ++i)
            weights[i] = distribution_hidden(generator);

        memset(bias, 0, n_bias*sizeof(double));
    }

    ~MLP() {
        free_mem();
    }


    inline void set_learning_rate(double v) { learning_rate = v; }

    double* recall(double* input_values) {
        input = input_values;

        // Input to 1st hidden layer
        for (uint hidden_to = 0; hidden_to < n_hidden_neurons; ++hidden_to) {
            double sum = hiddenbiasat(0, hidden_to);
            for (uint input_from = 0; input_from < n_inputs; ++input_from)
                sum += input[input_from]*inputweightat(hidden_to, input_from);
            hiddenat(0, hidden_to) = sigmoid(sum);
        }

        // layer - 1 to layer
        for (uint layer_to = 1; layer_to < n_hidden_layers; ++layer_to) {
            for (uint neuron_to = 0; neuron_to < n_hidden_neurons; ++neuron_to) {
                double sum = hiddenbiasat(layer_to, neuron_to);
                for (uint neuron_from = 0; neuron_from < n_hidden_neurons; ++neuron_from)
                    sum += hiddenat(layer_to - 1, neuron_from)*hiddenweightat(layer_to, neuron_to, neuron_from);
                hiddenat(layer_to, neuron_to) = sigmoid(sum);
            }
        }

        // Last layer to output
        uint last_layer = n_hidden_layers - 1;
        for (uint output_to = 0; output_to < n_outputs; ++output_to) {
            double sum = outputbiasat(output_to);
            for (uint hidden_from = 0; hidden_from < n_hidden_neurons; ++hidden_from)
                sum += hiddenat(last_layer, hidden_from) * outputweightat(output_to, hidden_from);
            output[output_to] = sum;
        }

        return output;
    }


    double train(double* input_values, double* expected) {
        return backpropagate(recall(input_values), expected);
    }

    void pretty_print(std::ostream& o) {
        o << "Inputs: ";
        for (uint i = 0; i < n_inputs; ++i) o << input[i] << ' ';
        o << std::endl << std::endl;
        for (uint i = 0; i < n_hidden_layers; ++i) {
            o << "Hidden layer " << i << ": ";
            for (uint j = 0; j < n_hidden_neurons; ++j) o << hiddenat(i, j) << ' ';
            o << std::endl;
        }
        o << std::endl << "Outputs: ";
        for (uint i = 0; i < n_outputs; ++i) o << output[i] << ' ';
        o << std::endl;

    }

    friend std::istream& operator>>(std::istream& s, MLP& mlp) {
        mlp.free_mem();

        s >> mlp.n_inputs >> mlp.n_hidden_layers >> mlp.n_hidden_neurons >> mlp.n_outputs;

        mlp.reserve();

        for (uint i = 0; i < mlp.n_bias; ++i)    s >> mlp.bias[i];
        for (uint i = 0; i < mlp.n_weights; ++i) s >> mlp.weights[i];

        return s;
    }

    friend std::ostream& operator<<(std::ostream& o, const MLP& mlp) {
        o << mlp.n_inputs << ' ' << mlp.n_hidden_layers << ' ' << mlp.n_hidden_neurons << ' ' << mlp.n_outputs << std::endl << std::endl;

        for (uint i = 0; i < mlp.n_bias; ++i)    o << mlp.bias[i] << ' ';
        for (uint i = 0; i < mlp.n_weights; ++i) o << mlp.weights[i] << ' ';

        return o;
    }
};


#endif //SRC_MLP_HH
