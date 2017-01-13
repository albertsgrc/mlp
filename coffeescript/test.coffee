NeuralNetwork = require '.'

inputs = [
    [1, 1]
    [0, 1]
    [1, 0]
    [0, 0]
]

outputs = [
    [0]
    [1]
    [1]
    [0]
]

N_EPOCHS = 10000
LEARNING_RATE = 0.1
N_HIDDEN_LAYERS = 1
N_HIDDEN_NEURONS = 2

nn = new NeuralNetwork(inputs[0].length, outputs[0].length, N_HIDDEN_LAYERS, N_HIDDEN_NEURONS, LEARNING_RATE)

for i in [0...N_EPOCHS]
    mse = 0
    for sample, j in inputs
        mse += nn.train(sample, outputs[j])
    mse /= inputs.length

    process.stdout.write("\rMSE: #{mse}") if i%20 is 0

console.log ""

inputs.forEach((input) -> console.log "#{input}: #{nn.recall(input)[0]}")
