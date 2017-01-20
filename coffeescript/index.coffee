assert = require 'assert'

weightInit = -> Math.random()*0.6 - 0.3;

sigmoid = (x) -> 1/(1 + Math.exp(-x))
dersigmoid = (x) -> x*(1.0 - x)

class Layer
    constructor: (@nNeurons, @previousLayer, @activation=((x)->x), @activationDer=((x)->x)) ->
        @neurons = [0...@nNeurons]
        @delta = [0...@nNeurons]

        if @previousLayer?
            @previousLayer.setNextLayer(this)

    setNextLayer: (@nextLayer) ->
        @outputBiases = [0...@nextLayer.nNeurons].map(weightInit)
        @outputWeights = [0...@nextLayer.nNeurons].map(=>[0...@nNeurons].map(weightInit))

    feedForward: ->
        if @nextLayer?
            for neuronTo, j in @nextLayer.neurons
                sum = @outputBiases[j]
                for neuronFrom, i in @neurons
                    sum += @outputWeights[j][i]*neuronFrom
                @nextLayer.neurons[j] = sum

            @nextLayer.neurons = @nextLayer.neurons.map(@nextLayer.activation)
            @nextLayer.feedForward()

    backPropagate: (lr) ->
        if @previousLayer?
            for neuronFrom, i in @previousLayer.neurons
                @previousLayer.delta[i] = 0
                for neuronTo, j in @neurons
                    @previousLayer.delta[i] += @delta[j]*@previousLayer.outputWeights[j][i]
                @previousLayer.delta[i] *= @previousLayer.activationDer(neuronFrom)

            @previousLayer.backPropagate(lr)

            for neuronTo, i in @neurons
                @previousLayer.outputBiases[i] += lr*@delta[i]
                for neuronFrom, j in @previousLayer.neurons
                    @previousLayer.outputWeights[i][j] += lr*@delta[i]*neuronFrom

module.exports = class NeuralNetwork
    constructor: (@nInputs, @nOutputs, @nHiddenLayers, @nHiddenNeurons, @learningRate) ->
        @layers = [@inputLayer = new Layer(@nInputs)]

        for i in [1..@nHiddenLayers]
            @layers.push(new Layer(@nHiddenNeurons, @layers[i-1], sigmoid, dersigmoid))

        @layers.push(@outputLayer = new Layer(@nOutputs, @layers[-1..-1][0]))

    recall: (input) ->
        @inputLayer.neurons[i] = input[i] for i in [0...@nInputs]
        @inputLayer.feedForward()
        @outputLayer.neurons

    train: (input, expected) ->
        output = @recall(input)

        mse = 0
        for neuron, i in output
            delta = @outputLayer.delta[i] = expected[i] - output[i]
            mse += delta*delta
        mse /= @nOutputs

        @outputLayer.backPropagate(@learningRate)

        mse
