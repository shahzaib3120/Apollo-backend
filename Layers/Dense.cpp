//
// Created by HP on 11/25/2022.
//

#include "Dense.h"
using namespace std;
Dense::Dense(int numNeurons, int numInputs, int numOutputs):Layer(numNeurons, numInputs, numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
}
Dense::Dense(int numNeurons, int *shape): Layer(numNeurons, shape) {
    this->numNeurons = numNeurons;
    this->numInputs = shape[0];
    this->numOutputs = shape[1];
}
void Dense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->outputs = this->weights * this->inputs + this->biases;
}
void Dense::backward(Eigen::MatrixXd gradientsIn) {
    this->weightsGradients = gradientsIn * this->inputs.transpose();
    this->biasesGradients = gradientsIn;
    this->gradients = this->weights.transpose() * gradientsIn;
}
void Dense::update(float learningRate) {
    this->weights = this->weights - learningRate * this->weightsGradients;
    this->biases = this->biases - learningRate * this->biasesGradients;
}


