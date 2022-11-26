//
// Created by HP on 11/25/2022.
//

#include "Dense.h"
Dense::Dense(int numNeurons, int numInputs, int numOutputs):Layer(numNeurons, numInputs, numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
}
void Dense::forward() {
    this->outputs = this->weights * this->inputs + this->biases;
}
void Dense::backward() {
    this->gradients = this->weights.transpose() * this->gradients;
}
void Dense::update() {
    this->weights = this->weights - this->weightsGradients;
    this->biases = this->biases - this->biasesGradients;
}
void Dense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->forward();
}
void Dense::backward(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
    this->backward();
}

