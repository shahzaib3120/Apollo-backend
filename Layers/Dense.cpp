//
// Created by HP on 11/25/2022.
//

#include "Dense.h"
#include <iostream>
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
void Dense::backward(Eigen::MatrixXd gradients) {
    this->weightsGradients = gradients * this->inputs.transpose();
    this->biasesGradients = gradients;
    this->gradients = this->weights.transpose() * gradients;
}
void Dense::update(float learningRate) {
//    std::cout << "Updating weights and biases" << std::endl;
//    std::cout << "Biases before update: " << this->biases << std::endl;
//    system("pause");
    this->weights = this->weights - learningRate * this->weightsGradients;
    this->biases = this->biases - learningRate * this->biasesGradients;
//    std::cout << "weights: " << this->weights << std::endl;
//    std::cout << "Biases after Update: " << this->biases << std::endl;
//    system("pause");
}


