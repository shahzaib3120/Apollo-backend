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
void Dense::backward(Eigen::MatrixXd gradients) {
    this->weightsGradients = gradients * this->inputs.transpose();
    this->biasesGradients = gradients;
    this->gradients = this->weights.transpose() * gradients;
//    cout << "Dense backward" << endl;
//    cout << "weightsGradients: " << endl << this->weightsGradients << endl;
//    cout << "biasesGradients: " << endl << this->biasesGradients << endl;
//    cout << "gradientsOut: " << endl << this->gradientsOut << endl;

}
void Dense::update(float learningRate) {
    this->weights = this->weights - learningRate * this->weightsGradients;
    this->biases = this->biases - learningRate * this->biasesGradients;
}


