//
// Created by HP on 11/25/2022.
//

#include <iostream>
#include "Dense.h"
#include "../Utils/linalg.h"
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
//    this->outputs = this->weights * this->inputs + this->biases;
    this->outputs = this->weights * this->inputs;
    this->outputs = this->outputs.colwise() + this->biases;
//    this->outputs = this->outputs + linalg::broadcast(this->biases, this->outputs.cols(), 1);
}
void Dense::backward(Eigen::MatrixXd gradientsIn) {
    this->weightsGradients = (gradientsIn * this->inputs.transpose())/gradientsIn.cols();
//    cout << "grad in: " << endl << gradientsIn << endl;
    this->biasesGradients = gradientsIn.rowwise().sum()/gradientsIn.cols();
//    cout << "biasesGradients: " << endl << this->biasesGradients << endl;
    this->gradients = this->weights.transpose() * gradientsIn;
}
void Dense::update(float learningRate) {
    this->weights = this->weights - learningRate * this->weightsGradients;
//    cout << "biases shape: " << this->biases.rows() << " x " << this->biases.cols() << endl;
//    cout << "biasesGradients shape: " << this->biasesGradients.rows() << " x " << this->biasesGradients.cols() << endl;
//    this->biases = linalg::broadcast(this->biases, this->biasesGradients.cols(), 1) - learningRate * this->biasesGradients;
    this->biases = this->biases - learningRate * this->biasesGradients;
}