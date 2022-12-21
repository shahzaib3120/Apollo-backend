//
// Created by HP on 11/25/2022.
//

#include <iostream>
#include "Dense.h"
#include <iomanip>
#include "../Utils/linalg.h"
using namespace std;
Apollo::Dense::Dense(int numNeurons, int numInputs, int numOutputs):Layer(numNeurons, numInputs, numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
}
Apollo::Dense::Dense(int numNeurons, int *shape): Layer(numNeurons, shape) {
    this->numNeurons = numNeurons;
    this->numInputs = shape[0];
    this->numOutputs = shape[1];
}
Apollo::Dense::Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs): Layer(weights, biases, numOutputs) {
    this->weights = weights;
    this->biases = biases;
    this->numNeurons = weights.rows();
    this->numInputs = weights.cols();
    this->numOutputs = numOutputs;
}

void Apollo::Dense::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->outputs = this->weights * this->inputs;
    this->outputs = this->outputs.colwise() + this->biases;
}
void Apollo::Dense::backward(Eigen::MatrixXd gradientsIn) {
    this->weightsGradients = (gradientsIn * this->inputs.transpose())/gradientsIn.cols();
    this->biasesGradients = gradientsIn.rowwise().sum()/gradientsIn.cols();
    this->gradients = this->weights.transpose() * gradientsIn;
}
void Apollo::Dense::update(float learningRate) {
    this->weights = this->weights - learningRate * this->weightsGradients;
    this->biases = this->biases - learningRate * this->biasesGradients;
}
void Apollo::Dense::summary() {
    cout << left << setw(15) << "Dense Layer" << setw(20) << "| Neurons: " << setw(10) << this->numNeurons << setw(20) << "| Inputs: " << setw(10) << this->numInputs << setw(20) << "| Outputs: " << setw(10) <<this->numOutputs << endl;

}

int Apollo::Dense::getTrainableParams() {
    return this->numNeurons * (this->numInputs + 1);
}