//
// Created by HP on 11/25/2022.
//

#include <iostream>
#include "Dense.h"
#include <iomanip>
#include "../Utils/linalg.h"
using namespace std;
namespace Apollo{
    Dense::Dense(std::string name, int numNeurons, int numInputs, int numOutputs): Layer(name) {
        this->numNeurons = numNeurons;
        this->numInputs = numInputs;
        this->numOutputs = numOutputs;
        this->weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
        this->biases = Eigen::VectorXd::Zero(numNeurons);
        this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
        this->outputs = Eigen::MatrixXd::Random(numNeurons, numOutputs);
        this->gradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
        this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, numInputs);
        this->biasesGradients = Eigen::VectorXd::Random(numNeurons);
        this->weightsVelocity = Eigen::MatrixXd::Zero(numNeurons, numInputs);
        this->biasesVelocity = Eigen::VectorXd::Zero(numNeurons);

    }
    Dense::Dense(std::string name, int numNeurons, int *shape): Layer(name) {
        this->numNeurons = numNeurons;
        this->numInputs = shape[0];
        this->numOutputs = shape[1];
        this->weights = Eigen::MatrixXd::Random(numNeurons, shape[0]);
        this->biases = Eigen::VectorXd::Zero(numNeurons);
        this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
        this->outputs = Eigen::MatrixXd::Random(numNeurons, shape[1]);
        this->gradients = Eigen::MatrixXd::Random(numNeurons, shape[1]);
        this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, shape[0]);
        this->biasesGradients = Eigen::VectorXd::Random(numNeurons);
        this->weightsVelocity = Eigen::MatrixXd::Zero(numNeurons, numInputs);
        this->biasesVelocity = Eigen::VectorXd::Zero(numNeurons);
    }
    Dense::Dense(std::string name, Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs): Layer(name) {
        this->weights = weights;
        this->biases = biases;
        this->numNeurons = weights.rows();
        this->numInputs = weights.cols();
        this->numOutputs = numOutputs;
        this->inputs = Eigen::MatrixXd::Random(weights.cols(), numOutputs);
        this->outputs = Eigen::MatrixXd::Random(weights.rows(), numOutputs);
        this->gradients = Eigen::MatrixXd::Random(weights.rows(), numOutputs);
        this->weightsGradients = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        this->biasesGradients = Eigen::VectorXd::Random(weights.rows());
        this->weightsVelocity = Eigen::MatrixXd::Zero(numNeurons, numInputs);
        this->biasesVelocity = Eigen::VectorXd::Zero(numNeurons);
    }

    void Dense::forward(Eigen::MatrixXd &inputs) {
        this->inputs = inputs;
        this->outputs = this->weights * this->inputs;
        this->outputs = this->outputs.colwise() + this->biases;
    }
    void Dense::backward(Eigen::MatrixXd &gradientsIn) {
        this->weightsGradients = (gradientsIn * this->inputs.transpose())/gradientsIn.cols();
        this->biasesGradients = gradientsIn.rowwise().sum()/gradientsIn.cols();
        this->gradients = this->weights.transpose() * gradientsIn;
    }
    void Dense::update(float learningRate) {
        this->weights = this->weights - learningRate * this->weightsGradients;
        this->biases = this->biases - learningRate * this->biasesGradients;
    }
    void Dense::update(float learningRate, float gamma) {
        this->weightsVelocity = gamma * this->weightsVelocity + learningRate * this->weightsGradients;
        this->weights = this->weights - this->weightsVelocity;
        this->biasesVelocity = gamma * this->biasesVelocity + learningRate * this->biasesGradients;
        this->biases = this->biases - this->biasesVelocity;
    }
    void Dense::summary() {
        cout << left << setw(15) << "Dense Layer" << setw(20) << "| Neurons: " << setw(10) << this->numNeurons << setw(20) << "| Inputs: " << setw(10) << this->numInputs << setw(20) << "| Outputs: " << setw(10) <<this->numOutputs << endl;

    }

    int Dense::getTrainableParams() {
        return this->numNeurons * (this->numInputs + 1);
    }

    void Dense::setInputs(Eigen::MatrixXd inputs) {
        this->inputs = inputs;
    }
    Eigen::MatrixXd Dense::getOutputs() {
        return this->outputs;
    }
    Eigen::MatrixXd Dense::getGradients() {
        return this->gradients;
    }
    Eigen::MatrixXd Dense::getWeights() {
        return this->weights;
    }
    Eigen::VectorXd Dense::getBiases() {
        return this->biases;
    }
    void Dense::setWeights(Eigen::MatrixXd weights) {
        this->weights = weights;
    }
    void Dense::setBiases(Eigen::VectorXd biases) {
        this->biases = biases;
    }
    void Dense::setGradients(Eigen::MatrixXd gradients) {
        this->gradients = gradients;
    }
    void Dense::setWeightsGradients(Eigen::MatrixXd weightsGradients) {
        this->weightsGradients = weightsGradients;
    }
    void Dense::setBiasesGradients(Eigen::MatrixXd biasesGradients) {
        this->biasesGradients = biasesGradients;
    }
    void Dense::setOutputs(Eigen::MatrixXd outputs) {
        this->outputs = outputs;
    }
    void Dense::setNumNeurons(int numNeurons) {
        this->numNeurons = numNeurons;
    }
    void Dense::setNumInputs(int numInputs) {
        this->numInputs = numInputs;
    }
    void Dense::setNumOutputs(int numOutputs) {
        this->numOutputs = numOutputs;
    }
    int Dense::getNumNeurons() {
        return this->numNeurons;
    }
    int Dense::getNumInputs() {
        return this->numInputs;
    }
    int Dense::getNumOutputs() {
        return this->numOutputs;
    }
    Eigen::MatrixXd Dense::getWeightsGradients() {
        return this->weightsGradients;
    }
    Eigen::VectorXd Dense::getBiasesGradients() {
        return this->biasesGradients;
    }
    void Dense::initializeWeights() {
        this->weights = Eigen::MatrixXd::Random(this->numNeurons, this->numInputs);
    }
    void Dense::initializeBiases() {
        this->biases = Eigen::VectorXd::Zero(this->numNeurons);
    }
    int* Dense::getOutputShape() {
        int* outputShape = new int[2];
        outputShape[0] = this->numNeurons;
        outputShape[1] = this->numOutputs;
        return outputShape;
    }
    int* Dense::getInputShape() {
        int* inputShape = new int[2];
        inputShape[0] = this->numInputs;
        inputShape[1] = this->numOutputs;
        return inputShape;
    }

    void Dense::saveBiases(const std::string &path, bool append) {
        const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream file;
        if (append) {
            file.open(path, std::ios_base::app);
        } else {
            file.open(path);
        }
        // add new line if appending
        if (append) {
            file << "\n";
        }
        // write shape
//    file << this->numNeurons << "," << "1" << "\n";
        // write biases
        file << this->biases.format(CSVFormat);
        file << "\nend";
        file.close();
    }

    void Dense::saveWeights(const std::string &path, bool append) {
        const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream file;
        if (append) {
            file.open(path, std::ios_base::app);
        } else {
            file.open(path);
        }
        // add new line if appending
        if (append) {
            file << "\n";
        }
        // write shapes
//    file << this->numNeurons << "," << this->numInputs << "\n";
        // write weights
        file << this->weights.format(CSVFormat);
        file << "\nend";
        file.close();
    }

    void Dense::saveLayer(const std::string &path, bool append) {
        this->saveBiases(path, append);
        this->saveWeights(path, true);
    }

}
