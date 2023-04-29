//
// Created by HP on 4/25/2023.
//

#ifndef APOLLO_RELU_H
#define APOLLO_RELU_H
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include "Layer.h"

namespace Apollo {
    class Relu: public Layer{
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd gradientsOut;
    public:
        // constructors
        Relu();
        Relu(std::string name, Eigen::MatrixXd &inputs);
        Relu(std::string name, int numInputs, int numOutputs);
        Relu(std::string name, int *shape);

        // methods
        void setInputs(Eigen::MatrixXd &inputs);
        void setOutputs(Eigen::MatrixXd &outputs);
        void setGradients(Eigen::MatrixXd &gradients);

        Eigen::MatrixXd getInputs();
        Eigen::MatrixXd getOutputs();
        Eigen::MatrixXd getGradients();

        void forward(Eigen::MatrixXd &inputs);
        void backward(Eigen::MatrixXd &gradients);
        void update(float learningRate);
        void update(float lr, float gamma);

        int *getInputShape();
        int *getOutputShape();

        void summary();

    };

} // Apollo

#endif //APOLLO_RELU_H
