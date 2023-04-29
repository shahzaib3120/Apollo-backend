//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_SIGMOID_H
#define APOLLO_SIGMOID_H
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include "Layer.h"

namespace Apollo {
    using namespace std;
    class Sigmoid: public Layer {
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd gradientsOut;
    public:
        Sigmoid();
        Sigmoid(string name, Eigen::MatrixXd &inputs);
        Sigmoid(string name, int numInputs, int numOutputs);
        Sigmoid(string name, int *shape);

        void summary();
        
        // overridden methods
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


    };
}
#endif //APOLLO_SIGMOID_H
