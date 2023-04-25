//
// Created by HP on 4/25/2023.
//

#ifndef APOLLO_RELU_H
#define APOLLO_RELU_H
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
namespace Apollo {

    class Relu {
    private:
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd gradientsOut;
//        int* shape = new int[2];
    public:
        // constructors
        Relu();
        Relu(Eigen::MatrixXd &inputs);
        Relu(int numInputs, int numOutputs);
        Relu(int *shape);

        // methods
        void setInputs(Eigen::MatrixXd &inputs);
        Eigen::MatrixXd getOutputs();
        Eigen::MatrixXd getGradients();
        void setGradients(Eigen::MatrixXd &gradients);
        void setOutputs(Eigen::MatrixXd &outputs);

        void forward();
        void forward(Eigen::MatrixXd &inputs);
        void backward();
        void backward(Eigen::MatrixXd &gradients);
        void update(float learningRate);

        int *getInputShape();
        int *getOutputShape();

        void summary();

//        ~Relu(){
//            delete shape;
//        };
    };

} // Apollo

#endif //APOLLO_RELU_H
