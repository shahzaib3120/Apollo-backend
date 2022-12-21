//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_DENSE_H
#define APOLLO_DENSE_H

#include "Layer.h"
namespace Apollo{
    class Dense:public Layer {
    public:
        Dense(int numNeurons, int numInputs, int numOutputs);
        Dense(int numNeurons, int* shape);
        Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);
        void forward(Eigen::MatrixXd inputs) override;
        void backward(Eigen::MatrixXd gradientsIn) override;
        void update(float learningRate) override;
        void summary() override;
        int getTrainableParams() override;
    };
}


#endif //APOLLO_DENSE_H
