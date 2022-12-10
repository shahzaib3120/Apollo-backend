//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_DENSE_H
#define APOLLO_DENSE_H

#include "Layer.h"
class Dense:public Layer {
public:
    Dense(int numNeurons, int numInputs, int numOutputs);
    Dense(int numNeurons, int* shape);
    void forward(Eigen::MatrixXd inputs) override;
    void backward(Eigen::MatrixXd gradientsIn) override;
    void update(float learningRate) override;
};


#endif //APOLLO_DENSE_H
