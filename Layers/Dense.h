//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_DENSE_H
#define APOLLO_DENSE_H

#include "Layer.h"
class Dense:public Layer {
private:
    void forward() override;
    void backward() override;
    void update() override;
public:
    Dense(int numNeurons, int numInputs, int numOutputs);
    void forward(Eigen::MatrixXd inputs) override;
    void backward(Eigen::MatrixXd gradients) override;

};


#endif //APOLLO_DENSE_H
