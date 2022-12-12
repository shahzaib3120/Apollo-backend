//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_LAYER_H
#define APOLLO_LAYER_H
#include <Eigen/Dense>

class Layer {
protected:
    int numNeurons;
    int numInputs;
    int numOutputs;
    float gamma=0.9;
    float learningRate=0.01;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd outputs;
    Eigen::MatrixXd gradients;
    Eigen::MatrixXd weightsGradients;
    Eigen::VectorXd biasesGradients;

public:
    Layer(int numNeurons, int numInputs, int numOutputs);
    Layer(int numNeurons, int* shape);
    // TODO: CHECK IF PARENT CLASS ARRAY CAN HAVE CHILD ELEMENTS
//    Layer(Eigen::MatrixXd inputs);
    void setInputs(Eigen::MatrixXd inputs);
    Eigen::MatrixXd getOutputs();
    Eigen::MatrixXd getGradients();
    Eigen::MatrixXd getWeights();
    Eigen::VectorXd getBiases();
    void setWeights(Eigen::MatrixXd weights);
    void setBiases(Eigen::VectorXd biases);
    void setGradients(Eigen::MatrixXd gradients);
    void setWeightsGradients(Eigen::MatrixXd weightsGradients);
    void setBiasesGradients(Eigen::MatrixXd biasesGradients);
    void setOutputs(Eigen::MatrixXd outputs);
    void setNumNeurons(int numNeurons);
    void setNumInputs(int numInputs);
    void setNumOutputs(int numOutputs);
    int getNumNeurons();
    int getNumInputs();
    int getNumOutputs();
    int* getOutputShape();
    int* getInputShape();
    void initializeWeights();
    void initializeBiases();
    Eigen::MatrixXd getWeightsGradients();
    Eigen::VectorXd getBiasesGradients();
    virtual void update(float learningRate) = 0;
    virtual void forward(Eigen::MatrixXd inputs) = 0;
    virtual void backward(Eigen::MatrixXd gradients) = 0;
};


#endif //APOLLO_LAYER_H
