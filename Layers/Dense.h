//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_DENSE_H
#define APOLLO_DENSE_H

#include "Layer.h"
namespace Apollo{
    class Dense:public Layer {
    protected:
        int numNeurons;
        int numInputs;
        int numOutputs;
        Eigen::MatrixXd inputs;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd gradients;
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
        Eigen::MatrixXd weightsGradients;
        Eigen::VectorXd biasesGradients;
        Eigen::MatrixXd weightsVelocity;
        Eigen::VectorXd biasesVelocity;
    public:
        Dense(std::string name,int numNeurons, int numInputs, int numOutputs);
        Dense(std::string name, int numNeurons, int* shape);
        Dense(std::string name, Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);
        void forward(Eigen::MatrixXd &inputs) override;
        void backward(Eigen::MatrixXd &gradientsIn) override;
        void update(float learningRate) override;
        void update(float learningRate, float gamma) override;
        void summary() override;
        int getTrainableParams() override;

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

        int *getOutputShape();

        int *getInputShape();

        void initializeWeights();

        void initializeBiases();

        Eigen::MatrixXd getWeightsGradients();

        Eigen::VectorXd getBiasesGradients();


        void saveWeights(std::string const &path, bool append = false);
        void saveBiases(std::string const &path, bool append = false);
        void saveLayer(std::string const &path, bool append = false);

    };
}


#endif //APOLLO_DENSE_H
