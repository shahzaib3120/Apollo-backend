//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_LAYER_H
#define APOLLO_LAYER_H
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <string>

namespace Apollo
{
    class Layer
    {
    protected:
        std::string name;
    public:
        Layer();
        Layer(std::string name);
        virtual void update(float learningRate) = 0;
        virtual void forward(Eigen::MatrixXd &inputs) = 0;
        virtual void backward(Eigen::MatrixXd &gradients) = 0;

        // TODO: make summary optional and add a try catch block in the summary method of the model
        virtual void summary() = 0;
        virtual int getTrainableParams();

        // TODO: add a method to save the weights and biases to a file
        virtual void saveWeights(std::string const &path, bool append = false);
        virtual void saveBiases(std::string const &path, bool append = false);
        virtual void saveLayer(std::string const &path, bool append = false);
        virtual void saveGradients(std::string const &path, bool append = false);
        
        virtual void setInputs(Eigen::MatrixXd &inputs);
        virtual void setOutputs(Eigen::MatrixXd &outputs);
        virtual void setGradients(Eigen::MatrixXd &gradients);

        virtual Eigen::MatrixXd getInputs();
        virtual Eigen::MatrixXd getOutputs();
        virtual Eigen::MatrixXd getGradients();

        virtual int *getInputShape();
        virtual int *getOutputShape();

        std::string getName();
    };

}
#endif // APOLLO_LAYER_H
