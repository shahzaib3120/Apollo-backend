//
// Created by HP on 11/27/2022.
//

#ifndef APOLLO_MODEL_H
#define APOLLO_MODEL_H
#include <variant>
#include <vector>
#include "../Layers/Dense.h"
#include "../Layers/Softmax.h"
#include "../Layers/Sigmoid.h"
using namespace std;
using MultiType = variant<Dense,Sigmoid>;
enum lossFunction{
    BCE,
    MSE
};
class Model {
private:
    vector<variant<Dense,Sigmoid>> layers;
    int numClasses;
    bool verbose;
    float learningRate;
    int* inputShape;
    void forward(Eigen::MatrixXd inputs);
    void backward(Eigen::MatrixXd gradients);
    void update(float learningRate);
    double accuracy(Eigen::MatrixXd outputs, Eigen::MatrixXd targets);
    void lossFunction(Eigen::MatrixXd outputs, Eigen::MatrixXd targets, enum lossFunction loss);
    Eigen::MatrixXd gradients;
    float loss;
    bool compareShapes(int* shape1, int* shape2);
public:
    Model();
    Model(int* inputShape, bool verbose, float learningRate = 0.001, int numClasses=1);
    void addLayer(MultiType *layer);
    void compile();
    void fit(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, int epochs,enum lossFunction, bool verbose);
    Eigen::MatrixXd predict(Eigen::MatrixXd inputs);
    void evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction loss);
    void setLearningRate(float learningRate);
    float getLearningRate();
    void setVerbose(bool verbose);
    bool getVerbose();
    void setNumClasses(int numClasses);
    int getNumClasses();
    void setInputShape(int* inputShape);
    int* getInputShape();
    void setLayers(vector<variant<Dense,Sigmoid>> layers);
    vector<variant<Dense,Sigmoid>> getLayers();
    void summary();
    void saveModel(std::string path);
    void loadModel(std::string path);
    int* getLastLayerOutputShape();
    int* getLastLayerInputShape();
};


#endif //APOLLO_MODEL_H
