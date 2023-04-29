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
#include "../Layers/Relu.h"
using namespace std;
namespace Apollo {
    enum lossFunction{
        BCE,
        MSE
    };
    class Model {
    private:
        vector<Layer*> layers;
        int numClasses;
        bool verbose;
        float learningRate;
        float gamma;
        int *inputShape;
        Eigen::MatrixXd gradients;
        float loss;

//        void forward(Eigen::MatrixXd inputs);
        Eigen::MatrixXd forward(Eigen::MatrixXd inputs);

        void backward(Eigen::MatrixXd gradientsIn);

        void update(float learningRate);
        void update(float learningRate, float gamma);

        void lossFunction(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction loss);

        double validationLoss(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction loss);

        void saveData(string filename, Eigen::MatrixXd matrix);

        static double accuracy(Eigen::MatrixXd outputs, Eigen::MatrixXd targets);
        static bool compareShapes(int const *shape1, int const *shape2);

    public:
        Model();

        Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);
        void addLayer(Layer *layer);
        void compile();

        void
        fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY,
            string savePath, bool saveEpoch = true, int epochs = 1000, enum lossFunction lossType = MSE, bool verb = true,
            bool earlyStopping = true, int threshold=5, float gamma = 0.1);

        Eigen::MatrixXd predict(Eigen::MatrixXd inputs);

        void evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType);

        void setLearningRate(float learningRate);

        float getLearningRate();

        void setVerbose(bool verbose);

        bool getVerbose();

        void setNumClasses(int numClasses);

        int getNumClasses();

        void setInputShape(int *inputShape);

        int *getInputShape();

        void setLayers(Layer* layers);

        vector<Layer*> getLayers();

        int *getLastLayerOutputShape();

        int *getLastLayerInputShape();

        // To be implemented
        void summary();

        void saveModel(const std::string &path);

        void loadModel(const std::string &path);

    };

}
#endif //APOLLO_MODEL_H
