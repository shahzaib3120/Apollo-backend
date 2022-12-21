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
namespace Apollo {
    using MultiType = variant<Dense,Sigmoid>;
    enum lossFunction{
        BCE,
        MSE
    };
    class Model {
    private:
        vector<variant<Dense, Sigmoid>> layers;
        int numClasses;
        bool verbose;
        float learningRate;
        int *inputShape;

        void forward(Eigen::MatrixXd inputs);

        void backward(Eigen::MatrixXd gradientsIn);

        void update(float learningRate);

        static double accuracy(Eigen::MatrixXd outputs, Eigen::MatrixXd targets);

        void lossFunction(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction loss);

        double validationLoss(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction loss);

        Eigen::MatrixXd gradients;
        float loss;

        static bool compareShapes(int const *shape1, int const *shape2);

        float startTime;

        void saveData(string filename, Eigen::MatrixXd matrix);

    public:
        Model();

        Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);

        void addLayer(MultiType *layer);

        void compile();

        void fit(Eigen::MatrixXd &inputs, Eigen::MatrixXd &labels, int epochs, enum lossFunction, bool verb);

        void
        fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
            enum lossFunction, bool verb);

        void
        fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
            enum lossFunction, bool verb, bool saveEpoch, string filename);

        void
        fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
            enum lossFunction, bool verb, bool saveEpoch, string filename, bool earlyStopping, int threshold);

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

        void setLayers(vector<variant<Dense, Sigmoid>> layers);

        vector<variant<Dense, Sigmoid>> getLayers();

        int *getLastLayerOutputShape();

        int *getLastLayerInputShape();

        // To be implemented
        void summary();

        void saveModel(const std::string &path);

        void loadModel(const std::string &path);

    };

}
#endif //APOLLO_MODEL_H
