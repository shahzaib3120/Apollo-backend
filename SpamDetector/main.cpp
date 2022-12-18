#include <iostream>
#include "../Utils/Dataloader.h"
#include "../Layers/Dense.h"
#include "../Model/Model.h"
using namespace std;
int main() {
//    string path = "E:/Learning-E/Apollo-backend/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/winequality-red.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/data.csv";
//    string path = "F:/Machine-Learning/Apollo-backend/Dataset/data.csv";
    string path = "F:/Machine-Learning/Apollo-backend/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/sampleCircle.csv";
//    string path = "F:/Machine-Learning/Apollo-backend/Dataset/sampleCircle.csv";
    DataLoader::Dataloader dataloader(path, 0.8);
//    dataloader.head(5);
    int* shape = dataloader.getTrainDataShape();
    auto* model =  new Model(shape, true, 0.01, 1);
    MultiType d1 = Dense(3, shape);
    model->addLayer(&d1);
    MultiType s1 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s1);
    MultiType d2 = Dense(1, model->getLastLayerOutputShape());
    model->addLayer(&d2);
    MultiType s2 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s2);
    model->compile();
    model->summary();
    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(), 100, BCE, true);
    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);
    model->saveModel("F:/Machine-Learning/Apollo-backend/Models/spam.csv");
//    model->loadModel("F:/Machine-Learning/Apollo-backend/Models/spam.csv");
//    model->compile();
//    model->summary();
    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);
    return 0;
}
