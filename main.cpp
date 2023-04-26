#include <iostream>
#include "Utils/Dataloader.h"
#include "Layers/Dense.h"
#include "Model/Model.h"
#include "Preprocessing/Preprocessing.h"
using namespace std;
using namespace Apollo;
int main() {
//    Binary Classification
//    string path = "../Dataset/emails-formatted.csv";
//    Dataloader dataloader(path, 0.8);
//    dataloader.head(5);
//    int* shape = dataloader.getTrainDataShape();
//    auto* model =  new Model(shape, true, 0.01, 1);
//    MultiType d1 = Dense(5, shape);
//    model->addLayer(&d1);
//    MultiType s1 = Sigmoid(model->getLastLayerOutputShape());
//    model->addLayer(&s1);
//    MultiType d2 = Dense(1, model->getLastLayerOutputShape());
//    model->addLayer(&d2);
//    MultiType s2 = Sigmoid(model->getLastLayerOutputShape());
//    model->addLayer(&s2);
//    model->compile();
//    model->summary();
//    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(), 5, BCE, true, true, "../Models/test.csv", true, 3);
//    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);

//    Linear Regression
    string path = "../Dataset/quadratic.csv";
    Dataloader dataloader(path, 0.7);
//    dataloader.head(5);
    int* shape = dataloader.getTrainDataShape();
    auto* model =  new Model(shape, true, 0.002, 1);
    model->addLayer(new Dense("dense1",5, shape));
    model->addLayer(new Relu("relu1", model->getLastLayerOutputShape()));
    model->addLayer(new Dense("dense2",1, model->getLastLayerOutputShape()));
    model->compile();
    model->summary();
    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(),
               "../Models/testRegression.csv", true , 1000 , MSE, true, true, 3);
    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), MSE);
    Eigen::Matrix prediction = model->predict(dataloader.getTrainData());
    cout << dataloader.getTrainData() << endl;
    cout << prediction << endl;
    system("pause");
    return 0;
}
