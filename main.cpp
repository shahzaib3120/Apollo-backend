#include <iostream>
#include "Utils/Dataloader.h"
#include "Model/Model.h"
using namespace std;
int main() {
    string path = "E:/Learning-E/Apollo/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo/Dataset/winequality-red.csv";
//    string path = "E:/Learning-E/Apollo/Dataset/data.csv";
//    string path = "F:/Machine-Learning/Apollo/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo/Dataset/sampleCircle.csv";
    Dataloader dataloader(path, 0.8);
    dataloader.head(5);
    int* shape = dataloader.getTrainDataShape();
    int* labelShape = dataloader.getTrainLabelsShape();
    // print shape of trainData
    cout << "Data shape: " << shape[0] << " " << shape[1] << endl;
    // print shape of trainLabels
    cout << "Labels shape: " << labelShape[0] << " " << labelShape[1] << endl;
    // invert shape
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
    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(), 100, BCE, true);
//    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), 1000, BCE, true);
//    // TODO: check if dimension changes would be required for GUI
    system("pause");
    return 0;
}
