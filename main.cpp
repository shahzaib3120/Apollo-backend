#include <iostream>
#include <Eigen/Dense>
#include "Utils/Dataloader.h"
#include "Model/Model.h"
#include "Loss/Loss.h"
using namespace std;
int main() {
    string path = "E:/Learning-E/Apollo/Dataset/emails-formatted.csv";
//    string path = "F:/Machine-Learning/Apollo/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo/Dataset/sampleCircle.csv";
    Dataloader dataloader(path);
    dataloader.head(5);
    int* shape = dataloader.getDataShape();
    int* labelShape = dataloader.getLabelsShape();
    // print shape of data
    cout << "Data shape: " << shape[0] << " " << shape[1] << endl;
    // print shape of labels
    cout << "Labels shape: " << labelShape[0] << " " << labelShape[1] << endl;
    // invert shape
    Model* model =  new Model(shape, true, 0.01, 1);
    MultiType d1 = Dense(3, shape);
    model->addLayer(&d1);
    MultiType s1 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s1);
    MultiType d2 = Dense(1, model->getLastLayerOutputShape());
    model->addLayer(&d2);
    MultiType s2 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s2);
    model->compile();
    model->fit(dataloader.getData(), dataloader.getLabels(), 100, BCE, true);
//    // TODO: check if dimension changes would be required for GUI
    return 0;
}
