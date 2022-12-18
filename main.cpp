#include <iostream>
#include "Utils/Dataloader.h"
#include "Layers/Dense.h"
#include "Model/Model.h"
#include "Preprocessing/Preprocessing.h"
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
    MultiType d1 = Dense(5, shape);
    model->addLayer(&d1);
    MultiType s1 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s1);
    MultiType d2 = Dense(1, model->getLastLayerOutputShape());
    model->addLayer(&d2);
    MultiType s2 = Sigmoid(model->getLastLayerOutputShape());
    model->addLayer(&s2);
    model->compile();
//    model->summary();
    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(), 100, BCE, true, true, "F:/Machine-Learning/Apollo-backend/Models/spam-100.csv", true, 3);
    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);
//    string email;
//    cout << "Enter email: ";
//    getline(cin, email);
    // shape for spam detector = {1, 3000}
//    int shape[2] = {1, 3000};
//    auto* model =  new Model(shape, true, 0.01, 1);
//    model->loadModel("F:/Machine-Learning/Apollo-backend/Models/spam.csv");
//    model->compile();
//    string emailPath = "F:/Machine-Learning/Apollo-backend/email.txt";
//    Eigen::MatrixXd emailMatrix = Preprocessing::spamPreprocessingFile(emailPath);
//    // print emailMatrix shape
////    cout << emailMatrix.rows() << " " << emailMatrix.cols() << endl;
////    cout << emailMatrix << endl;
//    cout << model->predict(emailMatrix);
    system("pause");
    return 0;
}
