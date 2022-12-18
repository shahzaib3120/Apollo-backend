//
// Created by shahr on 18/12/2022.
//

#ifndef APOLLO_SPAMDETECTOR_H
#define APOLLO_SPAMDETECTOR_H

#include <string>
#include "../Model/Model.h"
#include "../Utils/Dataloader.h"
class SpamDetector{
private:
    string path;
    Dataloader dataloader;
    Model* model;
public:
    SpamDetector(string path, double trainRatio, double learningRate, int epochs, int batchSize, bool verbose);
    void train();
    void evaluate();
    void saveModel();
    void loadModel();
    void summary();
    void predict(string email);
    void
};


#endif //APOLLO_SPAMDETECTOR_H
