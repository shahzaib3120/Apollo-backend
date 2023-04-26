//
// Created by HP on 11/27/2022.
//

#include "Model.h"
#include "../Loss/Loss.h"
#include <iostream>
#include <cassert>
#include "iomanip"
#include <chrono>
#include <fstream>

Apollo::Model::Model() {
    this->layers = {};
}
Apollo::Model::Model(int *inputShape, bool verb, float learningRate, int numClasses) {
    this->inputShape = inputShape;
    this->verbose = verb;
    this->learningRate = learningRate;
    this->numClasses = numClasses;
}
void Apollo::Model::addLayer(Layer *layer) {
    this->layers.push_back(layer);
}
void Apollo::Model::compile() {
    //TODO: optimizations. get shape once and store it in a variable instead of calling it every time
    if(this->verbose){
        cout<<"Compiling model..."<<endl;
    }
// check if layers are added
    assert(this->layers.empty() == false);
// check if input shape is set
    assert(this->inputShape != nullptr);
    // loop over layers and check if input shape of next layer is equal to output shape of previous layer
    for(int i=0; i<this->layers.size()-1; i++){
        if(!compareShapes(this->layers[i]->getOutputShape(), this->layers[i+1]->getInputShape())){
            cout<<"Input shape of layer "<<i+2<<" is not equal to output shape of layer "<<i+1<<endl;
            cout<<"Input shape of layer "<<i+2<<" is: ";
            cout << this->layers[i+1]->getInputShape()[0] << "x" << this->layers[i+1]->getInputShape()[1] << endl;
            cout<<endl;
            cout<<"Output shape of layer "<<i+1<<" is: ";
            cout << this->layers[i]->getOutputShape()[0] << "x" << this->layers[i]->getOutputShape()[1] << endl;
            cout<<endl;
            cout<<"Please check the input and output shapes of the layers"<<endl;
            exit(1);
        }
    }

    if(this->verbose){
        cout<<"Model compiled successfully"<<endl;
        system("pause");
    }
}
Eigen::MatrixXd Apollo::Model::forward(Eigen::MatrixXd inputs) {
    for(auto &layer: this->layers){
        layer->forward(inputs);
        inputs = layer->getOutputs();
    }
    return inputs;
}
void Apollo::Model::backward(Eigen::MatrixXd gradientsIn) {
    for_each(this->layers.rbegin(), this->layers.rend(), [&](Layer *layer){
        layer->backward(gradientsIn);
        gradientsIn = layer->getGradients();
    });
}
void Apollo::Model::update(float learningRate) {
    for(auto &layer: this->layers){
        layer->update(learningRate);
    }
}

void Apollo::Model::fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY,
                        string savePath, bool saveEpoch , int epochs , enum lossFunction lossType, bool verb,
                        bool earlyStopping, int threshold, float gamma) {
    /* Args: trainX, trainY, valX, valY, epochs, lossType, verbose, saveEpoch, savePath, earlyStopping, threshold
            trainX: training data - Eigen::MatrixXd
            trainY: training labels - Eigen::MatrixXd
            valX: validation data - Eigen::MatrixXd
            valY: validation labels - Eigen::MatrixXd
            epochs: number of epochs - int
            lossType: loss function - enum lossFunction
            verb: verbose - bool
            saveEpoch: save model after each epoch - bool
            savePath: path to save the model - string
            earlyStopping: early stopping - bool
            threshold: threshold for early stopping - int
            Returns: None
     */
    // set verbose
    this->verbose = verb;
    // set start time
    auto start = chrono::high_resolution_clock::now();
    double prevValAccuracy = 0;
    for(int i=0; i<epochs; i++){
        // compute validation lossType and accuracy
        Eigen::MatrixXd valOutputs = this->forward(valX);
        double valLoss = this->validationLoss(valOutputs, valY, lossType);
        double valAccuracy = accuracy(valOutputs, valY);
        // compute training forward prop
        Eigen::MatrixXd outputs = this->forward(trainX);
        this->lossFunction(outputs, trainY, lossType);
        // print the stats
        if(this->verbose){
            // clear the screen
            system("cls");
            cout << "Training model ..." << endl;
            cout<<"Epoch: "<<i+1<<"/"<<epochs << " :" << endl;
            cout<<left << setw(20) <<"Training Loss: "<< setw(10)<<this->loss<<setw(25) <<" | Training Accuracy: "<<setw(10)<< accuracy(outputs, trainY)<<endl;
            cout<< left<< setw(20)<<"Validation Loss: "<<setw(10)<<valLoss<<setw(25) <<" | Validation Accuracy: "<<setw(10)<<valAccuracy<<endl;
//            cout << "----------------------------------------------------------" << endl;
            // create a progress bar for the training
            cout << "Training Progress: " << flush;
            for(int j=0; j<50; j++){
                if(j < (i+1)*50/epochs){
                    cout << "#" << flush;
                }
                else{
                    cout << " " << flush;
                }
            }
            cout << " " << (i+1)*100/epochs << "%";
            // compute eta
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end - start);
            int eta = (duration.count()/(i+1))*(epochs-(i+1));
            cout << " | ETA: " << eta/3600 << "h " << (eta%3600)/60 << "m " << eta%60 << "s" << endl;

        }
        this->backward(this->gradients);
        this->update(this->learningRate);
        int thresholdCounter = 0;
        if(earlyStopping){
            // if valAccuracy is less than previous valAccuracy in threshold epochs, stop training
            if(valAccuracy < prevValAccuracy){
                thresholdCounter++;
                if(thresholdCounter == threshold){
                    cout << "Early stopping at epoch " << i+1 << endl;
                    break;
                }
            }else{
                prevValAccuracy = valAccuracy;
                cout << "Validation accuracy increased to " << valAccuracy << endl;
                if(saveEpoch){
                    this->saveModel(savePath);
                    cout << "Saving model ..." << endl;
                }
                thresholdCounter = 0;
            }
        }

    }
    if(this->verbose){
        cout<<"Model trained successfully"<<endl;
        system("pause");
    }
}

void Apollo::Model::lossFunction(Eigen::MatrixXd& outputs, Eigen::MatrixXd& targets, enum lossFunction lossType) {
    if(lossType == MSE){
        auto [grad, lossVal] = Loss::MSE(outputs, targets);
        this->gradients = grad;
        this->loss = lossVal;
    }
    else if(lossType == BCE){
        // get gradientsOut and loss from the loss function
        auto [grad, lossVal] = Loss::BCE(outputs, targets);
        this->gradients = grad;
//        cout << "Grad: " << endl << this->gradientsOut << endl;
        this->loss = lossVal;
    }
}
Eigen::MatrixXd Apollo::Model::predict(Eigen::MatrixXd inputs) {
    Eigen::MatrixXd outputs = this->forward(inputs);
//    Eigen::MatrixXd outputs;
//    int lastIndex = this->layers.size()-1;
//    switch(this->layers[lastIndex].index()){
//        case 0:
//            outputs = get<Dense>(this->layers[lastIndex]).getOutputs();
//            break;
//        case 1:
//            outputs = get<Sigmoid>(this->layers[lastIndex]).getOutputs();
//            break;
//        case 2:
//            outputs = get<Relu>(this->layers[lastIndex]).getOutputs();
//            break;
//        default:
//            break;
//    }
    return outputs;
}
void Apollo::Model::evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType) {
    Eigen::MatrixXd outputs = this->predict(inputs);
    this->lossFunction(outputs, labels, lossType);
    cout<<"Loss: "<<this->loss<<endl;
}
double Apollo::Model::accuracy(Eigen::MatrixXd outputs, Eigen::MatrixXd labels) {
    double accuracy = 0.0;
    for(int i=0; i<outputs.cols(); i++){
        if(outputs(0, i) > 0.5 && labels(0, i) == 1){
            accuracy += 1;
        }
        else if(outputs(0, i) <= 0.5 && labels(0, i) == 0){
            accuracy += 1;
        }
    }
    return accuracy/double(outputs.cols());
}
int* Apollo::Model::getLastLayerOutputShape() {
    int* shape = new int[2];
    assert(this->layers.size() > 0);
    shape = this->layers[this->layers.size()-1]->getOutputShape();
}
int* Apollo::Model::getLastLayerInputShape() {
    int* shape = new int[2];
    assert(this->layers.size() > 0);
    shape = this->layers[this->layers.size()-1]->getInputShape();
}
bool Apollo::Model::compareShapes(int const *shape1, int const *shape2) {
    if(shape1[0] == shape2[0] && shape1[1] == shape2[1]){
        return true;
    }
    return false;
}

double Apollo::Model::validationLoss(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction lossType) {
    if(lossType == MSE){
        auto [grad, lossVal] = Loss::MSE(outputs, targets);
        this->gradients = grad;
        this->loss = lossVal;
    }
    else if(lossType == BCE){
        auto [grad, lossVal] = Loss::BCE(outputs, targets);
        return lossVal;
    }
    return 0.0;
}

// Model save
void Apollo::Model::saveModel(const std::string & path) {
    std::ofstream file(path);
    if(file.is_open()){
        int denseLayers = 0;
        for(auto layer : this->layers){
            if(typeid(*layer) == typeid(Dense)){
                denseLayers++;
            }
        }
        file << denseLayers;
        bool append = true;
        for(auto layer : this->layers){
            if(typeid(*layer) == typeid(Dense)){
                layer->saveLayer(path, append);
                append = false;
            }
        }
        file.close();
    }
    else{
        cout<<"Error opening the file"<<endl;
    }
}

void Apollo::Model::loadModel(const std::string &path) {
    // Args: path to the model file (string)
    // Returns: None
    // description : loads the model from the file

    this->layers.clear();
    std::ifstream file(path);
    if(file.is_open()){
        // load the model
        // load the layers
        int numLayers;
        file >> numLayers;
        std::string line;
        std::getline(file, line);
        for(int i=0; i<numLayers; i++){
            Eigen::VectorXd biases;
            Eigen::MatrixXd weights;
            vector<double> matrixEntries;
            string matrixRowString;
            string matrixEntry;
            int matrixRowNumber = 0;
            while(std::getline(file, matrixRowString)){
                if(matrixRowString == "end"){
                    break;
                }
                else{
                    std::stringstream matrixRowStream(matrixRowString);
                    while(std::getline(matrixRowStream, matrixEntry, ',')){
                        matrixEntries.push_back(std::stod(matrixEntry));
                    }
                    matrixRowNumber++;
                }
            }
            biases = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
            matrixEntries.clear();
            matrixRowNumber = 0;
            while(std::getline(file, matrixRowString)){
                if(matrixRowString == "end"){
                    break;
                }
                else{
                    std::stringstream matrixRowStream(matrixRowString);
                    while(std::getline(matrixRowStream, matrixEntry, ',')){
                        matrixEntries.push_back(std::stod(matrixEntry));
                    }
                    matrixRowNumber++;
                }
            }
            weights = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
            Layer* layer = new Dense("dense" ,weights, biases, this->inputShape[1]);
//            MultiType layer = Dense(weights, biases, this->inputShape[1]);
            this->layers.push_back(layer);
            // append sigmoid layer after each dense layer
            int* sigmoidShape = new int[2];
            sigmoidShape[0] = weights.rows();
            sigmoidShape[1] = this->inputShape[1];
            layer = new Sigmoid("sigmoid", sigmoidShape);
            this->layers.push_back(layer);
        }
        file.close();
    }
    else{
        cout<<"Error opening the file"<<endl;
    }
}

void Apollo::Model::summary() {
    cout<<"Model Summary"<<endl;
    for(auto layer : this->layers){
        layer->summary();
    }
    // print the output shape
    int* outputShape = this->getLastLayerOutputShape();
    cout << "==========================================" << endl;
    cout<<"Input Shape: "<<this->inputShape[0]<<", "<<this->inputShape[1]<<endl;
    cout<<"Output Shape: "<<outputShape[0]<<", "<<outputShape[1]<<endl;
    // print trainable parameters
    int trainableParams = 0;
    for(int i=0; i<this->layers.size(); i++){
        try{
            trainableParams += this->layers[i]->getTrainableParams();
        }
        catch(int e){
            cout<<"Layer "<<i<<" is not trainable"<<endl;
        }
    }
    cout<< "Total Layers: " << this->layers.size() << endl;
    cout<<"Trainable Parameters: "<<trainableParams<<endl;
    cout << "==========================================" << endl;
    system("pause");
}
