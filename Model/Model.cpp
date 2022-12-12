//
// Created by HP on 11/27/2022.
//

#include "Model.h"
#include "../Loss/Loss.h"
#include <iostream>
#include <cassert>
#include "iomanip"
#include <chrono>
Model::Model() {
    this->layers = {};
}
Model::Model(int *inputShape, bool verb, float learningRate, int numClasses) {
    this->inputShape = inputShape;
    this->verbose = verb;
    this->learningRate = learningRate;
    this->numClasses = numClasses;
}
void Model::addLayer(MultiType *layer) {
    this->layers.push_back(*layer);
}
void Model::compile() {
    if(this->verbose){
        cout<<"Compiling model..."<<endl;
    }
// check if layers are added
    assert(this->layers.empty() == false);
// check if input shape is set
    assert(this->inputShape != nullptr);
    // loop over layers and check if input shape of next layer is equal to output shape of previous layer
    for(int i=0; i<this->layers.size()-1; i++){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            if(this->layers[i+1].index() == 0){
                Dense dense2 = get<Dense>(this->layers[i+1]);
                // check that input shape of next layer is equal to output shape of previous layer
                // if not, throw an error mentioning the layer number
//                if(dense.getOutputShape() != dense2.getInputShape()){
                if(!compareShapes(dense.getOutputShape(), dense2.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\ndense.getOutputShape() != dense2.getInputShape()"+"\n"+to_string(dense.getOutputShape()[0])+" x "+to_string(dense.getOutputShape()[1])+" != "+to_string(dense2.getInputShape()[0])+" x "+to_string(dense2.getInputShape()[1]));
                }
            }

            else if(this->layers[i+1].index() == 1){
                Sigmoid sigmoid = get<Sigmoid>(this->layers[i+1]);
//                assert(dense.getOutputShape() == sigmoid.getInputShape());
                if(!compareShapes(dense.getOutputShape(), sigmoid.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\ndense.getOutputShape() != sigmoid.getInputShape()"+"\n"+to_string(dense.getOutputShape()[0])+" x "+to_string(dense.getOutputShape()[1])+" != "+to_string(sigmoid.getInputShape()[0])+" x "+to_string(sigmoid.getInputShape()[1]));
                }
            }
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            if(this->layers[i+1].index() == 0){
                Dense dense2 = get<Dense>(this->layers[i+1]);
//                assert(sigmoid.getOutputShape() == dense2.getInputShape());
                if(!compareShapes(sigmoid.getOutputShape(), dense2.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nsigmoid.getOutputShape() != dense2.getInputShape()"+"\n"+to_string(sigmoid.getOutputShape()[0])+" x "+to_string(sigmoid.getOutputShape()[1])+" != "+to_string(dense2.getInputShape()[0])+" x "+to_string(dense2.getInputShape()[1]));
                }
            }
            else if(this->layers[i+1].index() == 1){
                Sigmoid sigmoid2 = get<Sigmoid>(this->layers[i+1]);
//                assert(sigmoid.getOutputShape() == sigmoid2.getInputShape());
                if(!compareShapes(sigmoid.getOutputShape(), sigmoid2.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nsigmoid.getOutputShape() != sigmoid2.getInputShape()"+"\n"+to_string(sigmoid.getOutputShape()[0])+" x "+to_string(sigmoid.getOutputShape()[1])+" != "+to_string(sigmoid2.getInputShape()[0])+" x "+to_string(sigmoid2.getInputShape()[1]));
                }
            }
        }
    }
    if(this->verbose){
        cout<<"Model compiled successfully"<<endl;
        system("pause");
    }
}
void Model::forward(Eigen::MatrixXd inputs) {
    for(auto &layer: this->layers){
        if(layer.index() == 0){
            Dense dense = get<Dense>(layer);
            dense.forward(inputs);
            inputs = dense.getOutputs();
            // update layer
            layer = dense;
        }
        else if(layer.index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(layer);
            sigmoid.forward(inputs);
            inputs = sigmoid.getOutputs();
            // update the layer in the layers vector
            layer = sigmoid;
        }
    }
}
void Model::backward(Eigen::MatrixXd gradientsIn) {
    for(int i=this->layers.size()-1; i>=0; i--){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            dense.backward(gradientsIn);
            gradientsIn = dense.getGradients();
            // update the layer in the layers vector
            this->layers[i] = dense;
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            sigmoid.backward(gradientsIn);
            gradientsIn = sigmoid.getGradients();
            // update the layer in the layers vector
            this->layers[i] = sigmoid;
        }
    }
}
void Model::update(float learningRate) {
    for(auto &layer: this->layers){
        if(layer.index() == 0){
            Dense dense = get<Dense>(layer);
            dense.update(learningRate);
            // update the layer in the layers vector
            layer = dense;
        }
        else if(layer.index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(layer);
            sigmoid.update(learningRate);
            // update the layer in the layers vector
            layer = sigmoid;
        }
    }
}

// TODO: compute validation accuracy and loss

void Model::fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction lossType, bool verb) {
    this->verbose = verb;
    // set start time
    auto start = chrono::high_resolution_clock::now();
    for(int i=0; i<epochs; i++){
        // compute validation lossType and accuracy
        this->forward(valX);
        Eigen::MatrixXd valOutputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();this->forward(trainX);
        double valLoss = this->validationLoss(valOutputs, valY, lossType);
        double valAccuracy = accuracy(valOutputs, valY);

        // compute training forward prop
        this->forward(trainX);
        Eigen::MatrixXd outputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();this->forward(trainX);
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
    }
    if(this->verbose){
        cout<<"Model trained successfully"<<endl;
        system("pause");
    }
}
 void Model::fit(Eigen::MatrixXd &inputs, Eigen::MatrixXd &labels, int epochs, enum lossFunction lossType , bool verb) {
    this->verbose = verb;
    if(this->verbose){
        cout<<"Training model..."<<endl;
    }
    for(int i=0; i<epochs; i++){
        this->forward(inputs);
        Eigen::MatrixXd outputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();
        this->lossFunction(outputs, labels, lossType);
        // print the stats
        if(this->verbose){
            cout << "===================================================" << endl;
            cout<<"Epoch: "<<i+1<<"/"<<epochs<<", Loss: "<<this->loss<<", Accuracy: "<< accuracy(outputs, labels)<<endl;
            cout << "===================================================" << endl;
        }
        this->backward(this->gradients);
        this->update(this->learningRate);
    }
    if(this->verbose){
        cout<<"Model trained successfully"<<endl;
    }
}
void Model::lossFunction(Eigen::MatrixXd& outputs, Eigen::MatrixXd& targets, enum lossFunction lossType) {
    if(lossType == MSE){
        this->gradients = Loss::MSE(outputs, targets);
    }
    else if(lossType == BCE){
        // get gradientsOut and loss from the loss function
        auto [grad, lossVal] = Loss::BCE(outputs, targets);
        this->gradients = grad;
//        cout << "Grad: " << endl << this->gradientsOut << endl;
        this->loss = lossVal;
    }
}
Eigen::MatrixXd Model::predict(Eigen::MatrixXd inputs) {
    this->forward(inputs);
    Eigen::MatrixXd outputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();
    return outputs;
}
void Model::evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType) {
    Eigen::MatrixXd outputs = this->predict(inputs);
    this->lossFunction(outputs, labels, lossType);
    cout<<"Loss: "<<this->loss<<endl;
}
double Model::accuracy(Eigen::MatrixXd outputs, Eigen::MatrixXd labels) {
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
int* Model::getLastLayerOutputShape() {
    int* shape = new int[2];
    if(this->layers[this->layers.size()-1].index() == 0){
        Dense dense = get<Dense>(this->layers[this->layers.size()-1]);
        shape= dense.getOutputShape();
    }
    else if(this->layers[this->layers.size()-1].index() == 1){
        Sigmoid sigmoid = get<Sigmoid>(this->layers[this->layers.size()-1]);
        shape= sigmoid.getOutputShape();
    }
    return shape;
}
int* Model::getLastLayerInputShape() {
    int* shape = new int[2];
    if(this->layers[this->layers.size()-1].index() == 0){
        Dense dense = get<Dense>(this->layers[this->layers.size()-1]);
        shape = dense.getInputShape();
    }
    else if(this->layers[this->layers.size()-1].index() == 1){
        Sigmoid sigmoid = get<Sigmoid>(this->layers[this->layers.size()-1]);
        shape= sigmoid.getInputShape();
    }
    return shape;
}
bool Model::compareShapes(int const *shape1, int const *shape2) {
    if(shape1[0] == shape2[0] && shape1[1] == shape2[1]){
        return true;
    }
    return false;
}

double Model::validationLoss(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets, enum lossFunction lossType) {
    if(lossType == MSE){
        this->gradients = Loss::MSE(outputs, targets);
        return 0.0;
    }
    else if(lossType == BCE){
        auto [grad, lossVal] = Loss::BCE(outputs, targets);
        return lossVal;
    }
}

