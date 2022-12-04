//
// Created by HP on 11/27/2022.
//

#include "Model.h"
#include "../Loss/Loss.h"
#include <iostream>
#include <cassert>
Model::Model() {
    this->layers = {};
}
Model::Model(int *inputShape, bool verbose, float learningRate, int numClasses) {
    this->inputShape = inputShape;
    this->verbose = verbose;
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
    assert(this->layers.size() > 0);
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
                if(compareShapes(dense.getOutputShape(), dense2.getInputShape()) == false){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\ndense.getOutputShape() != dense2.getInputShape()"+"\n"+to_string(dense.getOutputShape()[0])+" x "+to_string(dense.getOutputShape()[1])+" != "+to_string(dense2.getInputShape()[0])+" x "+to_string(dense2.getInputShape()[1]));
                }
            }

            else if(this->layers[i+1].index() == 1){
                Sigmoid sigmoid = get<Sigmoid>(this->layers[i+1]);
//                assert(dense.getOutputShape() == sigmoid.getInputShape());
                if(compareShapes(dense.getOutputShape(), sigmoid.getInputShape()) == false){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\ndense.getOutputShape() != sigmoid.getInputShape()"+"\n"+to_string(dense.getOutputShape()[0])+" x "+to_string(dense.getOutputShape()[1])+" != "+to_string(sigmoid.getInputShape()[0])+" x "+to_string(sigmoid.getInputShape()[1]));
                }
            }
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            if(this->layers[i+1].index() == 0){
                Dense dense2 = get<Dense>(this->layers[i+1]);
//                assert(sigmoid.getOutputShape() == dense2.getInputShape());
                if(compareShapes(sigmoid.getOutputShape(), dense2.getInputShape()) == false){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nsigmoid.getOutputShape() != dense2.getInputShape()"+"\n"+to_string(sigmoid.getOutputShape()[0])+" x "+to_string(sigmoid.getOutputShape()[1])+" != "+to_string(dense2.getInputShape()[0])+" x "+to_string(dense2.getInputShape()[1]));
                }
            }
            else if(this->layers[i+1].index() == 1){
                Sigmoid sigmoid2 = get<Sigmoid>(this->layers[i+1]);
//                assert(sigmoid.getOutputShape() == sigmoid2.getInputShape());
                if(compareShapes(sigmoid.getOutputShape(), sigmoid2.getInputShape()) == false){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nsigmoid.getOutputShape() != sigmoid2.getInputShape()"+"\n"+to_string(sigmoid.getOutputShape()[0])+" x "+to_string(sigmoid.getOutputShape()[1])+" != "+to_string(sigmoid2.getInputShape()[0])+" x "+to_string(sigmoid2.getInputShape()[1]));
                }
            }
        }
    }
    if(this->verbose){
        cout<<"Model compiled successfully"<<endl;
    }
}
void Model::forward(Eigen::MatrixXd inputs) {
    for(int i=0; i<this->layers.size(); i++){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            dense.forward(inputs);
            inputs = dense.getOutputs();
//            // update the layer in the layers vector
            this->layers[i] = dense;
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            sigmoid.forward(inputs);
            inputs = sigmoid.getOutputs();
//            // update the layer in the layers vector
            this->layers[i] = sigmoid;
        }
    }
}
void Model::backward(Eigen::MatrixXd gradients) {
    for(int i=this->layers.size()-1; i>=0; i--){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            dense.backward(gradients);
            gradients = dense.getGradients();
            // update the layer in the layers vector
            this->layers[i] = dense;
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            sigmoid.backward(gradients);
            gradients = sigmoid.getGradients();
            // update the layer in the layers vector
            this->layers[i] = sigmoid;
        }
    }
}
void Model::update(float learningRate) {
    for(int i=0; i<this->layers.size(); i++){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            dense.update(learningRate);
            // update the layer in the layers vector
            this->layers[i] = dense;
        }
        else if(this->layers[i].index() == 1){
            continue;
        }
    }
}
void Model::fit(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, int epochs, enum lossFunction loss , bool verbose) {
    this->verbose = verbose;
    if(this->verbose){
        cout<<"Training model..."<<endl;
    }
    for(int i=0; i<epochs; i++){
        this->forward(inputs);
        Eigen::MatrixXd outputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();
        this->lossFunction(outputs, labels, BCE);
        // print the stats
        if(this->verbose){
            cout << "==========================================" << endl;
            cout<<"Epoch: "<<i+1<<"/"<<epochs<<", Loss: "<<this->loss<<", Accuracy: "<< this->accuracy(outputs, labels)<<endl;
            cout << "==========================================" << endl;
        }
        this->backward(this->gradients);
        this->update(this->learningRate);
    }
    if(this->verbose){
        cout<<"Model trained successfully"<<endl;
    }
}
void Model::lossFunction(Eigen::MatrixXd outputs, Eigen::MatrixXd targets, enum lossFunction lossType) {
    if(lossType == MSE){
        // TODO: Add reference instead of value
        this->gradients = Loss::MSE(outputs, targets);
    }
    else if(lossType == BCE){
        // get gradients and loss from the loss function
        auto [grad, lossVal] = Loss::BCE(outputs, targets);
        this->gradients = grad;
        this->loss = lossVal;
        // TODO : loss BECOMES NULL AFTERWARDS (fixed)
    }
}

Eigen::MatrixXd Model::predict(Eigen::MatrixXd inputs) {
    this->forward(inputs);
    Eigen::MatrixXd outputs = this->layers[this->layers.size()-1].index() == 0 ? get<Dense>(this->layers[this->layers.size()-1]).getOutputs() : get<Sigmoid>(this->layers[this->layers.size()-1]).getOutputs();
    return outputs;
}
void Model::evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction loss) {
    Eigen::MatrixXd outputs = this->predict(inputs);
    this->lossFunction(outputs, labels, loss);
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
bool Model::compareShapes(int *shape1, int *shape2) {
    if(shape1[0] == shape2[0] && shape1[1] == shape2[1]){
        return true;
    }
    return false;
}


