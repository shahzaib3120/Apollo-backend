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
void Apollo::Model::addLayer(MultiType *layer) {
    this->layers.push_back(*layer);
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

            else if(this->layers[i+1].index() == 2){
                Relu relu = get<Relu>(this->layers[i+1]);
//                assert(dense.getOutputShape() == sigmoid.getInputShape());
                if(!compareShapes(dense.getOutputShape(), relu.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\ndense.getOutputShape() != relu.getInputShape()"+"\n"+to_string(dense.getOutputShape()[0])+" x "+to_string(dense.getOutputShape()[1])+" != "+to_string(relu.getInputShape()[0])+" x "+to_string(relu.getInputShape()[1]));
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
        else if(this->layers[i].index() == 2){
            Relu relu = get<Relu>(this->layers[i]);
            if(this->layers[i+1].index() == 0){
                Dense dense = get<Dense>(this->layers[i+1]);
//                assert(relu.getOutputShape() == dense.getInputShape());
                if(!compareShapes(relu.getOutputShape(), dense.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nrelu.getOutputShape() != dense.getInputShape()"+"\n"+to_string(relu.getOutputShape()[0])+" x "+to_string(relu.getOutputShape()[1])+" != "+to_string(dense.getInputShape()[0])+" x "+to_string(dense.getInputShape()[1]));
                }
            }
            else if(this->layers[i+1].index() == 1){
                Sigmoid sigmoid = get<Sigmoid>(this->layers[i+1]);
//                assert(relu.getOutputShape() == sigmoid.getInputShape());
                if(!compareShapes(relu.getOutputShape(), sigmoid.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nrelu.getOutputShape() != sigmoid.getInputShape()"+"\n"+to_string(relu.getOutputShape()[0])+" x "+to_string(relu.getOutputShape()[1])+" != "+to_string(sigmoid.getInputShape()[0])+" x "+to_string(sigmoid.getInputShape()[1]));
                }
            }
            else if(this->layers[i+1].index() == 2){
                Relu relu2 = get<Relu>(this->layers[i+1]);
//                assert(relu.getOutputShape() == relu2.getInputShape());
                if(!compareShapes(relu.getOutputShape(), relu2.getInputShape())){
                    throw invalid_argument("Input shape of layer "+to_string(i+2)+" is not equal to output shape of layer "+to_string(i+1)+"\nrelu.getOutputShape() != relu2.getInputShape()"+"\n"+to_string(relu.getOutputShape()[0])+" x "+to_string(relu.getOutputShape()[1])+" != "+to_string(relu2.getInputShape()[0])+" x "+to_string(relu2.getInputShape()[1]));
                }
            }
        }
    }
    if(this->verbose){
        cout<<"Model compiled successfully"<<endl;
        system("pause");
    }
}
Eigen::MatrixXd Apollo::Model::forward(Eigen::MatrixXd inputs) {
    for(int i=0; i<this->layers.size(); i++){
        if(layers[i].index() == 0){
            Dense dense = get<Dense>(layers[i]);
            dense.forward(inputs);
            inputs = dense.getOutputs();
            // update layer
            layers[i] = dense;
        }
        else if(layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(layers[i]);
            sigmoid.forward(inputs);
            inputs = sigmoid.getOutputs();
            // update the layer in the layers vector
            layers[i] = sigmoid;
        } else if(layers[i].index() == 2){
            Relu relu = get<Relu>(layers[i]);
            relu.forward(inputs);
            inputs = relu.getOutputs();
//            cout << inputs << endl;
            // update the layer in the layers vector
            layers[i] = relu;
        }
    }
    return inputs;
}
void Apollo::Model::backward(Eigen::MatrixXd gradientsIn) {
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
        else if(this->layers[i].index() == 2){
            Relu relu = get<Relu>(this->layers[i]);
            relu.backward(gradientsIn);
            gradientsIn = relu.getGradients();
            // update the layer in the layers vector
            this->layers[i] = relu;
        }
    }
}
void Apollo::Model::update(float learningRate) {
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
        // TODO: dummy; could be removed
        else if(layer.index() == 2){
            Relu relu = get<Relu>(layer);
            relu.update(learningRate);
            // update the layer in the layers vector
            layer = relu;
        }
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
    if(layers.size()>0){
        if(this->layers[this->layers.size()-1].index() == 0){
            Dense dense = get<Dense>(this->layers[this->layers.size()-1]);
            shape= dense.getOutputShape();
        }
        else if(this->layers[this->layers.size()-1].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[this->layers.size()-1]);
            shape= sigmoid.getOutputShape();
        }
        else if(this->layers[this->layers.size()-1].index() == 2){
            Relu relu = get<Relu>(this->layers[this->layers.size()-1]);
//            int* shape1 = relu.getOutputShape();
//            shape[0] = shape1[0];
//            shape[1] = shape1[1];
            shape = relu.getOutputShape();
        }
        // print shape
        return shape;
    }else{
        shape[0] = 0;
        shape[1] = 0;
        return shape;
    }
}
int* Apollo::Model::getLastLayerInputShape() {
    int* shape = new int[2];
    if(layers.size()>0){
        if(this->layers[this->layers.size()-1].index() == 0){
            Dense dense = get<Dense>(this->layers[this->layers.size()-1]);
            shape = dense.getInputShape();
        }
        else if(this->layers[this->layers.size()-1].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[this->layers.size()-1]);
            shape= sigmoid.getInputShape();
        }
        else if(this->layers[this->layers.size()-1].index() == 2){
            Relu relu = get<Relu>(this->layers[this->layers.size()-1]);
            shape= relu.getInputShape();
        }
        return shape;
    }else{
        shape[0] = 0;
        shape[1] = 0;
        return shape;
    }
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
        for(int i=0; i<this->layers.size(); i++){
            if(this->layers[i].index() == 0){
                denseLayers++;
            }
        }
        file << denseLayers;
        bool append = true;
        for(auto layer : this->layers){
            if(layer.index() == 0){
                Dense dense = get<Dense>(layer);
                dense.saveLayer(path, append);
                append = true;
            }
            else if(layer.index() == 1) {
                Sigmoid sigmoid = get<Sigmoid>(layer);
                append = true;
            }
            else if(layer.index() == 2) {
                Relu relu = get<Relu>(layer);
                append = true;
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
            MultiType layer = Dense(weights, biases, this->inputShape[1]);
            this->layers.push_back(layer);
            // append sigmoid layer after each dense layer
            int* sigmoidShape = new int[2];
            sigmoidShape[0] = weights.rows();
            sigmoidShape[1] = this->inputShape[1];
            MultiType sigmoid = Sigmoid(sigmoidShape);
            this->layers.push_back(sigmoid);
        }
        file.close();
    }
    else{
        cout<<"Error opening the file"<<endl;
    }
}

void Apollo::Model::summary() {
    cout<<"Model Summary"<<endl;
    for(int i=0; i<this->layers.size(); i++){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            dense.summary();
        }
        else if(this->layers[i].index() == 1){
            Sigmoid sigmoid = get<Sigmoid>(this->layers[i]);
            sigmoid.summary();
        }
        else if(this->layers[i].index() == 2){
            Relu relu = get<Relu>(this->layers[i]);
            relu.summary();
        }
    }
    // print the output shape
    int* outputShape = this->getLastLayerOutputShape();
    cout << "==========================================" << endl;
    cout<<"Input Shape: "<<this->inputShape[0]<<", "<<this->inputShape[1]<<endl;
    cout<<"Output Shape: "<<outputShape[0]<<", "<<outputShape[1]<<endl;
    // print trainable parameters
    int trainableParams = 0;
    for(int i=0; i<this->layers.size(); i++){
        if(this->layers[i].index() == 0){
            Dense dense = get<Dense>(this->layers[i]);
            trainableParams += dense.getTrainableParams();
        }
    }
    cout<< "Total Layers: " << this->layers.size() << endl;
    cout<<"Trainable Parameters: "<<trainableParams<<endl;
    cout << "==========================================" << endl;
    system("pause");
}
