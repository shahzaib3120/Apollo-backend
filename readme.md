# Documentation
## Introduction
This project aims at building a machine learning framework using C-plus-plus with minimal dependencies and a modular code structure. The framework is named Apollo and it provides necessary classes and operations for Neural Networks. The modular nature of code allows user to define custom layers type with custom activations and loss functions. We have demonstrated the framework application by building a binary classifier that detects a spam email.<br>
## Dependencies
* C++17
* [Eigen3](http://eigen.tuxfamily.org/)
* [Qt](https://www.qt.io/)
* [CMake](https://cmake.org/)

## GUI
This repository contains the standalone code for Neural Network operations. The GUI is available in a separate repository [here](https://github.com/shahzaib3120/Apollo).<br>
The GUI is built using Qt and it provides a user friendly interface to train and test the model. The GUI is shown below.<br>
Create a new model<br>
![model](https://github.com/shahzaib3120/Apollo-backend/blob/main/images/model.jpg)<br>
Add layers to the model<br>
![addDense](https://github.com/shahzaib3120/Apollo-backend/blob/main/images/addDense.jpg)<br>
Train the model<br>
![train](https://github.com/shahzaib3120/Apollo-backend/blob/main/images/train.jpg)<br>
## Layers
### Layer
This is the base class for all layers. It is an abstract class, and cannot be instantiated directly. It provides the basic functionality for all layers such as numNeurons, numInputs, dW, and dB. It also provides the abstract methods forward and backward as pure virtual functions. These methods must be implemented by all derived classes.
#### Constructors
```cpp
    Layer(int numNeurons, int numInputs, int numOutputs)
    
    Layer(int numNeurons, int *shape)
    
    Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs)
```

<code>Layer(int numNeurons, int numInputs, int numOutputs);</code><br>
Args:<br>
&emsp;<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Layer(int numNeurons, int *shape);</code><br>
Args:<br>
&emsp;<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>
Args:<br>
&emsp;<code>weights</code>: the weights of the layer.<br>
&emsp;<code>biases</code>: the biases of the layer.<br>
&emsp;<code>numOutputs</code>: the number of outputs of the layer.<br><br>

#### Attributes
<code>int numNeurons;</code><br>
<code>int numInputs;</code><br>
<code>int numOutputs;</code><br>
<code>float learningRate = 0.01;</code><br>
<code>Eigen::MatrixXd weights;</code><br>
<code>Eigen::VectorXd biases;</code><br>
<code>Eigen::MatrixXd inputs;</code><br>
<code>Eigen::MatrixXd outputs;</code><br>
<code>Eigen::MatrixXd gradients;</code><br>
<code>Eigen::MatrixXd weightsGradients;</code><br>
<code>Eigen::VectorXd biasesGradients;</code><br>
#### Methods
```c++
    virtual void update(float learningRate) = 0;

    virtual void forward(Eigen::MatrixXd inputs) = 0;

    virtual void backward(Eigen::MatrixXd gradients) = 0;

    virtual void summary() = 0;

    virtual int getTrainableParams() = 0;

    void saveWeights(std::string const &path, bool append = false);

    void saveBiases(std::string const &path, bool append = false);

    void saveGradients(std::string const &path, bool append = false);

    void saveLayer(std::string const &path, bool append = false);
```

<code>void update(float learningRate) = 0;</code><br>
Args:<br>
&emsp;<code>learningRate</code>: the learning rate of the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br><br>

<code>void forward(Eigen::MatrixXd inputs) = 0;</code><br>
Args:<br>
&emsp;<code>inputs</code>: the inputs of the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd gradients) = 0;</code><br>
Args:<br>
&emsp;<code>gradients</code>: the gradients of the layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br><br>

<code>void summary() = 0;</code><br>
This method prints a summary of the layer.<br><br>

<code>int getTrainableParams() = 0;</code><br>
This method returns the number of trainable parameters in the layer.<br><br>

<code>void saveWeights(std::string const &path, bool append = false);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the file where the weights will be saved.<br>
&emsp;<code>append</code>: whether to append the weights to the file or not.<br>
This method saves the weights of the layer to a file.<br><br>

<code>void saveBiases(std::string const &path, bool append = false);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the file where the biases will be saved.<br>
&emsp;<code>append</code>: if true, the biases will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the biases of the layer to a file.<br><br>

<code>void saveGradients(std::string const &path, bool append = false);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the file where the gradients will be saved.<br>
&emsp;<code>append</code>: if true, the gradients will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the gradients of the layer to a file.<br><br>

<code>void saveLayer(std::string const &path, bool append = false);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the file where the layer will be saved.<br>
&emsp;<code>append</code>: if true, the layer will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the layer to a file.<br><br>

### Dense
This is a fully connected layer. It is derived from the Layer class. It provides the forward and backward methods for a fully connected layer.
#### Constructors
```c++
    Dense(int numNeurons, int numInputs, int numOutputs);

    Dense(int numNeurons, int *shape);

    Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);
```
<code>Dense(int numNeurons, int numInputs, int numOutputs);</code><br>
Args:<br>
&emsp;<code>numNeurons</code>: the number of neurons in the layer.<br>
&emsp;<code>numInputs</code>: the number of inputs of the layer.<br>
&emsp;<code>numOutputs</code>: the number of outputs of the layer.<br>

<code>Dense(int numNeurons, int *shape);</code><br>
Args:<br>
&emsp;<code>numNeurons</code>: the number of neurons in the layer.<br>
&emsp;<code>shape</code>: the shape of the layer.<br>

<code>Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>
Args:<br>
&emsp;<code>weights</code>: the weights of the layer.<br>
&emsp;<code>biases</code>: the biases of the layer.<br>
&emsp;<code>numOutputs</code>: the number of outputs of the layer.<br><br>

#### Methods
```c++
    void update(float learningRate) override;

    void forward(Eigen::MatrixXd inputs) override;

    void backward(Eigen::MatrixXd gradients) override;

    void summary() override;

    int getTrainableParams() override;
```
<code>void update(float learningRate);</code><br>
Args:<br>
&emsp;<code>learningRate</code>: the learning rate of the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br><br>

<code>void forward(Eigen::MatrixXd inputs);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the inputs of the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd gradients);</code><br>
Args:<br>
&emsp;<code>gradients</code>: the gradients of the layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br><br>

<code>void summary();</code><br>
This method prints a summary of the layer.<br><br>

<code>int getTrainableParams();</code><br>
This method returns the number of trainable parameters in the layer.<br>

## Activation
This is an activation layer. It is derived from the Layer class. It provides the forward and backward methods for an activation layer.

### Sigmoid
This is a sigmoid activation layer. It is derived from the Activation class. It provides the forward and backward methods for a sigmoid activation layer.
#### Constructors
```c++
    Sigmoid(Eigen::MatrixXd &inputs);

    Sigmoid(int numInputs, int numOutputs);
    
    Sigmoid(int *shape);
```

<code> Sigmoid(Eigen::MatrixXd &inputs);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the inputs of the layer.<br><br>

<code> Sigmoid(int numInputs, int numOutputs);</code><br>
Args:<br>
&emsp;<code>numInputs</code>: the number of inputs of the layer.<br>
&emsp;<code>numOutputs</code>: the number of outputs of the layer.<br><br>


<code> Sigmoid(int *shape);</code><br>
Args:<br>
&emsp;<code>shape</code>: the shape of the layer.<br><br>

#### Attributes
<code>Eigen::MatrixXd inputs;</code><br>
<code>Eigen::MatrixXd outputs;</code><br>
<code>Eigen::MatrixXd gradientsOut;</code><br>
#### Methods
```c++
    void forward(Eigen::MatrixXd &inputs);

    void backward(Eigen::MatrixXd &gradients);

    int *getInputShape();

    int *getOutputShape();

    void summary();
```

<code>void forward(Eigen::MatrixXd &inputs);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the inputs of the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd &gradients);</code><br>
Args:<br>
&emsp;<code>gradients</code>: the gradients of the layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br><br>

<code>int *getInputShape();</code><br>
This method returns the shape of the input matrix to the layer.<br><br>

<code>int *getOutputShape();</code><br>
This method returns the shape of the output matrix from the layer.<br><br>

<code>void summary();</code><br>
This method prints a summary of the layer.<br>

## Loss
Loss is a namespace that contains the loss functions and their derivatives.
```c++
    enum lossFunction{
        BCE,
        MSE
    };
```
### Binary Cross Entropy
This is the binary cross entropy loss function. It is used for binary classification problems.
#### Functions
```c++
    namespace Loss {
        ...        
        std::tuple<Eigen::MatrixXd, float> BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets);

        float BCEValue(Eigen::MatrixXd &loss);
        ...
    }
```

<code>std::tuple<Eigen::MatrixXd, float> BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets);</code><br>
#### Arguments
<code>outputs</code>: output matrix from the network.<br>
<code>targets</code>: target matrix.<br>
This function calculates the binary cross entropy loss and its derivative.<br>
#### Returns
<code>std::tuple<Eigen::MatrixXd, float></code>: gradients and loss value.<br>

<code>float BCEValue(Eigen::MatrixXd &loss);</code><br>
#### Arguments
<code>loss</code>: loss matrix.<br>
This function calculates the binary cross entropy loss value.<br>
#### Returns
<code>float</code>: loss value.<br>

## Dataloader
Dataloader is a class that is used to load data from a file. It is used to load data from a csv file. It is derived from the DataLoader class. It provides the methods to load data from a csv file.

### Constructors
```c++
    Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);

    Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);

    Dataloader(std::string const &path);

    Dataloader(std::string const &path, float trainSplit);

    // NOTE: Make sure the file contains trainLabels in the first column
```

<code>Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);</code><br>
Args:<br>
&emsp;<code>data</code>: the data matrix.<br>
&emsp;<code>labels</code>: the labels matrix.<br>
&emsp;<code>batchSize</code>: the batch size.<br><br>

<code>Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);</code><br>
Args:<br>
&emsp;<code>data</code>: the data matrix.<br>
&emsp;<code>labels</code>: the labels matrix.<br><br>

<code>Dataloader(std::string const &path);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the csv file.<br><br>

<code>Dataloader(std::string const &path, float trainSplit);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the csv file.<br>
&emsp;<code>trainSplit</code>: the split ratio for the train and test data.<br>

### Attributes

<code>Eigen::MatrixXd trainData;</code><br>
<code>Eigen::MatrixXd trainLabels;</code><br>
<code>Eigen::MatrixXd valData;</code><br>
<code>Eigen::MatrixXd valLabels;</code><br>
<code>int batchSize;</code><br>
<code>int numBatches;</code><br>

### Methods
```c++
    Eigen::MatrixXd &getTrainLabels();

    Eigen::MatrixXd &getTrainData();

    Eigen::MatrixXd &getValData();

    Eigen::MatrixXd &getValLabels();
    
    void head(int n);

    int* getTrainDataShape();

    int* getTrainLabelsShape();

    int * getValDataShape();

    int* getValLabelsShape();
```

<code>void head(int n);</code><br>
Args:<br>
&emsp;<code>n</code>: the number of rows to print.<br>
This method prints the first n rows of first n columns of the training data and labels.<br><br>

## Model
Model is a class that is used to create a neural network. It is derived from the Model class. It provides the methods to create a neural network.

### Constructors
```c++
    Model();

    Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);
```


<code>Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);</code><br>
Args:<br>
&emsp;<code>inputShape</code>: the shape of the input matrix.<br>
&emsp;<code>verb</code>: the verbosity of the model.<br>
&emsp;<code>learningRate</code>: the learning rate of the model.<br>
&emsp;<code>numClasses</code>: the number of classes in the dataset.<br><br>

### Attributes
Some attributes are:<br>
```c++
    vector<variant<Dense, Sigmoid>> layers;

    float loss;
```

### Methods
```c++
    void addLayer(MultiType *layer);

    void compile();

    void fit(Eigen::MatrixXd &inputs, Eigen::MatrixXd &labels, int epochs, enum lossFunction, bool verb);

    void
    fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
        enum lossFunction, bool verb);

    void
    fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
        enum lossFunction, bool verb, bool saveEpoch, string filename);

    void
    fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs,
        enum lossFunction, bool verb, bool saveEpoch, string filename, bool earlyStopping, int threshold);

    Eigen::MatrixXd predict(Eigen::MatrixXd inputs);

    void evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType);

    int *getLastLayerOutputShape();

    int *getLastLayerInputShape();

    void summary();

    void saveModel(const std::string &path);

    void loadModel(const std::string &path);
```


<code>MultiType = variant<Dense,Sigmoid>;</code><br>
<code>std::variant</code> is a type-safe union introduced in c++ 17. It is used to store different types of layers in a single vector.<br><br>

<code>void addLayer(MultiType *layer);</code><br>
Args:<br>
&emsp;<code>layer</code>:pointer to the layer to be added.<br>
This method adds a layer to the model.<br><br>

<code>void compile();</code><br>
This method assert the input and output shapes of consecutive layers and throws an error if they are not compatible.<br><br>

<code>void fit(Eigen::MatrixXd &inputs, Eigen::MatrixXd &labels, int epochs, enum lossFunction, bool verb);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the input matrix.<br>
&emsp;<code>labels</code>: the labels matrix.<br>
&emsp;<code>epochs</code>: the number of epochs.<br>
&emsp;<code>lossFunction</code>: the loss function to be used.<br>
&emsp;<code>verb</code>: the verbosity of the model.<br>
This method trains the model.<br><br>
```c++
    enum lossFunction{
        BCE,
        MSE
    };
```

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb);</code><br>
Args:<br>
&emsp;<code>trainX</code>: the training input matrix.<br>
&emsp;<code>trainY</code>: the training labels matrix.<br>
&emsp;<code>valX</code>: the validation input matrix.<br>
&emsp;<code>valY</code>: the validation labels matrix.<br>
&emsp;<code>epochs</code>: the number of epochs.<br>
&emsp;<code>lossFunction</code>: the loss function to be used.<br>
&emsp;<code>verb</code>: the verbosity of the model.<br>
This method trains the model.<br><br>

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb, bool saveEpoch, string filename);</code><br>
Args:<br>
&emsp;<code>trainX</code>: the training input matrix.<br>
&emsp;<code>trainY</code>: the training labels matrix.<br>
&emsp;<code>valX</code>: the validation input matrix.<br>
&emsp;<code>valY</code>: the validation labels matrix.<br>
&emsp;<code>epochs</code>: the number of epochs.<br>
&emsp;<code>lossFunction</code>: the loss function to be used.<br>
&emsp;<code>verb</code>: the verbosity of the model.<br>
&emsp;<code>saveEpoch</code>: whether to save the model after every epoch.<br>
&emsp;<code>filename</code>: the name of the file to save the model.<br>
This method trains the model.<br><br>

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb, bool saveEpoch, string filename, bool earlyStopping, int threshold);</code><br>
Args:<br>
&emsp;<code>trainX</code>: the training input matrix.<br>
&emsp;<code>trainY</code>: the training labels matrix.<br>
&emsp;<code>valX</code>: the validation input matrix.<br>
&emsp;<code>valY</code>: the validation labels matrix.<br>
&emsp;<code>epochs</code>: the number of epochs.<br>
&emsp;<code>lossFunction</code>: the loss function to be used.<br>
&emsp;<code>verb</code>: the verbosity of the model.<br>
&emsp;<code>saveEpoch</code>: whether to save the model after every epoch.<br>
&emsp;<code>filename</code>: the name of the file to save the model.<br>
&emsp;<code>earlyStopping</code>: whether to use early stopping.<br>
&emsp;<code>threshold</code>: the threshold for early stopping.<br>
This method trains the model.<br><br>

<code>Eigen::MatrixXd predict(Eigen::MatrixXd inputs);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the input matrix.<br>
Returns:<br>
&emsp;<code>outputs</code>: the output matrix.<br>
This method predicts the output for the given input.<br><br>

<code>void evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType);</code><br>
Args:<br>
&emsp;<code>inputs</code>: the input matrix.<br>
&emsp;<code>labels</code>: the labels matrix.<br>
&emsp;<code>lossType</code>: the loss function to be used.<br>
This method evaluates the model.<br><br>

<code>int *getLastLayerOutputShape();</code><br>
This method returns the output shape of the last layer.<br><br>

<code>int *getLastLayerInputShape();</code><br>
This method returns the input shape of the last layer.<br><br>

<code>void summary();</code><br>
This method prints the summary of the model including details of each layer including its neurons, input/output shape and total number of trainable params.<br><br>

<code>void saveModel(const std::string &path);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to save the model.<br>
This method saves the model to the given path.<br><br>

<code>void loadModel(const std::string &path);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to load the model from.<br>
This method loads the model from the specified path.<br><br>

## Preprocessing
Preprocessing is a namespace that contains functions to preprocess the data before training or evaluating the model.<br><br>
```c++
    namespace Preprocessing {
        Eigen::MatrixXd normalize(Eigen::MatrixXd matrix);

        Eigen::MatrixXd standardize(Eigen::MatrixXd matrix);

        Eigen::MatrixXd spamPreprocessingFile(const std::string &path);

        Eigen::MatrixXd spamPreprocessing(const std::string &email);
    }
```

### Functions
<code>Eigen::MatrixXd normalize(Eigen::MatrixXd matrix);</code><br>
Args:<br>
&emsp;<code>matrix</code>: the matrix to be normalized.<br>
This function normalizes the input matrix by dividing each element by the maximum value in the matrix.<br><br>

<code>Eigen::MatrixXd standardize(Eigen::MatrixXd matrix);</code><br>
Args:<br>
&emsp;<code>matrix</code>: the matrix to be standardized.<br>
This function standardizes the input matrix by subtracting the mean and dividing by the standard deviation.<br><br>

<code>Eigen::MatrixXd spamPreprocessingFile(const std::string &path);</code><br>
Args:<br>
&emsp;<code>path</code>: the path to the file to be preprocessed.<br>
This function preprocesses the data from the file and returns the input matrix containing the frequency of 3000 predefined words in the email in a specific order.<br>
<b><i>This function is exclusively for the spam email classification problem.</i></b><br><br>

<code>Eigen::MatrixXd spamPreprocessing(const std::string &email);</code><br>
Args:<br>
&emsp;<code>email</code>: the email to be preprocessed.<br>
This function preprocesses the data from the email string and returns the input matrix containing the frequency of 3000 predefined words in the email in a specific order.<br>
<b><i>This function is exclusively for the spam email classification problem.</i></b><br><br>

Dataset used for the spam email classification problem can be found [here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv). Check for more details.<br>
Header of the dataset must be copied to <code>Preprocessing/files</code> directory.<br>

## linalg
linalg is a namespace which provides some linear algebra functions used in the training process.<br><br>
```c++
    namespace linalg {
        Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, int size, int axis);

        Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis);
    }
```

### Functions
<code>Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, int size, int axis);</code><br>
Args:<br>
&emsp;<code>matrix</code>: the matrix to be broadcasted.<br>
&emsp;<code>size</code>: the size of the matrix to be broadcasted to.<br>
&emsp;<code>axis</code>: the axis along which the matrix is to be broadcasted.<br>
This function broadcasts the input matrix to the given size along the given axis.<br><br>

<code>Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis);</code><br>
Args:<br>
&emsp;<code>matrix</code>: the matrix to be broadcasted.<br>
&emsp;<code>shape</code>: the shape of the matrix to be broadcasted to.<br>
&emsp;<code>axis</code>: the axis along which the matrix is to be broadcasted.<br>
This function broadcasts the input matrix to the given shape along the given axis.<br><br>