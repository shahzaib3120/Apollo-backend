# Documentation
## Introduction
This is a simple example of a readme file. It is written in markdown, and can be used to provide documentation for your project. It is a good idea to include a readme file in your project, as it will help others to understand what your project is about, and how to use it.
## Layers
### Layer
This is the base class for all layers. It is an abstract class, and cannot be instantiated directly. It provides the basic functionality for all layers such as numNeurons, numInputs, dW, and dB. It also provides the abstract methods forward and backward as pure virtual functions. These methods must be implemented by all derived classes.
#### Constructors
        Layer(int numNeurons, int numInputs, int numOutputs)

        Layer(int numNeurons, int *shape)

        Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs)

<code>Layer(int numNeurons, int numInputs, int numOutputs);</code><br>
<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Layer(int numNeurons, int *shape);</code><br>
<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>
<code>weights</code>: the weights of the layer.<br>
<code>biases</code>: the biases of the layer.<br>
<code>numOutputs</code>: the number of outputs of the layer.<br><br>
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

        virtual void update(float learningRate) = 0;

        virtual void forward(Eigen::MatrixXd inputs) = 0;

        virtual void backward(Eigen::MatrixXd gradients) = 0;

        virtual void summary() = 0;

        virtual int getTrainableParams() = 0;

        void saveWeights(std::string const &path, bool append = false);

        void saveBiases(std::string const &path, bool append = false);

        void saveGradients(std::string const &path, bool append = false);

        void saveLayer(std::string const &path, bool append = false);

<code>void update(float learningRate) = 0;</code><br>
<code>learningRate</code>: learning rate for the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br><br>

<code>void forward(Eigen::MatrixXd inputs) = 0;</code><br>
<code>inputs</code>: input matrix to the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd gradients) = 0;</code><br>
<code>gradients</code>: gradients from the next layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br><br>

<code>void summary() = 0;</code><br>
This method prints a summary of the layer.<br>
<code>int getTrainableParams() = 0;</code><br>
This method returns the number of trainable parameters in the layer.<br><br>

<code>void saveWeights(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the weights will be saved.<br>
<code>append</code>: if true, the weights will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the weights of the layer to a file.<br><br>

<code>void saveBiases(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the biases will be saved.<br>
<code>append</code>: if true, the biases will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the biases of the layer to a file.<br><br>

<code>void saveGradients(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the gradients will be saved.<br>
<code>append</code>: if true, the gradients will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the gradients of the layer to a file.<br><br>

<code>void saveLayer(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the layer will be saved.<br>
<code>append</code>: if true, the layer will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the layer to a file.<br><br>


### Dense
This is a fully connected layer. It is derived from the Layer class. It provides the forward and backward methods for a fully connected layer.
#### Constructors

        Dense(int numNeurons, int numInputs, int numOutputs);

        Dense(int numNeurons, int* shape);

        Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);

<code>Dense(int numNeurons, int numInputs, int numOutputs);</code><br>
<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Dense(int numNeurons, int *shape);</code><br>
<code>numNeurons</code>: the number of neurons in the layer.<br><br>
<code>Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>
<code>weights</code>: the weights of the layer.<br>
<code>biases</code>: the biases of the layer.<br>
<code>numOutputs</code>: the number of outputs of the layer.<br><br>

#### Methods

        void forward(Eigen::MatrixXd inputs) override;

        void backward(Eigen::MatrixXd gradientsIn) override;

        void update(float learningRate) override;

        void summary() override;

        int getTrainableParams() override;


<code>void update(float learningRate);</code><br>
<code>learningRate</code>: learning rate for the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br><br>

<code>void forward(Eigen::MatrixXd inputs);</code><br>
<code>inputs</code>: input matrix to the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd gradients);</code><br>
<code>gradients</code>: gradients from the next layer.<br>
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

        Sigmoid(Eigen::MatrixXd &inputs);

        Sigmoid(int numInputs, int numOutputs);

        Sigmoid(int *shape);

<code> Sigmoid(Eigen::MatrixXd &inputs);</code><br>
<code>inputs</code>: input matrix to the layer.<br><br>

<code> Sigmoid(int numInputs, int numOutputs);</code><br>
<code>numInputs</code>: number of inputs to the layer.<br>
<code>numOutputs</code>: number of outputs from the layer.<br><br>

<code> Sigmoid(int *shape);</code><br>
<code>shape</code>: shape of the input matrix to the layer.<br>

#### Attributes
<code>Eigen::MatrixXd inputs;</code><br>
<code>Eigen::MatrixXd outputs;</code><br>
<code>Eigen::MatrixXd gradientsOut;</code><br>
#### Methods

        void forward(Eigen::MatrixXd &inputs);

        void backward(Eigen::MatrixXd &gradients);

        int *getInputShape();

        int *getOutputShape();

        void summary();

<code>void forward(Eigen::MatrixXd &inputs);</code><br>
<code>inputs</code>: input matrix to the layer.<br>
This method calculates the output of the layer given the input.<br><br>

<code>void backward(Eigen::MatrixXd &gradients);</code><br>
<code>gradients</code>: gradients from the next layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br><br>

<code>int *getInputShape();</code><br>
This method returns the shape of the input matrix to the layer.<br><br>
<code>int *getOutputShape();</code><br>
This method returns the shape of the output matrix from the layer.<br><br>

<code>void summary();</code><br>
This method prints a summary of the layer.<br><br>

## Loss
Loss is a namespace that contains the loss functions and their derivatives.

        enum lossFunction{
            BCE,
            MSE
        };
### Binary Cross Entropy
This is the binary cross entropy loss function. It is used for binary classification problems.
#### Functions

        namespace Loss {
            ...        
            std::tuple<Eigen::MatrixXd, float> BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets);

            float BCEValue(Eigen::MatrixXd &loss);
            ...
        }

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

        Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);

        Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);

        Dataloader(std::string const &path);

        Dataloader(std::string const &path, float trainSplit);

        // NOTE: Make sure the file contains trainLabels in the first column

<code>Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);</code><br>
<code>data</code>: data matrix.<br>
<code>labels</code>: labels matrix.<br>
<code>batchSize</code>: batch size.<br><br>

<code>Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);</code><br>
<code>data</code>: data matrix.<br>
<code>labels</code>: labels matrix.<br><br>

<code>Dataloader(std::string const &path);</code><br>
<code>path</code>: path to the csv file.<br><br>

<code>Dataloader(std::string const &path, float trainSplit);</code><br>
<code>path</code>: path to the csv file.<br>
<code>trainSplit</code>: percentage of data to be used for training.<br>

### Attributes

<code>Eigen::MatrixXd trainData;</code><br>
<code>Eigen::MatrixXd trainLabels;</code><br>
<code>Eigen::MatrixXd valData;</code><br>
<code>Eigen::MatrixXd valLabels;</code><br>
<code>int batchSize;</code><br>
<code>int numBatches;</code><br>

### Methods

        Eigen::MatrixXd &getTrainLabels();

        Eigen::MatrixXd &getTrainData();

        Eigen::MatrixXd &getValData();

        Eigen::MatrixXd &getValLabels();
        
        void head(int n);

        int* getTrainDataShape();

        int* getTrainLabelsShape();

        int * getValDataShape();

        int* getValLabelsShape();

<code>void head(int n);</code><br>
<code>n</code>: number of rows to be printed.<br>
This method prints the first n rows of first n columns of the training data and labels.<br><br>

## Model
Model is a class that is used to create a neural network. It is derived from the Model class. It provides the methods to create a neural network.

### Constructors

        Model();

        Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);


<code>Model(int *inputShape, bool verb, float learningRate = 0.001, int numClasses = 1);</code><br>
<code>inputShape</code>: shape of the input matrix.<br>
<code>verb</code>: verbosity of the model, to show the progress.<br>
<code>learningRate</code>: learning rate of the model.<br>
<code>numClasses</code>: number of classes in the dataset.<br>

### Attributes
Some attributes are:<br>

    vector<variant<Dense, Sigmoid>> layers;

    float loss;

### Methods

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


<code>MultiType = variant<Dense,Sigmoid>;</code><br>
<code>std::variant</code> is a type-safe union introduced in c++ 17. It is used to store different types of layers in a single vector.<br><br>

<code>void addLayer(MultiType *layer);</code><br>
<code>layer</code>: pointer to the layer to be added.<br>
This method adds a layer to the model.<br><br>

<code>void compile();</code><br>
This method assert the input and output shapes of consecutive layers and throws an error if they are not compatible.<br><br>

<code>void fit(Eigen::MatrixXd &inputs, Eigen::MatrixXd &labels, int epochs, enum lossFunction, bool verb);</code><br>
<code>inputs</code>: input matrix.<br>
<code>labels</code>: labels matrix.<br>
<code>epochs</code>: number of epochs.<br>
<code>lossFunction</code>: loss function to be used.<br>

    enum lossFunction{
        BCE,
        MSE
    };
<code>verb</code>: verbosity of the model, to show the progress.<br>
This method trains the model.<br><br>

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb);</code><br>
<code>trainX</code>: training input matrix.<br>
<code>trainY</code>: training labels matrix.<br>
<code>valX</code>: validation input matrix.<br>
<code>valY</code>: validation labels matrix.<br>
<code>epochs</code>: number of epochs.<br>
<code>lossFunction</code>: loss function to be used.<br>
<code>verb</code>: verbosity of the model, to show the progress.<br>
This method trains the model.<br><br>

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb, bool saveEpoch, string filename);</code><br>
<code>trainX</code>: training input matrix.<br>
<code>trainY</code>: training labels matrix.<br>
<code>valX</code>: validation input matrix.<br>
<code>valY</code>: validation labels matrix.<br>
<code>epochs</code>: number of epochs.<br>
<code>lossFunction</code>: loss function to be used.<br>
<code>verb</code>: verbosity of the model, to show the progress.<br>
<code>saveEpoch</code>: boolean to save the model after every epoch.<br>
<code>filename</code>: filename to save the model.<br>
This method trains the model.<br><br>

<code>void fit(Eigen::MatrixXd &trainX, Eigen::MatrixXd &trainY, Eigen::MatrixXd &valX, Eigen::MatrixXd &valY, int epochs, enum lossFunction, bool verb, bool saveEpoch, string filename, bool earlyStopping, int threshold);</code><br>
<code>trainX</code>: training input matrix.<br>
<code>trainY</code>: training labels matrix.<br>
<code>valX</code>: validation input matrix.<br>
<code>valY</code>: validation labels matrix.<br>
<code>epochs</code>: number of epochs.<br>
<code>lossFunction</code>: loss function to be used.<br>
<code>verb</code>: verbosity of the model, to show the progress.<br>
<code>saveEpoch</code>: boolean to save the model after every epoch.<br>
<code>filename</code>: filename to save the model.<br>
<code>earlyStopping</code>: boolean to stop training if the validation loss does not decrease for a certain number of epochs.<br>
<code>threshold</code>: number of epochs to wait before stopping the training.<br>
This method trains the model.<br><br>

<code>Eigen::MatrixXd predict(Eigen::MatrixXd inputs);</code><br>
<code>inputs</code>: input matrix.<br>
This method returns the output of the model.<br><br>

<code>void evaluate(Eigen::MatrixXd inputs, Eigen::MatrixXd labels, enum lossFunction lossType);</code><br>
<code>inputs</code>: input matrix.<br>
<code>labels</code>: labels matrix.<br>
<code>lossType</code>: loss function to be used.<br>
This method evaluates the model.<br><br>

<code>int *getLastLayerOutputShape();</code><br>
This method returns the output shape of the last layer.<br><br>

<code>int *getLastLayerInputShape();</code><br>
This method returns the input shape of the last layer.<br><br>

<code>void summary();</code><br>
This method prints the summary of the model including details of each layer including its neurons, input/output shape and total number of trainable params.<br><br>

<code>void saveModel(const std::string &path);</code><br>
<code>path</code>: path to save the model.<br>
This method saves the model to the specified path.<br><br>

<code>void loadModel(const std::string &path);</code><br>
<code>path</code>: path to load the model from.<br>
This method loads the model from the specified path.<br><br>

## Preprocessing
Preprocessing is a namespace that contains functions to preprocess the data before training or evaluating the model.<br><br>

        namespace Preprocessing {
            Eigen::MatrixXd normalize(Eigen::MatrixXd matrix);
    
            Eigen::MatrixXd standardize(Eigen::MatrixXd matrix);
    
            Eigen::MatrixXd spamPreprocessingFile(const std::string &path);
    
            Eigen::MatrixXd spamPreprocessing(const std::string &email);
        }

### Functions
<code>Eigen::MatrixXd normalize(Eigen::MatrixXd matrix);</code><br>
<code>matrix</code>: input matrix.<br>
This function normalizes the input matrix by dividing each element by the maximum value in the matrix.<br><br>

<code>Eigen::MatrixXd standardize(Eigen::MatrixXd matrix);</code><br>
<code>matrix</code>: input matrix.<br>
This function standardizes the input matrix by subtracting the mean and dividing by the standard deviation.<br><br>

<code>Eigen::MatrixXd spamPreprocessingFile(const std::string &path);</code><br>
<code>path</code>: path to the file.<br>
This function preprocesses the data from the file and returns the input matrix containing the frequency of 3000 predefined words in the email in a specific order.<br>
<b><i>This function is exclusively for the spam email classification problem.</i></b><br><br>

<code>Eigen::MatrixXd spamPreprocessing(const std::string &email);</code><br>
<code>email</code>: email string.<br>
This function preprocesses the data from the email string and returns the input matrix containing the frequency of 3000 predefined words in the email in a specific order.<br>
<b><i>This function is exclusively for the spam email classification problem.</i></b><br><br>

Dataset used for the spam email classification problem can be found [here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv). Check for more details.<br>
Header of the dataset must be copied to <code>Preprocessing/files</code> directory.<br>

## linalg
linalg is a namespace which provides some linear algebra functions used in the training process.<br><br>

        namespace linalg {
            Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, int size, int axis);
    
            Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis);
        }

### Functions
<code>Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, int size, int axis);</code><br>
<code>matrix</code>: input matrix.<br>
<code>size</code>: size of the output matrix.<br>
<code>axis</code>: axis along which the matrix is broadcasted.<br>
This function broadcasts the input matrix along the specified axis.<br><br>

<code>Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis);</code><br>
<code>matrix</code>: input matrix.<br>
<code>shape</code>: shape of the output matrix.<br>
<code>axis</code>: axis along which the matrix is broadcasted.<br>
This function broadcasts the input matrix along the specified axis.<br><br>








