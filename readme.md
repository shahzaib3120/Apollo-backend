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
<code>Layer(int numNeurons, int *shape);</code><br>
<code>Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>
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
<code>void update(float learningRate) = 0;</code><br>
<code>learningRate</code>: learning rate for the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br><br>
<code>void forward(Eigen::MatrixXd inputs) = 0;</code><br>
<code>inputs</code>: input matrix to the layer.<br>
This method calculates the output of the layer given the input.<br>
<code>void backward(Eigen::MatrixXd gradients) = 0;</code><br>
<code>gradients</code>: gradients from the next layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br>
<code>void summary() = 0;</code><br>
This method prints a summary of the layer.<br>
<code>int getTrainableParams() = 0;</code><br>
This method returns the number of trainable parameters in the layer.<br>
<code>void saveWeights(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the weights will be saved.<br>
<code>append</code>: if true, the weights will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the weights of the layer to a file.<br>
<code>void saveBiases(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the biases will be saved.<br>
<code>append</code>: if true, the biases will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the biases of the layer to a file.<br>
<code>void saveGradients(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the gradients will be saved.<br>
<code>append</code>: if true, the gradients will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the gradients of the layer to a file.<br>
<code>void saveLayer(std::string const &path, bool append = false);</code><br>
<code>path</code>: path to the file where the layer will be saved.<br>
<code>append</code>: if true, the layer will be appended to the file, otherwise the file will be overwritten.<br>
This method saves the layer to a file.<br>

### Dense
This is a fully connected layer. It is derived from the Layer class. It provides the forward and backward methods for a fully connected layer.
#### Constructors
<code>Dense(int numNeurons, int numInputs, int numOutputs);</code><br>
<code>Dense(int numNeurons, int *shape);</code><br>
<code>Dense(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs);</code><br>

#### Methods
<code>void update(float learningRate);</code><br>
<code>learningRate</code>: learning rate for the layer.<br>
This method updates the weights and biases of the layer using the gradients calculated in the backward method.<br>
<code>void forward(Eigen::MatrixXd inputs);</code><br>
<code>inputs</code>: input matrix to the layer.<br>
This method calculates the output of the layer given the input.<br>
<code>void backward(Eigen::MatrixXd gradients);</code><br>
<code>gradients</code>: gradients from the next layer.<br>
This method calculates the gradients of the layer given the gradients from the next layer.<br>
<code>void summary();</code><br>
This method prints a summary of the layer.<br>
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
This method prints a summary of the layer.<br>





