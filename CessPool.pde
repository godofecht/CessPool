class Connection {
    double weight, deltaWeight;

    Connection(double value) {
        weight = 0.0;
        deltaWeight = 0.0;
        weight = random(0.0, 1.0);  // Assign a random weight
    }

    double getDeltaWeight() {
        return deltaWeight;
    }

    void setDeltaWeight(double val) {
        deltaWeight = val;
    }

    double getWeight() {
        return weight;
    }

    void setWeight(double value) {
        weight = value;
    }
}

class Neuron {
    int m_myIndex = 0;
    double eta = 0.15;  // Learning rate
    double alpha = 0.5; // Momentum
    double m_outputVal = 0.0;
    double m_gradient = 0.2;
    ArrayList<Connection> connections = new ArrayList<>();

    Neuron(int numOutputs, int myIndex) {
        for (int c = 0; c < numOutputs; ++c) {
            Connection newConnection = new Connection(randomWeight());
            connections.add(newConnection);
        }
        m_myIndex = myIndex;
    }

    void feedForward(ArrayList<Neuron> prevLayer) {
        double sum = 0.0;
        for (int n = 0; n < prevLayer.size(); ++n) {
            sum += prevLayer.get(n).getOutputVal() *
                    prevLayer.get(n).connections.get(m_myIndex).getWeight();
        }
        m_outputVal = transferFunction(sum);
    }

    double getOutputVal() {
        return m_outputVal;
    }

    void setOutputVal(double val) {
        m_outputVal = val;
    }

    void calcHiddenGradients(ArrayList<Neuron> nextLayer) {
        double dow = sumDOW(nextLayer);
        m_gradient = dow * transferFunctionDerivative(m_outputVal);
    }

    void calcOutputGradients(double targetVal) {
        double delta = targetVal - m_outputVal;
        m_gradient = delta * transferFunctionDerivative(m_outputVal);
    }

    ArrayList<Double> getWeights() {
        ArrayList<Double> weights = new ArrayList<>();
        for (Connection connection : connections) {
            weights.add(connection.getWeight());
        }
        return weights;
    }

    void updateInputWeights(ArrayList<Neuron> prevLayer) {
        for (Neuron neuron : prevLayer) {
            double oldDeltaWeight = neuron.connections.get(m_myIndex).getDeltaWeight();
            double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
            neuron.connections.get(m_myIndex).setDeltaWeight(newDeltaWeight);
            neuron.connections.get(m_myIndex).setWeight(neuron.connections.get(m_myIndex).getWeight() + newDeltaWeight);
        }
    }

    double randomWeight() {
        return Math.random();
    }

    double sumDOW(ArrayList<Neuron> nextLayer) {
        double sum = 0.0;
        for (int n = 0; n < nextLayer.size() - 1; ++n) {
            sum += connections.get(n).getWeight() * nextLayer.get(n).m_gradient;
        }
        return sum;
    }

    double transferFunction(double x) {
        return Math.tanh(x);
    }

    double transferFunctionDerivative(double x) {
        return 1.0 - x * x;
    }
}

class Layer {
    int neuronNum = 0;
    ArrayList<Neuron> neurons = new ArrayList<>();

    Layer(ArrayList<Integer> topology, int layerNum) {
        int numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);
        while (neuronNum < topology.get(layerNum)) {
            neurons.add(new Neuron(numOutputs, neuronNum));
            neuronNum++;
        }
    }

    ArrayList<Neuron> getNeurons() {
        return neurons;
    }
}

class Network {
    int numLayers;
    double m_error = 0.0;
    double m_recentAverageError = 0.0;
    double m_recentAverageSmoothingFactor = 100.0;
    ArrayList<Layer> m_layers = new ArrayList<>();

    Network(ArrayList<Integer> topology) {
        numLayers = topology.size();
        for (int layerNum = 0; layerNum < numLayers; ++layerNum) {
            m_layers.add(new Layer(topology, layerNum));
            m_layers.get(layerNum).getNeurons().get(m_layers.get(layerNum).getNeurons().size() - 1).setOutputVal(1.0);
        }
    }

    void backPropagate(ArrayList<Double> targetVals) {
        Layer outputLayer = m_layers.get(m_layers.size() - 1);
        m_error = 0.0;

        for (int n = 0; n < outputLayer.getNeurons().size() - 1; ++n) {
            double delta = targetVals.get(n) - outputLayer.getNeurons().get(n).getOutputVal();
            m_error += delta * delta;
        }
        m_error /= outputLayer.getNeurons().size() - 1;
        m_error = Math.sqrt(m_error);

        m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
                (m_recentAverageSmoothingFactor + 1.0);

        for (int n = 0; n < outputLayer.getNeurons().size(); ++n) {
            outputLayer.getNeurons().get(n).calcOutputGradients(targetVals.get(n));
        }

        for (int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
            Layer hiddenLayer = m_layers.get(layerNum);
            Layer nextLayer = m_layers.get(layerNum + 1);
            for (Neuron neuron : hiddenLayer.getNeurons()) {
                neuron.calcHiddenGradients(nextLayer.getNeurons());
            }
        }

        for (int layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
            Layer layer = m_layers.get(layerNum);
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (Neuron neuron : layer.getNeurons()) {
                neuron.updateInputWeights(prevLayer.getNeurons());
            }
        }
    }

    void feedForward(ArrayList<Double> inputVals) {
        assert (inputVals.size() == m_layers.get(0).getNeurons().size());

        for (int i = 0; i < inputVals.size(); ++i) {
            m_layers.get(0).getNeurons().get(i).setOutputVal(inputVals.get(i));
        }

        for (int layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
            Layer prevLayer = m_layers.get(layerNum - 1);
            for (Neuron neuron : m_layers.get(layerNum).getNeurons()) {
                neuron.feedForward(prevLayer.getNeurons());
            }
        }
    }

    void getResults(ArrayList<Double> resultVals) {
        resultVals.clear();
        for (Neuron neuron : m_layers.get(m_layers.size() - 1).getNeurons()) {
            resultVals.add(neuron.getOutputVal());
        }
    }

    ArrayList<Double> getWeights() {
        ArrayList<Double> weights = new ArrayList<>();
        for (Layer layer : m_layers) {
            for (Neuron neuron : layer.getNeurons()) {
                weights.addAll(neuron.getWeights());
            }
        }
        return weights;
    }

    void putWeights(ArrayList<Double> weights) {
        int cWeight = 0;
        for (Layer layer : m_layers) {
            for (Neuron neuron : layer.getNeurons()) {
                for (int k = 0; k < neuron.connections.size(); ++k) {
                    neuron.connections.get(k).setWeight(weights.get(cWeight++));
                }
            }
        }
    }

    ArrayList<Layer> getLayers() {
        return m_layers;
    }
}

class Computer {
    Network thisNetwork;
    ArrayList<Double> resultVals = new ArrayList<>();

    Computer(ArrayList<Integer> topology) {
        thisNetwork = new Network(topology);
    }

    void backPropagate(ArrayList<Double> targetVals) {
        thisNetwork.backPropagate(targetVals);
    }

    void feedForward(ArrayList<Double> inputs) {
        thisNetwork.feedForward(inputs);
    }

    ArrayList<Double> getWeights() {
        return thisNetwork.getWeights();
    }

    ArrayList<Double> getResult() {
        return resultVals;
    }

    void setWeights(ArrayList<Double> weights) {
        thisNetwork.putWeights(weights);
    }

    void train(int numIterations, ArrayList<Double> trainArray, ArrayList<Double> testArray) {
        for (int i = 0; i < numIterations; i++) {
            feedForward(trainArray);
            backPropagate(testArray);
            resultVals.clear();
            thisNetwork.getResults(resultVals);
        }
    }
}

ArrayList<Integer> topology = new ArrayList<>();
ArrayList<Double> trainArray = new ArrayList<>();
ArrayList<Double> testArray = new ArrayList<>();
Computer newComputer;

void setup() {
    size(1000, 1000);

    topology.add(30);
    topology.add(30);
    topology.add(3);
    newComputer = new Computer(topology);

    for (int n = 0; n < 10; n++) {
        trainArray.add(0.0);
        trainArray.add(1.0);
        trainArray.add(0.0);

        testArray.add(1.0);
        testArray.add(1.0);
        testArray.add(0.0);
    }
}

void draw() {
    background(255, 0, 0);

    newComputer.train(1, trainArray, testArray);
    float f1 = newComputer.getResult().get(0).floatValue();
    float f2 = newComputer.getResult().get(1).floatValue();
    float f3 = newComputer.getResult().get(2).floatValue();

    ellipse(400, 500 - f1 * 200, 10, 10);
    ellipse(500, 500 - f2 * 200, 10, 10);
    ellipse(600, 500 - f3 * 200, 10, 10);
}
