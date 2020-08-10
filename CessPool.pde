class Connection
{
  double weight, deltaweight;
  Connection(double value)
  {
    weight = 0.0;
    weight = value;
    deltaweight = 0.0;
    weight = random(0.0, 1.0);
  }

  double getDW()
  {
    return deltaweight;
  }
  void setDW(double val)
  {
    deltaweight = val;
  }

  double getWeight()
  {
    return weight;
  }
  void setWeight(double value)
  {
    weight = value;
  }
}

/////////////////////////////////////////////
////////////////////////////////////////////


class Neuron
{
  int m_myIndex = 0;
  double eta = 0.15;
  double alpha = 0.5;
  double m_outputVal = 0.0;
  double m_gradient = 0.2;
  ArrayList<Connection> connections = new ArrayList<Connection>();
  


  Neuron(int numOutputs, int myIndex, double weight_value)
  {
    for (int c = 0; c < numOutputs; ++c) 
    {
      Connection newConnection = new Connection(weight_value);
      newConnection.weight = randomWeight();
   //   print(newConnection.weight);
      connections.add(newConnection);
    }
    m_myIndex = myIndex;
  }

  void feedForward(ArrayList<Neuron> prevLayer)
  {
    double sum = 0.0;

    // We sum the previous layer's output
    // Include the bias node from the previous layer.

    for (int n = 0; n < prevLayer.size(); ++n) 
    {
      sum += prevLayer.get(n).getOutputVal() *
        prevLayer.get(n).connections.get(m_myIndex).weight;
    }

    m_outputVal = transferFunction(sum);
  }

  double getOutputVal()
  {
    return m_outputVal;
  }

  void setOutputVal(double n)
  {
    m_outputVal = n;
  }

  void calcHiddenGradients(ArrayList<Neuron> nextLayer)
  {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
  }

  void calcOutputGradients(double targetVal)
  {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
  }
  
  ArrayList<Double> getWeights()
  {
    ArrayList<Double> weights = new ArrayList<Double>();
    for(int i = 0;i<connections.size()-1;i++)
    {
      weights.add(connections.get(i).getWeight());
    }    
    return weights;
  }

  void updateInputWeights(ArrayList<Neuron> prevLayer)
  {
    for (int n = 0; n < prevLayer.size(); ++n) 
    {
      Neuron neuron = prevLayer.get(n);
      double oldDeltaWeight = neuron.connections.get(m_myIndex).deltaweight;

      double newDeltaWeight =
        // Individual input is magnified by the gradient and train rate:
        eta
        * neuron.getOutputVal()
        * m_gradient
        // Also adding momentum = a fraction of the previous delta weight;
        + alpha
        * oldDeltaWeight;

      neuron.connections.get(m_myIndex).deltaweight = newDeltaWeight;
      neuron.connections.get(m_myIndex).weight += newDeltaWeight;
    }
  }

  double randomWeight()
  {
    double rand_max = 2147483647.0;
    return (random(0.0, 1.0) / rand_max);
  }

  double sumDOW(ArrayList<Neuron> nextLayer)
  {
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (int n = 0; n < nextLayer.size() - 1; ++n) 
    {
      sum += connections.get(n).weight * nextLayer.get(n).m_gradient;
    }

    return sum;
  }

  double transferFunctionDerivative(double x)
  {
    return 1 - x*x;
  }
  double transferFunction(double x)
  {
    return Math.tanh(x);
  }
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////
class Layer
{
  int neuronNum = 0;
  ArrayList<Neuron> neurons = new ArrayList<Neuron>();
  int numOutputs;

  Layer(ArrayList<Integer> topology, int layerNum)
  {
    while (neuronNum < topology.get(layerNum))
    {
      numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);
      neurons.add(new Neuron(numOutputs, neuronNum, random(0.0, 1.0)));
      neuronNum = neuronNum + 1;
    }
  }

  void add(Neuron neuron)
  {
    neurons.add(neuron);
  }

  ArrayList<Neuron> getNeurons()
  {
    return neurons;
  }
}


//Network
/////////////////////////////////////////////////////


class Network
{
  int numLayers;
  double m_error = 0.0;
  double m_recentAverageSmoothingFactor = 100.0;
  double m_recentAverageError = 0.0;
  double delta = 0.0;

  ArrayList<Layer> m_layers = new ArrayList<Layer>();
  Network(ArrayList<Integer> topology)
  {
    numLayers= topology.size();
    for (int layerNum = 0; layerNum < numLayers; ++layerNum) 
    {
      m_layers.add(new Layer(topology, layerNum));
      int numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);






      // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
      m_layers.get(m_layers.size()-1).getNeurons().get(m_layers.get(m_layers.size()-1).getNeurons().size()-1).setOutputVal(1.0);
    }
  }

  void backPropagate(ArrayList<Double> targetVals)
  {
    // Calculate overall net error (RMS of output neuron errors)
    Layer outputLayer = m_layers.get(m_layers.size()-1);
    m_error = 0.0;
    for (int n = 0; n < outputLayer.getNeurons().size() - 1; ++n) 
    {
      double delta = targetVals.get(n) - outputLayer.getNeurons().get(n).getOutputVal();
      m_error += delta * delta;
    }
    m_error /= outputLayer.getNeurons().size() - 1; // get average error squared
    m_error = Math.sqrt(m_error); // RMS
    // Implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients

    for (int n = 0; n < outputLayer.getNeurons().size(); ++n) 
    {
      outputLayer.getNeurons().get(n).calcOutputGradients(targetVals.get(n));
    }

    // Calculate hidden layer gradients

    for (int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) 
    {
      Layer hiddenLayer = m_layers.get(layerNum);
      Layer nextLayer = m_layers.get(layerNum + 1);

      for (int n = 0; n < hiddenLayer.getNeurons().size(); ++n) 
      {
        hiddenLayer.getNeurons().get(n).calcHiddenGradients(nextLayer.getNeurons());
      }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (int layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) 
    {
      Layer layer = m_layers.get(layerNum);
      Layer prevLayer = m_layers.get(layerNum - 1);

      for (int n = 0; n < layer.getNeurons().size(); ++n) 
      {
        layer.getNeurons().get(n).updateInputWeights(prevLayer.getNeurons());
      }
    }
  }

  void feedForward(ArrayList<Double> inputVals)
  {
    assert(inputVals.size() == m_layers.get(0).getNeurons().size());

    // Assign (latch) the input values into the input neurons
    for (int i = 0; i < inputVals.size(); ++i) 
    {
      m_layers.get(0).getNeurons().get(i).setOutputVal(inputVals.get(i));
    }

    // forward propagate
    for (int layerNum = 1; layerNum < m_layers.size(); ++layerNum) 
    {
      Layer prevLayer = m_layers.get(layerNum - 1);
      for (int n = 0; n < m_layers.get(layerNum).getNeurons().size(); ++n) 
      {
        m_layers.get(layerNum).getNeurons().get(n).feedForward(prevLayer.getNeurons());
      }
    }
  }
  void getResults(ArrayList resultVals)
  {
 //   print(resultVals);
    resultVals.clear();

    for (int n = 0; n < m_layers.get(m_layers.size()-1).getNeurons().size(); ++n) 
    {
      resultVals.add(m_layers.get(m_layers.size()-1).getNeurons().get(n).getOutputVal());
    }
  }

  ArrayList GetWeights()
  {
    //this will hold the weights
    ArrayList<Double>weights = new ArrayList<Double>();

    //for each layer
    for (int i = 0; i<m_layers.size(); ++i)
    {
      //for each neuron
      for (int j = 0; j<m_layers.get(i).getNeurons().size(); ++j)
      {
        //for each weight
        for (int k = 0; k<m_layers.get(i).getNeurons().get(j).connections.size(); ++k)
        {
          weights.add(m_layers.get(i).getNeurons().get(j).connections.get(k).weight);
        }
      }
    }
    return weights;
  }

  void PutWeights(ArrayList<Double> weights)
  {
    int cWeight = 0;
    //for each layer
    for (int i = 0; i<m_layers.size(); ++i)
    {
      //for each neuron
      for (int j = 0; j<m_layers.get(i).getNeurons().size(); ++j)
      {
        //for each weight
        for (int k = 0; k<m_layers.get(i).getNeurons().get(j).connections.size(); ++k)
        {
          m_layers.get(i).getNeurons().get(j).connections.get(k).weight = weights.get(cWeight++);
        }
      }
    }
    return;
  }
  
  ArrayList getLayers()
  {
    ArrayList<Layer> all_layers = new ArrayList<Layer>();
    for( int i = 0;i<m_layers.size()-1;i++)
    {
      all_layers.add(m_layers.get(i));
    }
    return all_layers;
  }
}


class Computer
{
  Network thisNetwork;
  ArrayList<Layer> layers = new ArrayList<Layer>();
  ArrayList<Double> resultVals = new ArrayList<Double>();
  
  ArrayList<Double> weights = new ArrayList<Double>();
  Computer(ArrayList<Integer> topology)
  {
    thisNetwork = new Network(topology);
  }

  void BackPropagate(ArrayList<Double> targetVals)
  {
    thisNetwork.backPropagate(targetVals);
  }

  Network getNetwork()
  {
    return thisNetwork;
  }


  ArrayList<Double> getWeights()
  {
    Network network = getNetwork();
    layers = network.getLayers();
  //  weights = [];
    for (int i=0; i<layers.size(); i++)
    {
      for (int j = 0; j<layers.get(i).getNeurons().size(); j++)
      {
        for (int k = 0; k<layers.get(i).getNeurons().get(j).getWeights().size(); k++)
        {
          weights.add(layers.get(i).getNeurons().get(j).getWeights().get(k));
        }
      }
    }
    return weights;
  }


  void feedforward(ArrayList<Double> inputs)
  {
    thisNetwork.feedForward(inputs);
  }

  ArrayList<Double> GetResult()
  {
    return resultVals;
  }

  void SetWeights(ArrayList<Double> weights)
  {
    thisNetwork.PutWeights(weights);
  }

  void train(int num_iterations, ArrayList<Double> trainArray, ArrayList<Double> testArray)
  {
    for (int i = 0; i<num_iterations; i++)
    {
      feedforward(trainArray);
      BackPropagate(testArray);
      resultVals = new ArrayList<Double>();
      resultVals.clear();
      getNetwork().getResults(resultVals);
    }
  }
  
  void train_draw(int num_iterations, ArrayList<Double> trainArray, ArrayList<Double> testArray)
  {
      feedforward(trainArray);
      BackPropagate(testArray);
      resultVals = new ArrayList<Double>();
      resultVals.clear();
      getNetwork().getResults(resultVals);
  }
}


ArrayList<Integer> topology = new ArrayList<Integer>();
ArrayList<Double> trainArray = new ArrayList<Double>();
ArrayList<Double> testArray = new ArrayList<Double>();
ArrayList<Double> weights = new ArrayList<Double>();
Computer newComputer;

void setup()
{
size(1000,1000);

topology.add(30);
topology.add(30);
topology.add(3);
newComputer = new Computer(topology);
//testingWeights = newComputer.GetWeights();

int n=10;
while(n>0)
{
trainArray.add(0.0d);
trainArray.add(1.0d);
trainArray.add(0.0d);

testArray.add(1.0d);
testArray.add(1.0d);
testArray.add(0.0d);
n--;
}


weights = newComputer.getWeights();



}


void draw()
{
  
  background(255,0,0);
  
  
  newComputer.train(1, trainArray, testArray);
  for(int i=0;i<newComputer.GetResult().size();i++)
  {
   //   print(newComputer.GetResult().get(i));
  }
  print(newComputer.GetResult().get(2) + "\n");
  
  float f1 = newComputer.GetResult().get(0).floatValue();
  float f2 = newComputer.GetResult().get(1).floatValue();
  float f3 = newComputer.GetResult().get(2).floatValue();
  
  
  ellipse(400,500-f1*200,10,10);
  ellipse(500,500-f2*200,10,10);
  ellipse(600,500-f3*200,10,10);
}
