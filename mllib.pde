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
  ArrayList<Connection> m_outputWeights;


  Neuron(int numOutputs, int myIndex, double weight_value)
  {
    for (int c = 0; c < numOutputs; ++c) 
    {
      Connection newConnection = new Connection(weight_value);
      newConnection.weight = randomWeight();
      m_outputWeights.add(newConnection);
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
        prevLayer.get(n).m_outputWeights.get(m_myIndex).weight;
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

  void updateInputWeights(ArrayList<Neuron> prevLayer)
  {
    for (int n = 0; n < prevLayer.size(); ++n) 
    {
      Neuron neuron = prevLayer.get(n);
      double oldDeltaWeight = neuron.m_outputWeights.get(m_myIndex).deltaweight;

      double newDeltaWeight =
        // Individual input is magnified by the gradient and train rate:
        eta
        * neuron.getOutputVal()
        * m_gradient
        // Also adding momentum = a fraction of the previous delta weight;
        + alpha
        * oldDeltaWeight;

      neuron.m_outputWeights.get(m_myIndex).deltaweight = newDeltaWeight;
      neuron.m_outputWeights.get(m_myIndex).weight += newDeltaWeight;
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
      sum += m_outputWeights.get(n).weight * nextLayer.get(n).m_gradient;
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
  
  Layer(ArrayList<Integer> topology,int layerNum)
  {
    while(neuronNum < topology.get(layerNum))
    {
      numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);
      neurons.add(new Neuron(numOutputs,neuronNum,random(0.0,1.0)));
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
      m_layers.add(new Layer(topology,layerNum));
      int numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);

      // We have a new layer, now fill it with neurons, and a bias neuron in each layer.
  //    for (int neuronNum = 0; neuronNum <= topology.get(layerNum); ++neuronNum) 
  //    {
     //   m_layers.get(m_layers.size()).add(new Neuron(numOutputs, neuronNum));
  //   m_layers.add(new Layer(topology,layerNum)
        //   cout << "Made a Neuron!" << endl;
 //     }

      // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
      m_layers.get(m_layers.size()).getNeurons().get(m_layers.get(m_layers.size()).getNeurons().size()).setOutputVal(1.0);
    }
  }

  void backPropagate(ArrayList<Double> targetVals)
  {
    // Calculate overall net error (RMS of output neuron errors)
    Layer outputLayer = m_layers.get(m_layers.size());
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

    for (int n = 0; n < outputLayer.getNeurons().size() - 1; ++n) 
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

      for (int n = 0; n < layer.getNeurons().size() - 1; ++n) 
      {
        layer.getNeurons().get(n).updateInputWeights(prevLayer.getNeurons());
      }
    }
  }

  void feedForward(ArrayList<Double> inputVals)
  {
    assert(inputVals.size() == m_layers.get(0).getNeurons().size() - 1);

    // Assign (latch) the input values into the input neurons
    for (int i = 0; i < inputVals.size(); ++i) 
    {
      m_layers.get(0).getNeurons().get(i).setOutputVal(inputVals.get(i));
    }

    // forward propagate
    for (int layerNum = 1; layerNum < m_layers.size(); ++layerNum) 
    {
      Layer prevLayer = m_layers.get(layerNum - 1);
      for (int n = 0; n < m_layers.get(layerNum).getNeurons().size() - 1; ++n) 
      {
        m_layers.get(layerNum).getNeurons().get(n).feedForward(prevLayer.getNeurons());
      }
    }
  }
  void getResults(ArrayList resultVals)
  {
    resultVals.clear();

    for (int n = 0; n < m_layers.get(m_layers.size()).getNeurons().size() - 1; ++n) 
    {
      resultVals.add(m_layers.get(m_layers.size()).getNeurons().get(n).getOutputVal());
    }
  }

  ArrayList GetWeights()
  {
    //this will hold the weights
    ArrayList<Double>weights = new ArrayList<Double>();

    //for each layer
    for (int i = 0; i<m_layers.size()-1; ++i)
    {
      //for each neuron
      for (int j = 0; j<m_layers.get(i).getNeurons().size(); ++j)
      {
        //for each weight
        for (int k = 0; k<m_layers.get(i).getNeurons().get(j).m_outputWeights.size(); ++k)
        {
          weights.add(m_layers.get(i).getNeurons().get(j).m_outputWeights.get(k).weight);
        }
      }
    }
    return weights;
  }

  void PutWeights(ArrayList<Double> weights)
  {
    int cWeight = 0;
    //for each layer
    for (int i = 0; i<m_layers.size()-1; ++i)
    {
      //for each neuron
      for (int j = 0; j<m_layers.get(i).getNeurons().size(); ++j)
      {
        //for each weight
        for (int k = 0; k<m_layers.get(i).getNeurons().get(j).m_outputWeights.size(); ++k)
        {
          m_layers.get(i).getNeurons().get(j).m_outputWeights.get(k).weight = weights.get(cWeight++);
        }
      }
    }
    return;
  }
}
