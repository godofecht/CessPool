# CessPool

A machine-learning library for Processing.

## What this is

CessPool is a neural network written from scratch in one [Processing](https://processing.org/) sketch, with no libraries. Processing is a beginner-friendly language and editor, built on Java, for making pictures and animations with code. A neural network is a set of connected artificial neurons that learns to turn inputs into outputs.

When you run it, a red window opens with three white dots. Each dot is one of the network's three outputs: the higher the dot, the larger the output. Every frame the network takes one training step. Because training and drawing happen in the same loop, any change in the network shows up on screen straight away.

The network classes (`Connection`, `Neuron`, `Layer`, `Network` and `Computer`) are written so they can be copied into other sketches, which is where the "library" in the description comes from.

## What you can learn from it

- Computer Science: what a neuron, a weight and a layer are, written out in plain Java.
- Computer Science: how backpropagation trains a network by nudging each weight to reduce the error.
- Computer Science: what learning rate and momentum do.
- Maths: the tanh function and its derivative, used to squash and train each neuron.
- Creative coding: how Processing's `setup()` and `draw()` functions run a program frame by frame.

## Run it

### On your own computer

**You will need:** [Processing](https://processing.org/download) 4 (free, for Windows, macOS and Linux). These steps have not been tested for this guide.

1. Download the code with [Git](https://git-scm.com/downloads):

   ```bash
   git clone https://github.com/godofecht/CessPool.git
   ```

   Or choose **Code**, then **Download ZIP** on the repo page and unzip it. A Processing sketch must sit in a folder with the same name as its `.pde` file. If the folder is called `CessPool-master`, rename it to `CessPool`, or let Processing move it when it asks.

2. Open `CessPool/CessPool.pde` in Processing.
3. Press the **Run** button (the triangle in the top-left).

A 1000 by 1000 red window opens with three white dots in a row, 200 pixels above the middle of the window. The targets are 1, 1 and 0, so the right dot ought to drop to the middle line. With the current settings it stays where it is. The [How it works](#how-it-works) section explains why, and idea 1 below fixes it. There are no controls. Press **Stop** or close the window to end.

## How it works

A neuron adds up the outputs of the neurons in the layer before it, each multiplied by a weight (a number that sets how much that input matters). It passes the total through tanh, which squashes any number into the range -1 to 1.

The sketch builds a network with three layers: 30 input neurons, 30 hidden neurons and 3 output neurons. Each frame it does two things:

1. **Feed forward.** It puts the same 30 numbers (the pattern 0, 1, 0 repeated ten times) into the input layer and works out every neuron's output, layer by layer.
2. **Backpropagate.** It compares the three outputs with the targets 1, 1 and 0. It works out how much each weight added to the error, then moves each weight a small step to reduce it.

The size of each step is the learning rate (`eta`, 0.15). Momentum (`alpha`, 0.5) adds half of the previous step to the new one, which helps the weights keep moving in a steady direction. That is why each `Connection` stores its last change, `deltaWeight`, as well as its weight.

`draw()` then places each dot at height `500 - output * 200`, so an output of 1 sits 200 pixels above the middle line and an output of 0 sits on it.

Why the right dot does not move: every starting weight is a random number between 0 and 1, so every weight is positive. Each neuron's total is large (around 5 in the hidden layer and 15 in the output layer), and tanh of a large number is almost exactly 1. This is called saturation. The slope of tanh there is almost 0, and backpropagation multiplies by that slope, so the weight changes are almost 0 and the network stops learning. A Python copy of the training loop, run for 600 frames, kept all three outputs at 1.0.

| File or folder | What is in it |
| --- | --- |
| `CessPool.pde` | the whole program: the network classes, `setup()` and `draw()` |
| `code/sketch.properties` | a Processing settings file that sets Java mode |

## Ideas to try

1. In `CessPool.pde`, fix the saturation by making the starting weights run from -1 to 1: change `random(0.0, 1.0)` in `Connection` to `random(-1.0, 1.0)` and run it again. In the Python copy this change let the right dot fall to 0 within a few hundred frames for most random starts, though some starts still got stuck.
2. Change the targets added to `testArray` in `setup()` (for example 0, 1, 1) and watch the dots move to new places.
3. In the `Neuron` class, change `eta` to 0.01 or 0.5. How does the speed of learning change? What happens to the dots when it is too large?
4. Set `alpha` to 0 to turn momentum off and compare.
5. In `draw()`, use `newComputer.thisNetwork.getLayers()` to draw every neuron as a circle and every weight as a line whose thickness shows its size.
6. Teach the network XOR (output 1 when exactly one of two inputs is 1). Make the topology 2, 4, 2 and read only the first output, because the code treats the last neuron of each layer specially (see Technical details). In `draw()`, cycle through the four input pairs and their targets, and use `Computer.train` with one pair per frame.
7. Add a proper bias neuron (a neuron whose output is always 1). At present the `Network` constructor sets the last neuron of each layer to 1, but `feedForward` overwrites it. Change `Layer` and `Network` so the bias neuron keeps its value.

## Technical details

- `Connection` holds a weight and its last delta. `Neuron` owns its outgoing connections and its gradient. A `Layer` is a list of neurons, and `Network` does the feed-forward and backward passes over them. `Computer` wraps a `Network` with a `train(numIterations, inputs, targets)` method.
- Starting weights are random numbers between 0 and 1.
- The output error is the root mean square of the output errors, and the network keeps a smoothed `m_recentAverageError` with a smoothing factor of 100.
- Weights can be read and written as one flat list with `Network.getWeights()` and `Network.putWeights()`.
- `trainArray` and `testArray` each hold 30 values. Only the first three values of `testArray` are used as targets, because the output layer has three neurons.
- The error measure and the hidden-layer gradients skip the last neuron of the layer they read from, which follows the C++ design this was ported from, where the last neuron was a bias neuron.

## Project status

The sketch is a small, finished experiment. Known problems:

- With weights that start between 0 and 1, the network saturates and does not learn (see How it works and idea 1).
- There is no working bias neuron (see idea 7).
- The repo has no screenshot of the running sketch.
- `sketch.properties` sits in a folder called `code`. Processing uses the `code` folder for extra Java libraries, so this file probably has no effect there.

## Licence

You can use, study, share and change this code. If you share a changed version, you must share its source code under the same licence. See [LICENSE](LICENSE) (GNU GPL v3).

## Credits

- Written by Abhishek Shivakumar.
- Built with [Processing](https://processing.org/).
