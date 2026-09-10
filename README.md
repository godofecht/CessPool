# CessPool

A backpropagation network written from scratch in a single Processing sketch.

No library. `CessPool.pde` contains the whole thing: `Connection` holds a weight
and its last delta, `Neuron` owns its outgoing connections and its gradient,
layers are lists of neurons, and the net does the feed-forward and backward
passes over them.

Learning rate is 0.15 and momentum is 0.5, both set on the neuron. The momentum
term is why `Connection` keeps `deltaWeight` around rather than just the weight.

Being a Processing sketch means the training loop and the drawing loop are the
same loop, so the network can be watched while it learns rather than inspected
afterwards.

## Run

Open `CessPool.pde` in the Processing IDE and press run.
