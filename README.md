# Backpropagation

## Rhetorical Design

### Purpose

Deep learning is often presented as a black box: a network "learns" by being shown examples, in a way that resists direct explanation. This project makes the mechanism explicit. Training a neural network is nothing more than applying the chain rule to a computation graph — a fact that becomes concrete when the implementation is built from scratch, with no framework abstractions obscuring it.

### Strategy

Rather than training on a real-world dataset — where no analytic formula for the target exists and correctness can only be judged indirectly — the target is an analytic function: $\arcsin(xy) + \sin(0.2^x + 5y)$. The network's approximation can be compared directly against the ground truth, making the learning signal unambiguous.

## Technical Challenges

### Automatic Differentiation

The computation graph is constructed explicitly as a DAG. The forward pass traverses it recursively, caching intermediate values at each node. The backward pass then traverses in reverse, applying the chain rule at each node using the cached forward values to compute local gradients.

AD is simply an alternative formulation of the chain rule: given a computation graph, the partial derivative of a node $y$ with respect to any parameter $\theta$ used in the definition of another node $x$ can be computed as

$$ \frac{\partial y}{\partial \theta} = \sum_{p\ \in\ paths(y \rightarrow x)} \left[\left( \prod^{\curvearrowright}_{(u, v)\ \in\ p} \frac{\partial u}{\partial v} \right) \cdot \frac{\partial x}{\partial \theta} \right] $$ 

i.e. accumulating Jacobian matrices

$$ \left(\frac{\partial u}{\partial v}\right)_{ij} = \frac{\partial u_i}{\partial v_j} $$

along all possible paths from $y$ to $x$.