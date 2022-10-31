# Interpretability starter

Starter templates for doing interpretability research.

![](https://uploads-ssl.webflow.com/634e78132252d2b0203a9ac8/635662f4e41031331b88038e_2_interpretability_teaser.png)

## Made for the Interpretability Hackathon ([link](https://itch.io/jam/interpretability))

1.  Inspiration
2.  Project starter templates
3.  Related datasets

### Inspiration

Several ideas are represented in **the project of mechanistic interpretability** on aisi.ai. See the ideas here: [aisi.ai/project/mechanistic-interpretability](https://aisafetyideas.com/project/mechanistic-interpretability). A lot of interpretability research is available on [distill.pub](https://distill.pub/), [transformer circuits](https://transformer-circuits.pub/), and [Anthropic's research](https://www.anthropic.com/research).

### See also the tools available on interpretability:

- Redwood Research's interpretability tools: <http://interp-tools.redwoodresearch.org/>
- The activation atlas: <https://distill.pub/2019/activation-atlas/>
- The Tensorflow playground: <https://playground.tensorflow.org/>
- The Neural Network Playground (train simple neural networks in the browser): <https://nnplayground.com/>
- Visualize different neural network architectures: <http://alexlenail.me/NN-SVG/index.html>

### Introductions to interpretability

- A video walkthrough of A Mathematical Framework for Transformer Circuits: <https://www.youtube.com/watch?v=KV5gbOmHbjU>
- Jacob Hilton's deep learning curriculum interpretability: <https://github.com/jacobhilton/deep_learning_curriculum/blob/master/8-Interpretability.md>
- An annotated list of good interpretability papers, along with summaries and takes on what to focus on: <https://www.neelnanda.io/mechanistic-interpretability/favourite-papers>

### Digestible research

- [Distill publication on visualizing neural network weights](https://distill.pub/2020/circuits/visualizing-weights/)
- Andrej Karpathy's ["Understanding what convnets learn"](https://cs231n.github.io/understanding-cnn/)
- [Looking inside a neural net](https://ml4a.github.io/ml4a/looking_inside_neural_nets/)
- 12 toy language models designed to be easier to interpret, in the style of a Mathematical Framework for Transformer Circuits: 1, 2, 3 and 4 layer models, for each size one is attention-only, one has GeLU activations and one has SoLU activations (an activation designed to make the model's neurons more interpretable - <https://transformer-circuits.pub/2022/solu/index.html>) (these aren't well documented yet, but are available in EasyTransformer)

# Starter projects

## 🙋‍♀️ Simple templates & tools

### [Activation Atlas](https://distill.pub/2019/activation-atlas/) [tool]

The [Activation Atlas article](https://distill.pub/2019/activation-atlas/) has a lot of figures where each has a Google Colab associated with them. Click on the "Try in a notebook". An example is [this notebook](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb) that shows a simple activation atlas.

Additionally, they have [this tool](https://distill.pub/2019/activation-atlas/app.html) to explore to which sorts of images neurons activate the most to.

### [EasyTransformer](https://github.com/neelnanda-io/Easy-Transformer/) [code]

A library for mechanistic interpretability called EasyTransformer (still in beta and has bugs, but it's functional enough to be useful!): <https://github.com/neelnanda-io/Easy-Transformer/>

#### [Demo notebook of EasyTransformer](https://colab.research.google.com/drive/1mL4KlTG7Y8DmmyIlE26VjZ0mofdCYVW6)

A demo notebook of how to use Easy Transformer to explore a mysterious phenomena, looking at how language models know to answer "John and Mary went to the shops, then John gave a drink to" with Mary rather than John: <https://colab.research.google.com/drive/1mL4KlTG7Y8DmmyIlE26VjZ0mofdCYVW6>

### [Converting Neural Networks to graphs](https://github.com/apartresearch/interpretability/tree/main/graphs) [code]

This repository can be used to transform a linear neural network into a graph where each neuron is a node and the weights of the directional connections are decided by the actual weights and biases.

You can expand this project by using the graph visualization on the activation for specific inputs and change the conversion from weights into activations or you can try to adapt it to convolutional neural networks. **Check out the code below**.

| File                                                                                                    | Description                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`train.py`](https://github.com/apartresearch/interpretability/blob/main/graphs/model_init/train.py)    | Creates [`model.pt`](https://github.com/apartresearch/interpretability/blob/main/graphs/model_init/model.pt) with a 500 hidden layer linear MNIST classifier. |
| [`to_graph.py`](https://github.com/apartresearch/interpretability/blob/main/graphs/to_graph.py)         | Generates a graph from [`model.pt`](https://github.com/apartresearch/interpretability/blob/main/graphs/model_init/model.pt).                                  |
| [`vertices.csv`](https://github.com/apartresearch/interpretability/blob/main/graphs/data/vertices.csv)  | Each neuron in the MNIST linear classifier with its bias and layer.                                                                                           |
| [`edges.csv`](https://github.com/apartresearch/interpretability/blob/main/graphs/data/edges.csv)        | Each connection in the neural network: `from_id, to_id, weight`.                                                                                              |
| [`network_eda.Rmd`](https://github.com/apartresearch/interpretability/blob/main/graphs/network_eda.Rmd) | The R script for initial EDA and visualization of the network.                                                                                                |

## 👩‍🔬 Advanced templates and tools

### [Redwood Research's interpretability on Transformers](http://interp-tools.redwoodresearch.org/) [tool]

Redwood Research has created a [wonderful tool](http://interp-tools.redwoodresearch.org/) that can be used to do research into how language models understand text. The ["How to use" document](https://docs.google.com/document/d/1ECwTXrgTqgiMN24L7IantJTaFpyJM2LxXXGq50meFKc/edit) and [their instruction videos](https://www.youtube.com/channel/UCwvzObS_ayucGlYIJCyagdA) are very good introductions and we recommend reading/watching them since the interface can be a bit daunting otherwise.

Watch this video as an intro:

[![Understanding interp-tools by Redwood Research](https://img.youtube.com/vi/zH8YBqdIB-w/0.jpg)](https://www.youtube.com/watch?v=zH8YBqdIB-w)
