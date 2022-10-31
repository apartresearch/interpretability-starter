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

#### [Christoph Molnar's book about interpretability](https://christophm.github.io/interpretable-ml-book)

An amazing introduction to traditional interpretability research. We recommend reading the [taxonomy of interpretability](https://christophm.github.io/interpretable-ml-book/taxonomy-of-interpretability-methods.html) and about the specific methods of [PDP](https://christophm.github.io/interpretable-ml-book/pdp.html), [ALE](https://christophm.github.io/interpretable-ml-book/ale.html), [ICE](https://christophm.github.io/interpretable-ml-book/ice.html), [LIME](https://christophm.github.io/interpretable-ml-book/lime.html), [Shapley values](https://christophm.github.io/interpretable-ml-book/shapley.html), and [SHAP ](https://christophm.github.io/interpretable-ml-book/shap.html). Also read the chapter on [neural network interpretation](https://christophm.github.io/interpretable-ml-book/neural-networks.html) such as [saliency maps](https://christophm.github.io/interpretable-ml-book/pixel-attribution.html) and [adversarial examples](https://christophm.github.io/interpretable-ml-book/adversarial.html). He has also published a ["Common pitfalls of interpretability" post](https://mindfulmodeler.substack.com/p/8-pitfalls-to-avoid-when-interpreting) that is recommended reading.

[<img src="https://christophm.github.io/interpretable-ml-book/images/cutout.png" width="200px">](https://christophm.github.io/interpretable-ml-book/)

#### Other resources

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

### Reviewing explainability tools

There are a few tools that use interpretability to create understandable explanations of why they give the output they give. [This notebook](https://colab.research.google.com/drive/1_OfQuqEZsd6fC_cLGu43S58eUo_bIAmB?usp=sharing) provides a small intro to the most relevant libraries:

- [ELI5](https://pypi.org/project/eli5/): ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions. It implements a few different analysis frameworks that work with a lot of different ML libraries. It is the most complete tool for explainability.

![Explanations of output](https://warehouse-camo.ingress.cmh1.psfhosted.org/657108d350a6db09fede2cf5d02ff2c6eb2ac6d7/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f5465616d48472d4d656d65782f656c69352f6d61737465722f646f63732f736f757263652f7374617469632f776f72642d686967686c696768742e706e67)

![Image explanations of output](https://warehouse-camo.ingress.cmh1.psfhosted.org/3223146dc3811a97ebe287fda4e910ffb61ff263/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f5465616d48472d4d656d65782f656c69352f6d61737465722f646f63732f736f757263652f7374617469632f6772616463616d2d636174646f672e706e67)

- [LIME](https://christophm.github.io/interpretable-ml-book/lime.html): Local Interpretable Model-agnostic Explanations. The [TextExplainer library](https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html) does a good job of using LIME on language models. Check out Christoph Molnar's introduction [here](https://christophm.github.io/interpretable-ml-book/lime.html).
- SHAP: SHapley Additive exPlanations
- MLXTEND: Machine Learning Extensions

## 👩‍🔬 Advanced templates and tools

### [Redwood Research's interpretability on Transformers](http://interp-tools.redwoodresearch.org/) [tool]

Redwood Research has created a [wonderful tool](http://interp-tools.redwoodresearch.org/) that can be used to do research into how language models understand text. The ["How to use" document](https://docs.google.com/document/d/1ECwTXrgTqgiMN24L7IantJTaFpyJM2LxXXGq50meFKc/edit) and [their instruction videos](https://www.youtube.com/channel/UCwvzObS_ayucGlYIJCyagdA) are very good introductions and we recommend reading/watching them since the interface can be a bit daunting otherwise.

Watch this video as an intro:

[![Understanding interp-tools by Redwood Research](https://img.youtube.com/vi/zH8YBqdIB-w/0.jpg)](https://www.youtube.com/watch?v=zH8YBqdIB-w)
