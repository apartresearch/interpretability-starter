# Interpretability starter

Interpretability research is an exciting and growing field of machine learning. If we are able to understand what happens within neural networks in diverse domains, we can see **why** the network gives us specific outputs and detect deception, understand choices, and change how they work.

![](https://uploads-ssl.webflow.com/634e78132252d2b0203a9ac8/635662f4e41031331b88038e_2_interpretability_teaser.png)

This list of resources was made for the Interpretability Hackathon ([link](https://itch.io/jam/interpretability)) and contains an array of useful starter templates, tools to investigate model activations, and a number of introductory resources. Check out [aisi.ai](https://aisi.ai/) for some ideas for projects within ML & AI safety.

- [Interpretability starter](#interpretability-starter)
  - [Inspiration](#inspiration)
  - [Introductions to mechanistic interpretability](#introductions-to-mechanistic-interpretability)
  - [See also the tools available on interpretability:](#see-also-the-tools-available-on-interpretability)
  - [Digestible research](#digestible-research)
- [Starter projects](#starter-projects)
  - [🙋‍♀️ Simple templates & tools](#️-simple-templates--tools)
    - [Activation Atlas [tool]](#activation-atlas-tool)
    - [EasyTransformer [code]](#easytransformer-code)
      - [Demo notebook of EasyTransformer](#demo-notebook-of-easytransformer)
    - [Converting Neural Networks to graphs [code]](#converting-neural-networks-to-graphs-code)
    - [Reviewing explainability tools](#reviewing-explainability-tools)
    - [The IML R package [code]](#the-iml-r-package-code)
  - [👩‍🔬 Advanced templates and tools](#-advanced-templates-and-tools)
    - [Redwood Research's interpretability on Transformers [tool]](#redwood-researchs-interpretability-on-transformers-tool)

### Inspiration

We have many ideas available for inspiration on the [aisi.ai Interpretability Hackathon ideas list](http://localhost:3000/list/interpretability-hackathon). A lot of interpretability research is available on [distill.pub](https://distill.pub/), [transformer circuits](https://transformer-circuits.pub/), and [Anthropic's research page](https://www.anthropic.com/research).

### Introductions to mechanistic interpretability

- [Catherine Olsson's on getting starter with mechanistic interpretability research](https://www.youtube.com/watch?v=ll0oduwDEwI)
  - [02:00](https://youtu.be/ll0oduwDEwI?t=117) Start of the talk
  - [03:30](https://youtu.be/ll0oduwDEwI?t=205) What is mechanistic interpretability?
  - [Slideshow](https://docs.google.com/presentation/d/1BNY1xaJLBfMzcgrY_zjqtUAu1QXlzkbbhOlV7XVVlC4/edit#slide=id.g175ecc9ec98_0_63)
- [A video walkthrough of A Mathematical Framework for Transformer Circuits](https://www.youtube.com/watch?v=KV5gbOmHbjU).
- The [Transformer Circuits YouTube series](https://www.youtube.com/watch?v=V3NQaDR3xI4&list=PLoyGOS2WIonajhAVqKUgEMNmeq3nEeM51)
- Jacob Hilton's deep learning curriculum [week on interpretability](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/8-Interpretability.md)
- [An annotated list of good interpretability papers](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers), along with summaries and takes on what to focus on.
- [Christoph Molnar's book about traditional interpretability](https://christophm.github.io/interpretable-ml-book)
- [Neel's barebones prerequisites for mechanistic interpretability research](https://www.neelnanda.io/mechanistic-interpretability/prereqs)

### See also the tools available on interpretability:

- Redwood Research's interpretability tools: <http://interp-tools.redwoodresearch.org/>
- The activation atlas: <https://distill.pub/2019/activation-atlas/>
- The Tensorflow playground: <https://playground.tensorflow.org/>
- The Neural Network Playground (train simple neural networks in the browser): <https://nnplayground.com/>
- Visualize different neural network architectures: <http://alexlenail.me/NN-SVG/index.html>

### Digestible research

- [Opinions on Interpretable Machine Learning and 70 Summaries of Recent Papers](https://www.alignmentforum.org/posts/GEPX7jgLMB8vR2qaK/opinions-on-interpretable-machine-learning-and-70-summaries) summarizes a long list of papers that is definitely useful for your interpretability projects.
- [Distill publication on visualizing neural network weights](https://distill.pub/2020/circuits/visualizing-weights/)
- Andrej Karpathy's ["Understanding what convnets learn"](https://cs231n.github.io/understanding-cnn/)
- [Looking inside a neural net](https://ml4a.github.io/ml4a/looking_inside_neural_nets/)
- 12 toy language models designed to be easier to interpret, in the style of a Mathematical Framework for Transformer Circuits: 1, 2, 3 and 4 layer models, for each size one is attention-only, one has GeLU activations and one has SoLU activations (an activation designed to make the model's neurons more interpretable - <https://transformer-circuits.pub/2022/solu/index.html>) (these aren't well documented yet, but are available in EasyTransformer)
- [Anthropic Twitter thread going through some language model results](https://twitter.com/anthropicai/status/1541469936354136064?lang=en)

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

### [The IML R package](https://uc-r.github.io/iml-pkg) [code]

Check out [this tutorial](https://uc-r.github.io/iml-pkg) to using [the IML package](https://github.com/christophM/iml) in R. The package provides a good interface to working with LIME, feature importance, ICE, partial dependence plots, Shapley values, and more.

## 👩‍🔬 Advanced templates and tools

### [Redwood Research's interpretability on Transformers](http://interp-tools.redwoodresearch.org/) [tool]

Redwood Research has created a [wonderful tool](http://interp-tools.redwoodresearch.org/) that can be used to do research into how language models understand text. The ["How to use" document](https://docs.google.com/document/d/1ECwTXrgTqgiMN24L7IantJTaFpyJM2LxXXGq50meFKc/edit) and [their instruction videos](https://www.youtube.com/channel/UCwvzObS_ayucGlYIJCyagdA) are very good introductions and we recommend reading/watching them since the interface can be a bit daunting otherwise.

Watch this video as an intro:

[![Understanding interp-tools by Redwood Research](https://img.youtube.com/vi/zH8YBqdIB-w/0.jpg)](https://www.youtube.com/watch?v=zH8YBqdIB-w)
