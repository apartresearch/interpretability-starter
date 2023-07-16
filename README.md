# Interpretability starter

Interpretability research is an exciting and growing field of machine learning. If we are able to understand what happens within neural networks in diverse domains, we can see **why** the network gives us specific outputs and detect deception, understand choices, and change how they work.

![](https://uploads-ssl.webflow.com/634e78132252d2b0203a9ac8/635662f4e41031331b88038e_2_interpretability_teaser.png)

This list of resources was made for the Interpretability Hackathon ([link](https://alignmentjam.com/jam/interpretability)) and contains an array of useful starter templates, tools to investigate model activations, and a number of introductory resources. Check out [aisi.ai](https://aisi.ai/) for some ideas for projects within ML & AI safety.

- [Interpretability starter](#interpretability-starter)
  - [Inspiration](#inspiration)
    - [Introductions to mechanistic interpretability](#introductions-to-mechanistic-interpretability)
    - [See also the tools available on interpretability:](#see-also-the-tools-available-on-interpretability)
    - [Digestible research](#digestible-research)
- [Concepts](#concepts)
  - [Features \[Olah et al., 2020\]](#features-olah-et-al-2020)
  - [Superposition \[Elhage et al., 2022\]](#superposition-elhage-et-al-2022)
    - [Polysemanticity \[Elhage et al., 2022\]](#polysemanticity-elhage-et-al-2022)
    - [Privileged basis](#privileged-basis)
  - [Models of MLP neuron activation \[Foote et al., 2023; Bills et al., 2023\]](#models-of-mlp-neuron-activation-foote-et-al-2023-bills-et-al-2023)
  - [Identifying meaningful circuits of compoennts in Transformers](#identifying-meaningful-circuits-of-compoennts-in-transformers)
    - [Causal tracing \[Meng et al., 2022\]](#causal-tracing-meng-et-al-2022)
  - [Memory editing \[Meng et al., 2022; Meng et al., 2023; Hoelscher-Obermaier, 2023\]](#memory-editing-meng-et-al-2022-meng-et-al-2023-hoelscher-obermaier-2023)
    - [Machine unlearning](#machine-unlearning)
    - [Concept erasure](#concept-erasure)
  - [Ablation](#ablation)
    - [Ablation as model editing \[Li et al., 2023\]](#ablation-as-model-editing-li-et-al-2023)
  - [Adding activation vectors to modulate behavior \[Turner et al., 2023\]](#adding-activation-vectors-to-modulate-behavior-turner-et-al-2023)
  - [Automated circuit detection \[Conmy et al., 2023\]](#automated-circuit-detection-conmy-et-al-2023)
  - [Linear probes](#linear-probes)
    - [Sparse probing \[Gurnee et al., 2023\]](#sparse-probing-gurnee-et-al-2023)
- [Starter projects](#starter-projects)
  - [🙋‍♀️ Simple templates \& tools](#️-simple-templates--tools)
    - [Activation Atlas \[tool\]](#activation-atlas-tool)
    - [BertViz](#bertviz)
    - [EasyTransformer \[code\]](#easytransformer-code)
      - [Demo notebook of EasyTransformer](#demo-notebook-of-easytransformer)
    - [Converting Neural Networks to graphs \[code\]](#converting-neural-networks-to-graphs-code)
    - [Reviewing explainability tools](#reviewing-explainability-tools)
    - [The IML R package \[code\]](#the-iml-r-package-code)
  - [👩‍🔬 Advanced templates and tools](#-advanced-templates-and-tools)
    - [Redwood Research's interpretability on Transformers \[tool\]](#redwood-researchs-interpretability-on-transformers-tool)

## Inspiration

We have many ideas available for inspiration on the [aisi.ai Interpretability Hackathon ideas list](http://localhost:3000/list/interpretability-hackathon). A lot of interpretability research is available on [distill.pub](https://distill.pub/), [transformer circuits](https://transformer-circuits.pub/), and [Anthropic's research page](https://www.anthropic.com/research).

### Introductions to mechanistic interpretability

- [Keynote talk of the Interpretability Hackathon 3.0](https://youtu.be/lzPOspNnnYc) with [Neel Nanda](https://www.neelnanda.io/about)
- [A video walkthrough of A Mathematical Framework for Transformer Circuits](https://www.youtube.com/watch?v=KV5gbOmHbjU).
- The [Transformer Circuits YouTube series](https://www.youtube.com/watch?v=V3NQaDR3xI4&list=PLoyGOS2WIonajhAVqKUgEMNmeq3nEeM51)
- Callum McDougall's introduction to mechanistic interpretability
- Jacob Hilton's deep learning curriculum [week on interpretability](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/8-Interpretability.md)
- [An annotated list of good interpretability papers](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers), along with summaries and takes on what to focus on.
- [Christoph Molnar's book about traditional interpretability](https://christophm.github.io/interpretable-ml-book)

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
- [A Walkthrough of Finding Neurons In A Haystack](https://youtu.be/r1cfSpVAeqQ) (pt. 1)
- [Anthropic Twitter thread going through some language model results](https://twitter.com/anthropicai/status/1541469936354136064?lang=en)

# Concepts

## Features [[Olah et al., 2020](https://distill.pub/2020/circuits/zoom-in/)]

> A feature is a a scalar function of the input. In this essay, neural network features are directions, and often simply individual neurons. We claim such features in neural networks are typically meaningful features which can be rigorously studied. A meaningful feature is one that genuinely responds to an articulable property of the input, such as the presence of a curve or a floppy ear.

## Superposition [[Elhage et al., 2022]](https://transformer-circuits.pub/2022/toy_model/index.html)

Superposition is when a there are more features in the feature space than there are neurons. This is nearly always the case for e.g. large language models (LLMs). It leads to neurons with polysemanticity.

### Polysemanticity [[Elhage et al., 2022]](https://transformer-circuits.pub/2022/toy_model/index.html)

Polysemanticity is the phenomenon that a neuron corresponds to multiple features, i.e. it encodes multiple concepts / semantic features at a time. It often makes the neuron less interpretable. **Sparse features**[^1] are more likely to be encoded in a polysemantic neuron due to the probability of non-interference.

[^1]: Sparse features are infrequent in the data.

### Privileged basis

A privileged basis is when the standard vectors are human-understandable and meaningful. In the context of model neurons, this means that a neuron's activation represents a meaningful concept. If a neuron is not in a privileged basis, it will be significantly harder to interpret and any transformations (such as ReLU) will cause interference instead of improving the interpretability.

## Models of MLP neuron activation [[Foote et al., 2023](https://arxiv.org/abs/2305.19911); [Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)]

MLP neuron activation models are models that attempt to explain in which cases neurons fire. It's based on a few principles: 1) We expect MLP neurons to activate in specific token sequences, 2) we can create a simplified model of its activation that does not require the neural network, and 3) that model can be validated against real activation.

[Foote et al. [2023]](https://arxiv.org/abs/2305.19911) create a semantic graph model over the token sequences that a neuron activates to while [Bills et al. [2023]](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) use GPT-4 to create explanations and use these explanations to predict activatino.

## Identifying meaningful circuits of compoennts in Transformers

### Causal tracing [[Meng et al., 2022]](https://memit.baulab.info/)

## Memory editing [[Meng et al., 2022](https://memit.baulab.info/); [Meng et al., 2023](https://arxiv.org/pdf/2210.07229.pdf); [Hoelscher-Obermaier, 2023](https://arxiv.org/pdf/2305.17553.pdf)]

Memory editing of language models was introduced with

### Machine unlearning

### Concept erasure

## Ablation

### Ablation as model editing [[Li et al., 2023]](https://openreview.net/forum?id=ytYaiSQNCB)

Using activation ablastion, you can remove causal connections between parts of a model (e.g. attention heads) to modify behaviors. [Li et al. [2023]](https://openreview.net/forum?id=ytYaiSQNCB) reduce toxicity of a model from 45% to 33%. They do this by training a binary edge mask over the computational graph of causal connections to perform poorly on their "negative examples" dataset while maintaining performance.

## Adding activation vectors to modulate behavior [[Turner et al., 2023]](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)

## Automated circuit detection [[Conmy et al., 2023]](https://arxiv.org/pdf/2304.14997.pdf)

## Linear probes

### Sparse probing [[Gurnee et al., 2023]](https://arxiv.org/pdf/2305.01610.pdf)

This is basically linear probes that constrain the amount of neurons of the probe. It mitigates the problem that the linear probe itself does computation, even if it's just linear. Neel Nanda [describes](https://youtu.be/r1cfSpVAeqQ?t=867) a critique of linear probes as 3D: 1) You design what feature you're looking for, not getting the chance to find features from a model-first perspective. 2) There is a chance the linear probe does computation since we force it to fit the data, i.e. the model might not represent this. 3) Probing is correlational rather than causal. Sparse probing still suffers from (1) and (3). However, it is less susceptible to correlations and it identifies individual neurons perfectly with prileged bases. **Useful for first explorations**.

# Starter projects

## 🙋‍♀️ Simple templates & tools

### [Activation Atlas](https://distill.pub/2019/activation-atlas/) [tool]

The [Activation Atlas article](https://distill.pub/2019/activation-atlas/) has a lot of figures where each has a Google Colab associated with them. Click on the "Try in a notebook". An example is [this notebook](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb) that shows a simple activation atlas.

Additionally, they have [this tool](https://distill.pub/2019/activation-atlas/app.html) to explore to which sorts of images neurons activate the most to.

### [BertViz](https://github.com/jessevig/bertviz)

BertViz is an interactive tool for visualizing attention in [Transformer](https://jalammar.github.io/illustrated-transformer/) language models such as BERT, GPT2, or T5. It can be run inside a Jupyter or Colab notebook through a simple Python API that supports most [Huggingface models](https://huggingface.co/models). BertViz extends the [Tensor2Tensor visualization tool](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization) by [Llion Jones](https://medium.com/@llionj), providing multiple views that each offer a unique lens into the attention mechanism.

![BertViz example image](https://github.com/jessevig/bertviz/raw/master/images/neuron-view-dark.gif)

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

- [Inseq](https://github.com/inseq-team/inseq): Inseq is a Python library to perform feature attribution of decoder-only and encoder-decoder models from the Hugging Face Transformers library. It supports multiple gradient, attention and perturbation-based attribution methods, with visualizations in Jupyter and console. See the [demo paper](http://arxiv.org/abs/2302.13942) for more detail.

<img src="https://raw.githubusercontent.com/inseq-team/inseq/main/docs/source/images/inseq_python_console.gif" alt="Inseq console visualizations" width="800">

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
