ONNX Command Line Toolbox
=========================

[![Build and Test](https://github.com/zhenhuaw-me/onnxcli/workflows/Build%20and%20Test/badge.svg)](https://github.com/zhenhuaw-me/onnxcli/actions/workflows/build.yml)
[![CodeQL](https://github.com/zhenhuaw-me/onnxcli/workflows/CodeQL/badge.svg)](https://github.com/zhenhuaw-me/onnxcli/actions/workflows/codeql-analysis.yml)
[![Sanity](https://github.com/zhenhuaw-me/onnxcli/workflows/Sanity/badge.svg)](https://github.com/zhenhuaw-me/onnxcli/actions/workflows/sanity.yml)
[![Coverage](https://codecov.io/gh/zhenhuaw-me/onnxcli/branch/master/graph/badge.svg)](https://codecov.io/gh/zhenhuaw-me/onnxcli)


* Aim to improve your experience of investigating ONNX models.
* Use it like `onnx infershape /path/to/input/model.onnx /path/to/output/model.onnx`. (See the [usage section](#usage).)


## Installation

Recommand to install via [GitHub repo][github] for the latest functionality.
```
pip install git+https://github.com/zhenhuaw-me/onnxcli.git
```

_Two alternative ways are:_
1. Install via [pypi package][pypi] `pip install onnxcli`
2. Download and add the code tree to your `$PYTHONPATH`. (For development purpose and the command line is different.）
    ```
    git clone https://github.com/zhenhuaw-me/onnxcli.git
    export PYTHONPATH=$(pwd)/onnxcli:${PYTHONPATH}
    python onnxcli/cli/dispatcher.py <sub command> <more args...>
    ```

**Requirements**

`onnxcli` depends on different packages w.r.t. different functionality and may extend.
However, we only include only several basic ones (`onnx` for example) since you may use only a small portion of the functionalities, or you may like have a different version.

Depending on the sub command, error will be raised if the requirements are not met.
Follow the error message to install the requirements.


## Usage

Once installed, the `onnx` and `onnxcli` commands are avaiable.
You can play with commands such as `onnx infershape /path/to/input/model.onnx /path/to/output/model.onnx`.

The general format is `onnx <sub command> <dedicated arguments ...>`.
The sub commands are as sections below.

_Check the online help with `onnx --help` and `onnx <subcmd> --help` for latest usage._

### infershape

`onnx infershape` performs [shape inference](https://github.com/onnx/onnx/blob/master/docs/ShapeInference.md) of the ONNX model.
It's an CLI wrapper of [`onnx.shape_inference`](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model).
You will find it useful to generate shape information for the models that are extracted by [`onnx extract`](#extract).

### extract

`onnx extract` extracts the sub model that is determined by the names of the input and output tensor of the subgraph from the original model.
It's a CLI wrapper of [`onnx.utils.extract_model`](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#extracting-sub-model-with-inputs-outputs-tensor-names) (which I authorized in the ONNX repo).

### inspect

`onnx inspect` gives you quick view of the information of the given model.
It's inspired by [the tf-onnx tool](https://github.com/onnx/tensorflow-onnx/blob/master/tools/dump-onnx.py).

When working on deep learning, you may like to take a look at what's inside the model.
[Netron](https://github.com/lutzroeder/netron) is powerful but doesn't provide fine-grain view.

With `onnx inspect`, you no longer need to scroll the Netron window to look for nodes or tensors.
Instead, you can dump the node attributes and tensor values with a single command.

<details><summary>Click here to see a node example</summary>
<pre><code>
$ onnx inspect ./assets/tests/conv.float32.onnx --node --indices 0 --detail

Inpect of model ./assets/tests/conv.float32.onnx
  Graph name: 9
  Graph inputs: 1
  Graph outputs: 1
  Nodes in total: 1
  ValueInfo in total: 2
  Initializers in total: 2
  Sparse Initializers in total: 0
  Quantization in total: 0

Node information:
  Node "output": type "Conv", inputs "['input', 'Variable/read', 'Conv2D_bias']", outputs "['output']"
    attributes: [name: "dilations"
ints: 1
ints: 1
type: INTS
, name: "group"
i: 1
type: INT
, name: "kernel_shape"
ints: 3
ints: 3
type: INTS
, name: "pads"
ints: 1
ints: 1
ints: 1
ints: 1
type: INTS
, name: "strides"
ints: 1
ints: 1
type: INTS
]
</code></pre>
</details>

<details><summary>Click here to see a tensor example</summary>
<pre><code>
$ onnx inspect ./assets/tests/conv.float32.onnx --tensor --names Conv2D_bias --detail

Inpect of model ./assets/tests/conv.float32.onnx
  Graph name: 9
  Graph inputs: 1
  Graph outputs: 1
  Nodes in total: 1
  ValueInfo in total: 2
  Initializers in total: 2
  Sparse Initializers in total: 0
  Quantization in total: 0

Tensor information:
  Initializer "Conv2D_bias": type FLOAT, shape [16],
    float data: [0.4517577290534973, -0.014192663133144379, 0.2946248948574066, -0.9742919206619263, -1.2975586652755737, 0.7223454117774963, 0.7835700511932373, 1.7674627304077148, 1.7242872714996338, 1.1230682134628296, -0.2902531623840332, 0.2627834975719452, 1.0175092220306396, 0.5643373131752014, -0.8244842290878296, 1.2169424295425415]
</code></pre>
</details>

### draw

`onnx draw` draws the graph in `dot`, `svg`, `png` formats.
It gives you quick view of the type and shape of the tensors that are fed to a specific node.
You can view the model topology in image viewer of browser without waiting for the model to load,
which I found is really helpful for large models.

If you are viewing `svg` in browser, you can even quick search for the nodes and tensors.
Together with [`onnx inspect`](#inspect), it will be very efficient to understand the issue you are looking into.

The node are in ellipses and tensors are in rectangles where the rounded ones are initializers.
The node type of the node and the data type and shape of the tenors are also rendered.
Here is a Convolution node example.

![conv](assets/conv.svg)

**Note**: The [`onnx draw`](#draw) requires [`dot` command (graphviz)](https://graphviz.org/) to be avaiable on your machine - which can be installed by command as below on Ubuntu/Debian.
```
sudo apt install -y graphviz
```

### optimize

`onnx optimize` optimizes the input model with [ONNX Optimizer](https://github.com/onnx/optimizer).


## Contributing

Welcome to contribute new commands or enhance them.
Let's make our life easier together.

The workflow is pretty simple:

1. Starting with GitHub Codespace or clone locally.
  1. `make setup` to config the dependencies (or `pip install -r ./requirements.txt` if you prefer).

2. Create a new subcommand
   1. Starting by copying and modifying [infershape](./onnxcli/infer_shape.py).
   2. Register the command in the [dispatcher](./onnxcli/dispatcher.py)
   3. Create a new command line [test](./tests/test_dispatcher.py)
   4. `make test` to build and test.
   5. `make check` and `make format` to fix any code style issues.

3. Try out, debug, commit, push, and open pull request.
   1. The code has been protected by CI. You need to get a pass before merging.
   2. Ask if any questions.


## License

Apache License Version 2.0.


[pypi]: https://pypi.org/project/onnxcli
[github]: https://github.com/zhenhuaw-me/onnxcli
