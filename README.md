Lightweight Command Line Toolbox for ONNX
=========================================

[![Build and Test](https://github.com/jackwish/onnxcli/workflows/Build%20and%20Test/badge.svg)](https://github.com/jackwish/onnxcli/actions?query=workflow%3A%22Build+and+Test%22)
[![Sanity](https://github.com/jackwish/onnxcli/workflows/Sanity/badge.svg)](https://github.com/jackwish/onnxcli/actions?query=workflow%3ASanity)
[![Coverage](https://codecov.io/gh/jackwish/onnxcli/branch/master/graph/badge.svg)](https://codecov.io/gh/jackwish/onnxcli)


* Aims to improve your development or usage experience of ONNX.
* Serves as a CLI wrapper for most cases.
* Use it like `onnx infer-shape /path/to/model.onnx`.


## Installation

There are three ways to install (please try out with [virtualenv](https://virtualenv.pypa.io)):

1. Install via [pypi package][pypi] `pip install onnxcli`
2. Install via [GitHub repo][github]: `pip install git+https://github.com/jackwish/onnxcli.git`
3. Download and add the code tree to your `$PYTHONPATH`. This is for development purpose since the command line is different.
    ```sh
    git clone https://github.com/jackwish/onnxcli.git
    export PYTHONPATH=$(pwd)/onnxcli:${PYTHONPATH}
    python onnxcli/cli/dispatcher.py <more args>
    ```

## Usage

_Under construction, please refer to command line help._


## License

Apache License Version 2.0.


[pypi]: https://pypi.org/project/onnxcli
[github]: https://github.com/jackwish/onnxcli
