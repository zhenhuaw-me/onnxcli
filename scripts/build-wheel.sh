#!/bin/bash

if [ "$(uname -s)" == "Darwin" ]; then
  root_dir=$(dirname $(dirname $(greadlink -f $0})))
else
  root_dir=$(dirname $(dirname $(readlink -f $0})))
fi
rm -f ${root_dir}/assets/dist/onnxcli-*.whl

pip install build

python -m build \
  --outdir ${root_dir}/assets/dist
rm -rf ${root_dir}/onnxcli.egg-info
rm -rf ${root_dir}/build
