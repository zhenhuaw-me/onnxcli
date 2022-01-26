default: test

DIST_DIR=./assets/dist

build: clean
	pip3 install build
	python -m build --outdir $(DIST_DIR)
	-rm -rf ./onnxcli.egg-info
	-rm -rf ./build

test: install
	onnx setup
	python3 ./tests/test_dispatcher.py

check:
	pip3 install flake8 black
	flake8 --max-line-length 120 --max-complexity 20
	black --skip-string-normalization --line-length 120 --check .

format:
	black --skip-string-normalization --line-length 120 .

setup:
	pip3 install -r ./requirements.txt

install: clean build
	pip3 install --force-reinstall --no-dependencies $(DIST_DIR)/onnxcli-*-py3-none-any.whl

clean:
	-rm $(DIST_DIR)/*


.PHONY: sanity setup test
