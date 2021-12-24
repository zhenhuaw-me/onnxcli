default: test

DIST_DIR=./assets/dist

build: clean
	pip3 install build
	python -m build --outdir $(DIST_DIR)
	-rm -rf ./onnxcli.egg-info
	-rm -rf ./build

test: install
	python3 ./tests/test_dispatcher.py

check:
	pip3 install flake8 black
	flake8 --max-line-length 120 --max-complexity 20
	black --skip-string-normalization --line-length 120 --check .

setup:
	pip3 install -r ./requirements.txt

install: clean build
	pip3 install --force-reinstall $(DIST_DIR)/onnxcli-0.0.1-py3-none-any.whl

clean:
	-rm $(DIST_DIR)/*


.PHONY: sanity setup test
