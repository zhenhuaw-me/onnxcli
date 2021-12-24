default: test

build: clean
	./scripts/build-wheel.sh

test: build install
	python3 ./tests/test_dispatcher.py

sanity:
	flake8
	black  --skip-string-normalization --line-length 100 --check .

setup:
	pip3 install -r ./requirements.txt

install:
	pip3 install ./assets/dist/onnxcli-0.0.1-py3-none-any.whl

clean:
	-rm ./assets/dist/*


.PHONY: sanity setup test
