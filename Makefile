default: test

build: clean
	./scripts/build-wheel.sh

test: install
	python3 ./tests/test_dispatcher.py

sanity:
	flake8
	black  --skip-string-normalization --line-length 100 --check .

setup:
	pip3 install -r ./requirements.txt

install: clean build
	pip3 install --force-reinstall ./assets/dist/onnxcli-0.0.1-py3-none-any.whl

clean:
	-rm ./assets/dist/*


.PHONY: sanity setup test
