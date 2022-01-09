import logging
import shlex
import subprocess as sp

from onnxcli.dispatcher import dispatch_core

fmt = '%(asctime)s %(levelname).1s [%(name)s][%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=fmt, level=logging.DEBUG)

logger = logging.getLogger('testing')

cmds = [
    'infershape ./assets/tests/conv.float32.onnx -o shape.onnx',
    'extract ./assets/tests/conv.float32.onnx extract.onnx -i input -o output',
    'inspect ./assets/tests/conv.float32.onnx --meta --node --tensor',
    'inspect ./assets/tests/conv.float32.onnx --node --indices 0 --detail',
    'inspect ./assets/tests/conv.float32.onnx --node --names output --detail',
    'inspect ./assets/tests/conv.float32.onnx --tensor --indices 0 --detail',
    'inspect ./assets/tests/conv.float32.onnx --tensor --names output --detail',
    'draw ./assets/tests/conv.float32.onnx draw.dot --type dot',
    'draw ./assets/tests/conv.float32.onnx draw.svg',
    'optimize ./assets/tests/conv.float32.onnx optimized.onnx',
    'convert ./assets/tests/conv.float32.onnx converted.json --output_type onnx',
]


def test_dispatch_core():
    for cmd in cmds:
        logger.debug("Running {}".format(cmd))

        dispatch_core(shlex.split(cmd))


def test_dispatch_cmd():
    for cmd in cmds:
        cmd = 'onnx ' + cmd
        logger.debug("Running {}".format(cmd))
        p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf-8')
        while True:
            output = p.stdout.readline()
            if p.poll() is not None:
                break
            if output:
                print(output.strip())
        retval = p.poll()
        assert retval == 0


if __name__ == '__main__':
    test_dispatch_core()
    test_dispatch_cmd()
