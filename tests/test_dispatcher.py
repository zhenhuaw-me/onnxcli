import shlex
import subprocess as sp


def test_dispatcher():
    cmds = ['onnx infershape ./assets/tests/conv.float32.onnx -o shape.onnx',
            'onnx extract ./assets/tests/conv.float32.onnx extract.onnx --input_names input --output_names output',
           ]
    for cmd in cmds:
        print(cmd)
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
    test_dispatcher()
