import os, zipfile, stat
from pathlib import Path
import platform

package = {
    '.': [
        # Scripts
        '__main__.py',
        'argutils.py',
        'settings.py',
        'test_perf.py',
        'reports.py',
        'class_model.py',
        'class_server.py',
        'class_openai.py',
        'class_vllm.py',
        'class_sglang.py',
        # Resources
        '!StatViewer.xlsm',
        'requirements.txt',
    ],
    'models': 
        list(Path('./models/vllm').glob('**/*.py')) +
        list(Path('./models/sgl').glob('**/*.py')) +
        list(Path('./models/qwen').glob('**/*.py')) +
        list(Path('./models/glm').glob('**/*.py')) +
        list(Path('./models/onnx').glob('**/*.py')) +
        list(Path('./models/yolo').glob('**/*.py')) +
        list(Path('./models/effnet').glob('**/*.py')),
    'batches': 
        list(Path('./batches/glm').glob('**/*.py')) +
        list(Path('./batches/qwen').glob('**/*.py')),
}

def build_zip(package, out_file, interpreter=None, exclude=()):
    root_path = Path('.')
    stats = {'added': 0, 'skipped': 0, 'generated': 0, 'size': 0}
    with open(out_file, "wb") as f:
        if interpreter:
            f.write(f"#!{interpreter}\n".encode())          # shebang prefix (see caveat)
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            for root, files in package.items():
                for file in files:
                    lookup_file = Path(file)
                    if not lookup_file.exists():
                        raise FileNotFoundError(f"File {lookup_file} not found")
                    relative_path = lookup_file.relative_to(root_path).as_posix()
                    parent_path = lookup_file.parent.relative_to(root_path)
                    if relative_path in exclude:
                        stats['skipped'] += 1
                        continue
                    print(f"Adding {relative_path}")
                    z.write(lookup_file, relative_path)
                    stats['added'] += 1
                    current_path = parent_path
                    while current_path != root_path:
                        if not (current_path / '__init__.py').as_posix() in z.namelist():
                            print(f"Generating {current_path / '__init__.py'}")
                            # Need to allow use external modules in the models folder (and all subfolders if they have diff)
                            z.writestr((current_path / '__init__.py').as_posix(), """from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
""")
                            stats['generated'] += 1
                        current_path = current_path.parent

    if platform.system() != 'Windows':
        os.chmod(out_file, os.stat(out_file).st_mode | stat.S_IEXEC)

    stats['size'] = os.path.getsize(out_file)
    
    return stats

if __name__ == '__main__':
    stats = build_zip(package, 'testperf.pyz', interpreter='/usr/bin/env python3')
    print(f'Added {stats['added']} files')
    print(f'Skipped {stats['skipped']} files')
    print(f'Generated {stats['generated']} files')
    print(f'Size {stats['size']} bytes')