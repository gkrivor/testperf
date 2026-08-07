import sys
import settings

# Handling specific commands
if sys.argv.count('--show') > 0:
    idx = sys.argv.index('--show')
    try:
        filename = sys.argv[idx + 1]
        import zipfile
        with zipfile.ZipFile(settings.ARCHIVE_PATH.as_posix(), 'r') as zip_ref:
            if not filename in zip_ref.namelist():
                print(f"File {filename} not found in the archive")
                exit(1)
            with zip_ref.open(filename, 'r') as file:
                print(file.read().decode('utf-8'))
    except:
        print("Cannot show file content")
        exit(1)
    exit(0)

if sys.argv.count('--pip-install') > 0:
    import subprocess
    subprocess.call(['sh', '-c', f'{sys.executable} -m pip install `{settings.ARCHIVE_PATH.as_posix()} --show requirements.txt`'])
    exit(0)

if sys.argv.count('--docker-run') > 0:
    name = "hello-world"
    try:
        name = sys.argv[sys.argv.index('--docker-run') + 1]
    except:
        pass
    if sys.argv.count('--gpu') == 0:
        print(f"docker container run -it --device=/dev/kfd --device=/dev/dri --mount type=bind,src=.,dst=/workspace/shared --rm {name}")
    else:
        gpus = []
        try:
            gpus = [int(x.strip()) for x in sys.argv[sys.argv.index('--gpu') + 1].split(',')]
        except:
            pass
        if len(gpus) == 0:
            print("Select GPUs to use, e.g. --gpu 0,1,2")
            exit(1)
        print(f"docker container run -it --device=/dev/kfd \\")
        for gpu in gpus:
            print(f"--device=/dev/dri/card{(gpu + 1)} \\")
            for idx in range(128 + (gpu * 8), 128 + (gpu * 8 + 8)):
                print(f"--device=/dev/dri/renderD{idx} \\")
        print(f"--mount type=bind,src=.,dst=/workspace/shared --rm {name}")
    exit(0)

if sys.argv.count('--help') > 0:
    import platform
    if platform.system() == 'Windows':
        print("Usage: python testperf.pyz path.to.model")
        print("       python testperf.pyz path/to/model")
    else:
        print("Usage: testperf.pyz path.to.model")
        print("       testperf.pyz path/to/model")

    print("Options (testperf.pyz or python testperf.pyz):")
    print("  --show <filename> - Show the content of a file in the archive")
    print("  --pip-install - Install the requirements from the archive")
    print("  --docker-run <name> - Print a command to run a Docker container")
    print("    --gpu <gpu_ids> - Select GPUs to use, e.g. --gpu 0,1,2")
    print("  --help - Show this help message")
    exit(0)
