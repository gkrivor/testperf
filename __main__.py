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

if sys.argv.count('--help') > 0:
    import platform
    if platform.system() == 'Windows':
        print("Usage: python testperf.pyz path.to.model")
        print("Usage: python testperf.pyz path/to/model")
    else:
        print("Usage: testperf.pyz path.to.model")
        print("Usage: testperf.pyz path/to/model")
    
    print("Options (testperf.pyz or python testperf.pyz):")
    print("  --show <filename> - Show the content of a file in the archive")
    print("  --pip-install - Install the requirements from the archive")
    print("  --help - Show this help message")
    exit(0)

import test_perf