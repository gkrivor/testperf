import zipimport
from pathlib import Path
import os

def _this_loader():
    spec = globals().get("__spec__")            # preferred; required in 3.12+ (module __loader__ is deprecated)
    if spec is not None and getattr(spec, "loader", None) is not None:
        return spec.loader
    return globals().get("__loader__")          # legacy fallback

APP_LOADER = _this_loader()
RUNNING_FROM_ARCHIVE = isinstance(APP_LOADER, zipimport.zipimporter)
ARCHIVE_PATH = Path(APP_LOADER.archive) if RUNNING_FROM_ARCHIVE else None
APP_PATH = ARCHIVE_PATH.parent if RUNNING_FROM_ARCHIVE else Path(__file__).parent

def get_available_models(lookup_name):
  cur_dir = APP_PATH.as_posix()
  path_parts = lookup_name.split('.')
  model_name = ''
  fs_idx = 0
  for part in path_parts:
    last_dir = str(cur_dir)
    cur_dir = os.path.join(cur_dir, part)
    if not os.path.exists(cur_dir) or part == '':
      print('\n\nAvailable options: ')
      print("\n".join([f'{model_name[1:] + "." if model_name != "" else ""}{x}...' for x in os.listdir(last_dir)]))
      break
    fs_idx += 1
    model_name = model_name + '.' + part
  if RUNNING_FROM_ARCHIVE:
    import zipfile
    items = {}
    with zipfile.ZipFile(APP_LOADER.archive, 'r') as arch:
      for entry in arch.namelist():
        if not entry.startswith('models/') or not entry.endswith('.py') or entry.endswith('__init__.py'):
          continue
        current_branch = items
        for part in entry[:-3].split('/'):
          if part not in current_branch:
            current_branch[part] = {}
          current_branch = current_branch[part]
    current_branch = items if len(path_parts) > 0 and path_parts[0] == 'models' else {}
    model_name = ''
    arch_idx = 0
    for part in path_parts:
      if part in current_branch:
        current_branch = current_branch[part]
        model_name = model_name + '.' + part
        arch_idx += 1
        continue
      if len(current_branch) > 0 and arch_idx >= fs_idx:
        print('\nDefault options: ')
        print("\n".join([f'{model_name[1:] + "." if model_name != "" else ""}{x}...' for x in current_branch.keys()]))
      break
