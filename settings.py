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
  fs_path = [APP_PATH.as_posix()]
  archive_path = []
  path_parts = lookup_name.split('.')

  cur_str = fs_path[0]
  for part in path_parts:
    cur_str = os.path.join(cur_str, part)
    if os.path.exists(cur_str):
      fs_path.append(part)

  archive_branch = None
  if RUNNING_FROM_ARCHIVE:
    folder_tree = {'.': []}
    # Make a folder tree from the archive
    import zipfile
    with zipfile.ZipFile(APP_LOADER.archive, 'r') as arch:
      for entry in arch.namelist():
        entry_path = entry.split('/')
        current_branch = folder_tree
        for part in entry_path[:-1]:
          if part not in current_branch:
            current_branch[part] = {'.': []}
          current_branch = current_branch[part]
        current_branch['.'].append(entry_path[-1])

    archive_branch = folder_tree
    for part in path_parts:
      if part in archive_branch:
        archive_branch = archive_branch[part]
        archive_path.append(part)
      else:
        break

  if len(fs_path) >= (len(archive_path) + 1):
    print('\n\nAvailable options: ')
    model_name = '.'.join(fs_path[1:])
    print("\n".join([f'{model_name + "." if model_name != "" else ""}{x}...' for x in os.listdir('/'.join(fs_path)) if os.path.isdir(os.path.join('/'.join(fs_path), x))]))
    print("\n".join([f'{model_name + "." if model_name != "" else ""}{x}' for x in os.listdir('/'.join(fs_path)) if not os.path.isdir(os.path.join('/'.join(fs_path), x))]))

  if (len(archive_path) + 1) >= len(fs_path) and archive_branch is not None:
    print('\nDefault options: ')
    model_name = '.'.join(archive_path)
    print("\n".join([f'{model_name + "." if model_name != "" else ""}{x}...' for x in list(archive_branch.keys()) if x != '.']))
    print("\n".join([f'{model_name + "." if model_name != "" else ""}{x}' for x in archive_branch['.']]))
