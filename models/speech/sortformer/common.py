"""Streaming Sortformer geometries, in 80 ms encoder frames.

Designed wait = (chunk + right context) x 80 ms: audio the model must buffer before it can
emit. It depends only on the geometry, not on the hardware, so a geometry whose designed
wait already exceeds a latency budget cannot meet it on any GPU.

  --geometry NAME[,NAME...]   budget_480ms, card_1.04s, card_30.4s, or custom with --diar-* frames
  --diar-chunk/--diar-rc/--diar-fifo/--diar-spkcache/--diar-update-period/--diar-lc  custom geometry
"""
from ..common import get_arg, get_list_arg

FRAME_MS = 80

GEOMETRIES = {
  'budget_480ms': {'chunk': 6, 'rc': 0, 'fifo': 188, 'spkcache': 188, 'update-period': 144, 'lc': 0},
  'card_1.04s': {'chunk': 6, 'rc': 7, 'fifo': 188, 'spkcache': 188, 'update-period': 144, 'lc': 0},
  'card_30.4s': {'chunk': 340, 'rc': 40, 'fifo': 40, 'spkcache': 188, 'update-period': 300, 'lc': 0},
}


def custom_geometry():
  keys = ['chunk', 'rc', 'fifo', 'spkcache', 'update-period', 'lc']
  values = {key: get_arg(f'--diar-{key}', None, int) for key in keys}
  if all(value is None for value in values.values()):
    return None
  missing = [key for key, value in values.items() if value is None and key != 'lc']
  if missing:
    raise Exception(f'Custom geometry needs --diar-{", --diar-".join(missing)}')
  values['lc'] = values['lc'] or 0
  return values


def selected_geometries(default):
  """List of (name, frames) from --geometry and/or a custom --diar-* geometry."""
  custom = custom_geometry()
  names = get_list_arg('--geometry', [] if custom else default)
  unknown = [name for name in names if name not in GEOMETRIES]
  if unknown:
    raise Exception(f'Unknown --geometry {", ".join(unknown)}, use one of {", ".join(GEOMETRIES)}')
  geometries = [(name, GEOMETRIES[name]) for name in names]
  if custom:
    geometries.append(('custom', custom))
  return geometries


def designed_wait_ms(frames):
  return (frames['chunk'] + frames['rc']) * FRAME_MS


def compute_ms_per_chunk(frames, rtfx):
  """Mean device time per chunk step derived from throughput (not a p50/p95)."""
  return frames['chunk'] * FRAME_MS / rtfx if rtfx else None
