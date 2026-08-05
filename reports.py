import sys
import os
import subprocess
import datetime
import platform
from copy import deepcopy
from shutil import copy, which
import traceback

def performance_report(model,model_name, read_times, inference_times, warm_up_times, batches):
  workbook_path = None
  try:
    import openpyxl
    from openpyxl.chart import LineChart, Reference, Series
    from openpyxl.chart.series import SeriesLabel
    from openpyxl.chart.layout import Layout, ManualLayout
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    main_sheet = wb.active
    read_sheet = wb.create_sheet("Read")
    inference_sheet = wb.create_sheet("Inference")

    read_sheet.append(["Reading times"])

    offset_col = 2
    offset_stat_row = read_sheet.max_row + 1
    read_sheet.append(["Average"])
    read_sheet.append(["Median"])
    read_sheet.append(["90th Percentile"])
    read_sheet.append(["95th Percentile"])
    read_sheet.append(["99th Percentile"])
    read_sheet.append(["Minimum"])
    read_sheet.append(["Maximum"])

    read_sheet.append(["Run", "Time (s)"])

    col_letter = get_column_letter(offset_col)
    offset_row = read_sheet.max_row + 1
    last_row = offset_row + len(read_times) - 2
    read_sheet[col_letter + str(offset_stat_row + 0)] = "=AVERAGE(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
    read_sheet[col_letter + str(offset_stat_row + 1)] = "=MEDIAN(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
    read_sheet[col_letter + str(offset_stat_row + 2)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.9)"
    read_sheet[col_letter + str(offset_stat_row + 3)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.95)"
    read_sheet[col_letter + str(offset_stat_row + 4)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.99)"
    read_sheet[col_letter + str(offset_stat_row + 5)] = "=MIN(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
    read_sheet[col_letter + str(offset_stat_row + 6)] = "=MAX(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"

    offset_col = 2
    offset_row = read_sheet.max_row + 1
    last_row = offset_row + len(read_times) - 2

    idx = 1
    for item in read_times[:-1]:  
        read_sheet.append([idx, item])
        idx += 1

    series = Series(values=Reference(read_sheet, min_col=offset_col, min_row=offset_row, max_col=offset_col, max_row=last_row), title="Reading times")
    if len(read_times) <= 2:
        series.marker.symbol = "circle"
        series.marker.size = 6
    chart = LineChart()
    chart.series.append(series)
    chart.title = "Reading times"
    chart.x_axis.title = "Run"
    chart.y_axis.title = "Time (s)"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.legend = None
    chart.layout=Layout(
        manualLayout=ManualLayout(
            x=0.02, y=0.02,
            h=0.75, w=0.9,
        )
    )
    read_sheet.add_chart(chart, "C1")
    main_sheet.add_chart(deepcopy(chart), "F35")

    inference_sheet.append(["Inference times"])
    inference_table = [[] for _ in range(len(batches))]
    x_axis_max = 0
    x_axis_min = float('inf')
    for batch_index in range(len(batches)):
        for item in inference_times[batches[batch_index]][:-1]:
            inference_table[batch_index].append(item)
            x_axis_max = max(x_axis_max, item)
            x_axis_min = min(x_axis_min, item)

    batch_inference_lengths = [len(inference_table[batch_index]) for batch_index in range(len(batches))]

    # Table header
    inference_sheet.append(["Metric"] + [f"Batch {batch}" for batch in batches])
    # Aggregated statistics
    inference_sheet.column_dimensions[get_column_letter(1)].width = 30
    offset_col = 2
    offset_stat_row = inference_sheet.max_row + 1
    inference_sheet.append(["Average"])
    inference_sheet.append(["Median"])
    inference_sheet.append(["90th Percentile"])
    inference_sheet.append(["95th Percentile"])
    inference_sheet.append(["99th Percentile"])
    inference_sheet.append(["Minimum"])
    inference_sheet.append(["Maximum"])
    inference_sheet.append(["IPS (Average)"])
    inference_sheet.append(["IPS (Median)"])
    inference_sheet.append(["IPS (90th Percentile)"])
    inference_sheet.append(["IPS (95th Percentile)"])
    inference_sheet.append(["IPS (99th Percentile)"])
    inference_sheet.append(["BPS (Average)"])
    inference_sheet.append(["BPS (Median)"])
    inference_sheet.append(["BPS (90th Percentile)"])
    inference_sheet.append(["BPS (95th Percentile)"])
    inference_sheet.append(["BPS (99th Percentile)"])
    inference_sheet.append(["Warm Up Time"])

    # Table header
    inference_sheet.append(["Run"] + [f"Batch {batch}" for batch in batches])

    offset_col = 2
    offset_row = inference_sheet.max_row + 1

    for batch_index in range(len(batches)):
        last_row = offset_row + batch_inference_lengths[batch_index] - 1
        col_letter = get_column_letter(offset_col + batch_index)
        inference_sheet[col_letter + str(offset_stat_row + 0)] = "=AVERAGE(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
        inference_sheet[col_letter + str(offset_stat_row + 1)] = "=MEDIAN(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
        inference_sheet[col_letter + str(offset_stat_row + 2)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.9)"
        inference_sheet[col_letter + str(offset_stat_row + 3)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.95)"
        inference_sheet[col_letter + str(offset_stat_row + 4)] = "=_xlfn.PERCENTILE.INC(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ", 0.99)"
        inference_sheet[col_letter + str(offset_stat_row + 5)] = "=MIN(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
        inference_sheet[col_letter + str(offset_stat_row + 6)] = "=MAX(" + col_letter + str(offset_row) + ":" + col_letter + str(last_row) + ")"
        # Inference Per Second depending on calculated time
        inference_sheet[col_letter + str(offset_stat_row + 7)] = "=1 / " + col_letter + str(offset_stat_row + 0)
        inference_sheet[col_letter + str(offset_stat_row + 8)] = "=1 / " + col_letter + str(offset_stat_row + 1)
        inference_sheet[col_letter + str(offset_stat_row + 9)] = "=1 / " + col_letter + str(offset_stat_row + 2)
        inference_sheet[col_letter + str(offset_stat_row + 10)] = "=1 / " + col_letter + str(offset_stat_row + 3)
        inference_sheet[col_letter + str(offset_stat_row + 11)] = "=1 / " + col_letter + str(offset_stat_row + 4)
        # Batch Per Second depending on IPS
        inference_sheet[col_letter + str(offset_stat_row + 12)] = "=" + str(batches[batch_index]) + " * " + col_letter + str(offset_stat_row + 7)
        inference_sheet[col_letter + str(offset_stat_row + 13)] = "=" + str(batches[batch_index]) + " * " + col_letter + str(offset_stat_row + 8)
        inference_sheet[col_letter + str(offset_stat_row + 14)] = "=" + str(batches[batch_index]) + " * " + col_letter + str(offset_stat_row + 9)
        inference_sheet[col_letter + str(offset_stat_row + 15)] = "=" + str(batches[batch_index]) + " * " + col_letter + str(offset_stat_row + 10)
        inference_sheet[col_letter + str(offset_stat_row + 16)] = "=" + str(batches[batch_index]) + " * " + col_letter + str(offset_stat_row + 11)
        inference_sheet[col_letter + str(offset_stat_row + 17)] = str(warm_up_times[batches[batch_index]])

    chart = LineChart()
    chart.title = "Metrics"
    chart.x_axis.title = "Batch Size"
    chart.y_axis.title = "Time (s)"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = x_axis_max
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    metrics = ["Average", "Median", "90th Percentile", "95th Percentile", "99th Percentile", "Minimum", "Maximum"]
    for metric_index in range(len(metrics)):
        series = Series(values=Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row + metric_index, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row + metric_index), title=f"{metrics[metric_index]}")
        series.marker.symbol = "circle"
        series.marker.size = 6
        chart.series.append(series)
    batch_titles = Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row - 1, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row - 1)
    chart.set_categories(batch_titles)
    chart.legend.position = 'b'
    chart.layout=Layout(
        manualLayout=ManualLayout(
            x=0.02, y=0.02,
            h=0.65, w=0.9,
        )
    )
    chart.width = 25
    inference_sheet.add_chart(chart, get_column_letter(len(batches) + 2) + "1")
    main_sheet.add_chart(deepcopy(chart), "F5")

    chart = LineChart()
    chart.title = "IPS"
    chart.x_axis.title = "Batch Size"
    chart.y_axis.title = "Inferences Per Second"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    metrics = ["Average", "Median", "90th Percentile", "95th Percentile", "99th Percentile"]
    for metric_index in range(len(metrics)):
        series = Series(values=Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row + 7 + metric_index, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row + 7 + metric_index), title=f"{metrics[metric_index]}")
        series.marker.symbol = "circle"
        series.marker.size = 6
        chart.series.append(series)
    batch_titles = Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row - 1, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row - 1)
    chart.set_categories(batch_titles)
    chart.legend.position = 'b'
    chart.layout=Layout(
        manualLayout=ManualLayout(
            x=0.02, y=0.02,
            h=0.65, w=0.9,
        )
    )
    chart.width = 15
    inference_sheet.add_chart(chart, get_column_letter(len(batches) + 2) + "16")
    main_sheet.add_chart(deepcopy(chart), "F20")

    chart = LineChart()
    chart.title = "BPS"
    chart.x_axis.title = "Batch Size"
    chart.y_axis.title = "Batches Per Second"
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    metrics = ["Average", "Median", "90th Percentile", "95th Percentile", "99th Percentile"]
    for metric_index in range(len(metrics)):
        series = Series(values=Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row + 12 + metric_index, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row + 12 + metric_index), title=f"{metrics[metric_index]}")
        series.marker.symbol = "circle"
        series.marker.size = 6
        chart.series.append(series)
    batch_titles = Reference(inference_sheet, min_col=offset_col, min_row=offset_stat_row - 1, max_col=offset_col + len(batches) - 1, max_row=offset_stat_row - 1)
    chart.set_categories(batch_titles)
    chart.legend.position = 'b'
    chart.layout=Layout(
        manualLayout=ManualLayout(
            x=0.02, y=0.02,
            h=0.65, w=0.9,
        )
    )
    chart.width = 15
    inference_sheet.add_chart(chart, get_column_letter(len(batches) + 11) + "16")
    main_sheet.add_chart(deepcopy(chart), "P20")

    idx = 0
    max_rows = max(batch_inference_lengths)
    for idx in range(1, max_rows + 1):
        row = [idx]
        for batch_index in range(len(batches)):
            row.append(inference_table[batch_index][idx - 1] if len(inference_table[batch_index]) > idx - 1 else None)
        inference_sheet.append(row)
    chart = LineChart()
    chart.title = "Inference times"
    chart.x_axis.title = "Run"
    chart.y_axis.title = "Time (s)"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = x_axis_max
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    for batch_index in range(len(batches)):
        series = Series(values=Reference(inference_sheet, min_col=batch_index + offset_col, min_row=offset_row, max_col=batch_index + offset_col, max_row=last_row), title=f"Batch {batches[batch_index]}")
        chart.series.append(series)
    chart.width = 15
    chart.legend.position = 'b'
    chart.layout=Layout(
        manualLayout=ManualLayout(
            x=0.02, y=0.02,
            h=0.65, w=0.9,
        )
    )
    inference_sheet.add_chart(chart, get_column_letter(len(batches) + 2) + "33")
    main_sheet.add_chart(deepcopy(chart), "P35")

    report_datetime = datetime.datetime.now()
    main_sheet.title = "Overview"
    main_sheet.column_dimensions[get_column_letter(1)].width = 30
    main_sheet.append(['Model:', model_name])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Description:', str(model)])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Run Command:', ' '.join(sys.argv)])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Report Date:', report_datetime.strftime('%Y-%m-%d %H:%M:%S')])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=6)
    main_sheet.append(['Batches:', *batches])
    main_sheet.append(['Total Inference Runs:', model.total_inference_runs])
    main_sheet.append([])
    main_sheet.append(['System Information:'])
    try:
        main_sheet.append(['Hostname:', platform.node()])
        main_sheet.append(['OS:', platform.system()])
        main_sheet.append(['OS Version:', platform.version()])
        main_sheet.append(['OS Release:', platform.release()])
    except Exception as e:
        main_sheet.append([f'Cannot get OS information {e}'])
    main_sheet.append(['Python Version:', sys.version])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)

    try:
        main_sheet.append(['CPU:', platform.processor()])
        accelerators = enumerate_accelerators()
        for item in accelerators['gpu']:
            main_sheet.append(['GPU:', item['name']])
        for item in accelerators['npu']:
            main_sheet.append(['NPU:', item['name']])
    except Exception as e:
        main_sheet.append([f'Cannot get accelerators information {e}'])

    try:
        result = subprocess.run(
            ['pip', 'list', '--format', 'columns'],
            capture_output=True,
            text=True
        )
        output = result.stdout.split('\n')
        for item in output:
            main_sheet.append(item.split())
    except Exception as e:
        main_sheet.append([f'Cannot get PIP list {e}'])

    try:
        main_sheet.append(['Loaded Modules:'])
        for item in sorted(list_loaded_modules()['modules'], key=lambda x: x['name']):
            main_sheet.append([item['name'], item['path']])
    except Exception as e:
        main_sheet.append([f'Cannot get loaded modules {e}'])

    try:
        main_sheet.append([])
        main_sheet.append(['Environment Variables:'])
        for key, value in sorted(os.environ.items(), key=lambda x: x[0]):
            main_sheet.append([key, value])
    except Exception as e:
        main_sheet.append([f'Cannot get environment variables {e}'])

    workbook_path = f"{platform.node().lower()}_{model_name}_{report_datetime.strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb.save(workbook_path)

    reports_path = os.path.join(os.path.dirname(__file__), 'reports', report_datetime.strftime("%Y%m%d"))
    if not os.path.exists(reports_path):
      os.makedirs(reports_path)
      # Copying statistics aggregator to a reports folder
      try:
        copy(os.path.join(os.path.dirname(__file__), "!StatViewer.xlsm"), os.path.join(reports_path, "!StatViewer.xlsm"))
      except Exception as e:
        print(f'{{ "Error": "Failed to copy !StatViewer.xlsm {e}" }}')
    os.rename(workbook_path, os.path.join(reports_path, workbook_path))

    print('{ "Workbook": "' + os.path.join(reports_path, workbook_path).replace("\\", "/") + '" },')

  except Exception as e:
    print(f'{{ "Error": "Failed to load openpyxl {e}" }},')
    traceback.print_exc()
  return workbook_path

def _run(cmd, timeout=5):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        return out, err, p.returncode
    except Exception as e:
        return "", str(e), 1

def _windows_gpus():
    # Uses built-in PowerShell + CIM (WMI) to enumerate GPUs
    ps = [
        "powershell", "-NoProfile", "-Command",
        r"Get-CimInstance Win32_VideoController | "
        r"Select-Object Name,AdapterCompatibility,PNPDeviceID | ConvertTo-Json -Depth 3"
    ]
    out, _, rc = _run(ps, timeout=8)
    if rc != 0 or not out:
        return []
    try:
        import json
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "name": d.get("Name"),
                "vendor": d.get("AdapterCompatibility"),
                "pnp_device_id": d.get("PNPDeviceID"),
            }
            for d in data
            if d.get("Name")
        ]
    except Exception:
        return []

def _windows_npus():
    # Windows 11 often exposes NPUs under PnP class "Neural"
    ps = [
        "powershell", "-NoProfile", "-Command",
        r"Get-CimInstance Win32_PnPEntity | "
        r"Where-Object { $_.PNPClass -eq 'Neural' -or $_.Name -match '(?i)\bNPU\b|Neural Engine|Neural Processing|AI Accelerator' } | "
        r"Select-Object Name,PNPClass,DeviceID | ConvertTo-Json -Depth 3"
    ]
    out, _, rc = _run(ps, timeout=10)
    if rc != 0 or not out:
        return []
    try:
        import json
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [
            {"name": d.get("Name"), "class": d.get("PNPClass"), "device_id": d.get("DeviceID")}
            for d in data
            if d.get("Name")
        ]
    except Exception:
        return []

def _linux_lspci_lines():
    if not which("lspci"):
        return []
    out, _, rc = _run(["lspci", "-nn"], timeout=5)
    if rc != 0 or not out:
        return []
    return out.splitlines()

def _linux_gpus():
    lines = _linux_lspci_lines()
    gpu_markers = ("VGA compatible controller", "3D controller", "Display controller")
    gpus = []
    for ln in lines:
        if any(m in ln for m in gpu_markers):
            gpus.append({"name": ln})
    return gpus

def _linux_npus():
    lines = _linux_lspci_lines()
    # PCI class "Processing accelerators" is common for AI/NPUs, plus keyword heuristics
    npus = []
    import re
    for ln in lines:
        if ("Processing accelerators" in ln) or re.search(r"(?i)\bNPU\b|Neural|AI accelerator|TPU", ln):
            npus.append({"name": ln})
    return npus

def _mac_gpus():
    if not which("system_profiler"):
        return []
    out, _, rc = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10)
    if rc != 0 or not out:
        return []
    try:
        import json
        data = json.loads(out)
        items = data.get("SPDisplaysDataType", [])
        gpus = []
        for it in items:
            # Keys vary by macOS version; keep it simple
            name = it.get("sppci_model") or it.get("_name")
            if name:
                gpus.append({"name": name, "raw": it})
        return gpus
    except Exception:
        return []

def _mac_npus():
    # Apple Neural Engine shows up in hardware profile text on Apple Silicon
    if not which("system_profiler"):
        return []
    out, _, rc = _run(["system_profiler", "SPHardwareDataType"], timeout=8)
    if rc != 0 or not out:
        return []
    import re
    m = re.search(r"Neural Engine:\s*(.+)", out)
    return [{"name": f"Apple Neural Engine ({m.group(1).strip()})"}] if m else []

def enumerate_accelerators():
    osname = platform.system()
    if osname == "Windows":
        return {"gpu": _windows_gpus(), "npu": _windows_npus()}
    if osname == "Linux":
        return {"gpu": _linux_gpus(), "npu": _linux_npus()}
    if osname == "Darwin":
        return {"gpu": _mac_gpus(), "npu": _mac_npus()}
    return {"gpu": [], "npu": []}

def list_loaded_modules():
    osname = platform.system()
    result = {
        'pid': os.getpid(),
        'executable': sys.executable,
        'modules': []
    }

    if osname == "Windows":
        result['modules'] = _windows_list_modules()
    elif osname == "Linux":
        result['modules'] = _linux_list_modules()
    elif osname == "Darwin":
        result['modules'] = _mac_list_modules()

    return result

def _windows_list_modules():
    modules = []

    # Try using psutil first (most reliable cross-platform method)
    try:
        import psutil
        process = psutil.Process()
        for dll in process.memory_maps():
            modules.append({
                'name': os.path.basename(dll.path),
                'path': dll.path
            })
        return modules
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to ctypes approach
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi

        hProcess = kernel32.GetCurrentProcess()

        hMods = (wintypes.HMODULE * 1024)()
        cbNeeded = wintypes.DWORD()

        if psapi.EnumProcessModules(hProcess, ctypes.byref(hMods), ctypes.sizeof(hMods), ctypes.byref(cbNeeded)):
            count = int(cbNeeded.value / ctypes.sizeof(wintypes.HMODULE))

            for i in range(count):
                module_name = ctypes.create_unicode_buffer(260)
                module_path = ctypes.create_unicode_buffer(260)

                if psapi.GetModuleFileNameExW(hProcess, hMods[i], module_path, ctypes.sizeof(module_path)):
                    if psapi.GetModuleBaseNameW(hProcess, hMods[i], module_name, ctypes.sizeof(module_name)):
                        modules.append({
                            'name': module_name.value,
                            'path': module_path.value,
                            'base_address': hex(hMods[i]) if hMods[i] else None
                        })
    except Exception as e:
        # If all else fails, return error info
        modules.append({'error': str(e)})

    return modules

def _linux_list_modules():
    modules = []
    seen_paths = set()

    # Try using psutil first
    try:
        import psutil
        process = psutil.Process()
        for mmap in process.memory_maps():
            if mmap.path and mmap.path not in seen_paths:
                seen_paths.add(mmap.path)
                modules.append({
                    'name': os.path.basename(mmap.path),
                    'path': mmap.path
                })
        return modules
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to reading /proc/self/maps
    try:
        with open('/proc/self/maps', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6:
                    pathname = ' '.join(parts[5:])
                    if pathname and pathname not in ['[stack]', '[heap]', '[vdso]', '[vsyscall]']:
                        if pathname.startswith('/') and pathname not in seen_paths:
                            seen_paths.add(pathname)
                            address = parts[0].split('-')[0]
                            modules.append({
                                'name': os.path.basename(pathname),
                                'path': pathname,
                                'base_address': '0x' + address
                            })
    except Exception as e:
        modules.append({'error': str(e)})

    return modules

def _mac_list_modules():
    modules = []

    # Try using psutil first
    try:
        import psutil
        process = psutil.Process()
        for mmap in process.memory_maps():
            if mmap.path:
                modules.append({
                    'name': os.path.basename(mmap.path),
                    'path': mmap.path
                })
        return modules
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to vmmap command
    try:
        pid = os.getpid()
        out, _, rc = _run(['vmmap', str(pid)], timeout=10)
        if rc == 0 and out:
            seen_paths = set()
            for line in out.splitlines():
                if '/' in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith('/') and os.path.exists(part):
                            if part not in seen_paths:
                                seen_paths.add(part)
                                modules.append({
                                    'name': os.path.basename(part),
                                    'path': part
                                })
    except Exception as e:
        modules.append({'error': str(e)})

    return modules

# Canonical benchmark result schema used by the reports below. This is the vLLM
# result shape and is treated as the base: reports never emit anything outside
# of these keys.
_VLLM_RESULT_KEYS = [
    'num_prompts',
    'request_throughput',
    'output_throughput',
    'total_token_throughput',
    'max_output_tokens_per_s',
    'mean_ttft_ms',
    'median_ttft_ms',
    'std_ttft_ms',
    'p99_ttft_ms',
    'mean_tpot_ms',
    'median_tpot_ms',
    'std_tpot_ms',
    'p99_tpot_ms',
    'mean_itl_ms',
    'median_itl_ms',
    'std_itl_ms',
    'p99_itl_ms',
    'duration',
    'completed',
    'failed',
    'total_input_tokens',
    'total_output_tokens',
    'request_goodput',
    'max_concurrent_requests',
    'rtfx',
]

# Maps a vLLM base key to the equivalent key used by other backends (SGLang)
# when the name differs. Only keys that exist in the vLLM base schema are
# produced; backend-specific extras are dropped.
_RESULT_KEY_ALIASES = {
    'total_token_throughput': 'total_throughput',
    'num_prompts': 'completed',
}

def normalize_bench_result(result):
    """Normalize a single benchmark result line onto the vLLM schema.

    vLLM results already match the base schema and pass through with their
    values intact. SGLang results use a few different key names (and omit some
    vLLM-only fields); known aliases are remapped and any base key without a
    source value is filled with None so downstream reporting stays uniform. The
    vLLM schema is the base and is never extended with backend-specific fields.
    """
    if result is None:
        return None
    normalized = {}
    for key in _VLLM_RESULT_KEYS:
        if key in result:
            normalized[key] = result[key]
        elif key in _RESULT_KEY_ALIASES and _RESULT_KEY_ALIASES[key] in result:
            normalized[key] = result[_RESULT_KEY_ALIASES[key]]
        else:
            normalized[key] = None
    return normalized

def vllm_bench_report(model, model_name, batches, all_results):
  columns_mapping = {
    'num_prompts': 'Number of Prompts',
    'request_throughput': 'Request Throughput',
    'output_throughput': 'Output Throughput',
    'total_token_throughput': 'Total Token Throughput',
    'max_output_tokens_per_s': 'Max Output Tokens Per Second',
    'mean_ttft_ms': 'Mean TTFT (ms)',
    'median_ttft_ms': 'Median TTFT (ms)',
    'std_ttft_ms': 'Std TTFT (ms)',
    'p99_ttft_ms': 'P99 TTFT (ms)',
    'mean_tpot_ms': 'Mean TPOT (ms)',
    'median_tpot_ms': 'Median TPOT (ms)',
    'std_tpot_ms': 'Std TPOT (ms)',
    'p99_tpot_ms': 'P99 TPOT (ms)',
    'mean_itl_ms': 'Mean ITL (ms)',
    'median_itl_ms': 'Median ITL (ms)',
    'std_itl_ms': 'Std ITL (ms)',
    'p99_itl_ms': 'P99 ITL (ms)',
    'duration': 'Duration (s)',
    'completed': 'Completed',
    'failed': 'Failed',
    'total_input_tokens': 'Total Input Tokens',
    'total_output_tokens': 'Total Output Tokens',
    'request_goodput': 'Request Goodput',
    'max_concurrent_requests': 'Max Concurrent Requests',
    'rtfx': 'RTFX',
  }
  workbook_path = None
  try:
    import openpyxl
    from openpyxl.chart import LineChart, Reference, Series
    from openpyxl.chart.series import SeriesLabel
    from openpyxl.chart.layout import Layout, ManualLayout
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    main_sheet = wb.active
    benchmark_sheet = wb.create_sheet("Benchmark")

    benchmark_sheet.append(["Benchmark"])

    offset_col = 2
    offset_stat_row = benchmark_sheet.max_row + 1
    # Making columns header
    benchmark_sheet.append(['Batch', *list(columns_mapping.values())])
    benchmark_table = [benchmark_sheet.max_row + 1, benchmark_sheet.max_row + len(batches) + 1]
    # Making rows data
    for batch in batches:
        # Normalize each result line onto the vLLM schema before reporting
        result = normalize_bench_result(all_results[batch])
        # Skip empty results
        if result is None:
            print(f'{{ "Warning": "No results for batch {batch}" }},')
            continue
        benchmark_sheet.append([batch, *[result[key] for key in columns_mapping.keys()]])
    
    columns = list(columns_mapping.keys())
    for idx, key in enumerate(columns):
        chart = LineChart()
        chart.title = columns_mapping[key]
        chart.x_axis.title = "Batch"
        chart.y_axis.title = ""
        #chart.y_axis.scaling.min = 0
        #chart.y_axis.scaling.max = x_axis_max
        chart.x_axis.delete = False
        chart.y_axis.delete = False
        series = Series(values=Reference(benchmark_sheet, min_col=offset_col + idx, min_row=benchmark_table[0], max_col=offset_col + idx, max_row=benchmark_table[1]), title=f"{columns[idx]}")
        series.marker.symbol = "circle"
        series.marker.size = 6
        chart.series.append(series)
        batch_titles = Reference(benchmark_sheet, min_col=offset_col - 1, min_row=benchmark_table[0], max_col=offset_col - 1, max_row=benchmark_table[1])
        chart.set_categories(batch_titles)
        chart.legend.position = 'b'
        chart.layout=Layout(
            manualLayout=ManualLayout(
                x=0.02, y=0.02,
                h=0.65, w=0.9,
            )
        )
        chart.width = 25
        benchmark_sheet.add_chart(chart, get_column_letter(offset_col) + str(benchmark_sheet.max_row + 1 + idx * 16))


    report_datetime = datetime.datetime.now()
    main_sheet.title = "Overview"
    main_sheet.column_dimensions[get_column_letter(1)].width = 30
    main_sheet.append(['Model:', model_name])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Description:', str(model)])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Run Command:', ' '.join(sys.argv)])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)
    main_sheet.append(['Report Date:', report_datetime.strftime('%Y-%m-%d %H:%M:%S')])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=6)
    main_sheet.append(['Batches:', *batches])
    main_sheet.append(['Total Inference Runs:', model.total_inference_runs])
    main_sheet.append([])
    main_sheet.append(['System Information:'])
    try:
        main_sheet.append(['Hostname:', platform.node()])
        main_sheet.append(['OS:', platform.system()])
        main_sheet.append(['OS Version:', platform.version()])
        main_sheet.append(['OS Release:', platform.release()])
    except Exception as e:
        main_sheet.append([f'Cannot get OS information {e}'])
    main_sheet.append(['Python Version:', sys.version])
    main_sheet.merge_cells(start_row=main_sheet.max_row, start_column=2, end_row=main_sheet.max_row, end_column=10)

    try:
        main_sheet.append(['CPU:', platform.processor()])
        accelerators = enumerate_accelerators()
        for item in accelerators['gpu']:
            main_sheet.append(['GPU:', item['name']])
        for item in accelerators['npu']:
            main_sheet.append(['NPU:', item['name']])
    except Exception as e:
        main_sheet.append([f'Cannot get accelerators information {e}'])

    try:
        result = subprocess.run(
            ['pip', 'list', '--format', 'columns'],
            capture_output=True,
            text=True
        )
        output = result.stdout.split('\n')
        for item in output:
            main_sheet.append(item.split())
    except Exception as e:
        main_sheet.append([f'Cannot get PIP list {e}'])

    try:
        main_sheet.append(['Loaded Modules:'])
        for item in sorted(list_loaded_modules()['modules'], key=lambda x: x['name']):
            main_sheet.append([item['name'], item['path']])
    except Exception as e:
        main_sheet.append([f'Cannot get loaded modules {e}'])

    try:
        main_sheet.append([])
        main_sheet.append(['Environment Variables:'])
        for key, value in sorted(os.environ.items(), key=lambda x: x[0]):
            main_sheet.append([key, value])
    except Exception as e:
        main_sheet.append([f'Cannot get environment variables {e}'])

    workbook_path = f"{platform.node().lower()}_" + model_name.replace('/', '_').replace('\\', '_') + f"_{report_datetime.strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb.save(workbook_path)

    reports_path = os.path.join(os.path.dirname(__file__), 'reports', report_datetime.strftime("%Y%m%d"))
    if not os.path.exists(reports_path):
      os.makedirs(reports_path)
      # Copying statistics aggregator to a reports folder
      try:
        copy(os.path.join(os.path.dirname(__file__), "!StatViewer.xlsm"), os.path.join(reports_path, "!StatViewer.xlsm"))
      except Exception as e:
        print(f'{{ "Error": "Failed to copy !StatViewer.xlsm {e}" }}')
    os.rename(workbook_path, os.path.join(reports_path, workbook_path))

    print('{ "Workbook": "' + os.path.join(reports_path, workbook_path).replace("\\", "/") + '" },')

  except Exception as e:
    print(f'{{ "Error": "Failed to load openpyxl {e}" }},')
    print(traceback.format_exc())
  return workbook_path

HTML_REPORT_CSS = r"""
:root{
  --bg:#0c0c0e; --panel:#17171b; --panel2:#1e1e24; --border:#2c2c34;
  --text:#e9e9ec; --muted:#9b9ba4; --accent:#e23c3c; --accent-d:#8f1d1d;
  --amber:#f0a500; --green:#5cc66a;
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{
  margin:0;
  background:
    radial-gradient(1200px 620px at 12% -12%, rgba(226,60,60,0.10), transparent 60%),
    radial-gradient(900px 520px at 100% -6%, rgba(240,165,0,0.06), transparent 55%),
    var(--bg);
  color:var(--text);
  font-family:"Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55; font-size:15px; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:48px 28px 90px;}
a{color:var(--accent); text-decoration:none;}
a:hover{color:var(--amber); text-decoration:underline;}
::selection{background:rgba(226,60,60,0.35);}

.hero{position:relative; overflow:hidden; border:1px solid var(--border); border-radius:16px;
  background:linear-gradient(160deg,#1b1b20,#121215); padding:40px 42px 34px;}
.hero::before{content:""; position:absolute; left:0; top:0; bottom:0; width:6px;
  background:linear-gradient(180deg,var(--accent),var(--accent-d));}
.hero-kicker{letter-spacing:.28em; text-transform:uppercase; font-size:12px; font-weight:600;
  color:var(--amber); margin-bottom:12px;}
.hero-title{margin:0; font-size:40px; font-weight:700; letter-spacing:-.5px;
  background:linear-gradient(90deg,#ffffff,#c9c9cf); -webkit-background-clip:text;
  background-clip:text; color:transparent;}
.hero-sub{color:var(--muted); margin-top:8px; font-size:15px; max-width:820px;}
.hero-meta{display:flex; flex-wrap:wrap; gap:14px 34px; margin-top:24px;}
.hero-meta div{font-size:14px;}
.hero-meta span{display:block; color:var(--muted); font-size:11px; text-transform:uppercase;
  letter-spacing:.12em; margin-bottom:3px;}

section{margin-top:46px;}
.sec-head{display:flex; align-items:center; gap:14px; margin:0 0 20px;}
.sec-num{flex:0 0 auto; width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
  font-weight:700; font-size:15px; color:#fff;
  background:linear-gradient(160deg,var(--accent),var(--accent-d)); box-shadow:0 4px 14px rgba(226,60,60,0.35);}
.sec-head h2{margin:0; font-size:23px; font-weight:650; letter-spacing:-.2px;}
.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:20px 24px;}

.toc ol{margin:0; padding:0; list-style:none; counter-reset:toc 2;}
.toc li{counter-increment:toc; border-bottom:1px solid var(--border);}
.toc li:last-child{border-bottom:none;}
.toc li a{display:flex; align-items:center; gap:16px; padding:13px 6px; color:var(--text); font-size:16px;}
.toc li a::before{content:counter(toc,decimal-leading-zero); color:var(--accent); font-weight:700;
  font-size:14px; min-width:30px; font-variant-numeric:tabular-nums;}
.toc li a:hover{color:var(--amber); text-decoration:none;}
.toc li a:hover::before{color:var(--amber);}

.table-wrap{overflow-x:auto; border:1px solid var(--border); border-radius:14px;}
table.data{border-collapse:collapse; width:100%; font-size:13.5px; min-width:520px;}
table.data thead th{position:sticky; top:0; z-index:1; background:linear-gradient(180deg,#241417,#180d0f);
  color:#f3d9d9; text-align:right; font-weight:600; padding:12px 16px; white-space:nowrap;
  border-bottom:2px solid var(--accent-d);}
table.data thead th:first-child{text-align:left;}
table.data tbody th{text-align:left; font-weight:500; color:var(--text); padding:10px 16px;
  white-space:nowrap; background:var(--panel);}
table.data td{text-align:right; padding:10px 16px; color:var(--text); font-variant-numeric:tabular-nums; white-space:nowrap;}
table.data tbody tr:nth-child(odd) td, table.data tbody tr:nth-child(odd) th{background:var(--panel2);}
table.data tbody tr:hover td, table.data tbody tr:hover th{background:#26191b;}
table.data.lt thead th, table.data.lt tbody th{text-align:left;}
table.data.lt td{text-align:left; white-space:normal; word-break:break-word;}
table.data.lt tbody th{font-family:"Cascadia Code","Consolas",monospace; font-size:12.5px; color:var(--amber);}

table.kv{border-collapse:collapse; width:100%; font-size:13.5px;}
table.kv th{text-align:left; vertical-align:top; color:var(--muted); font-weight:500;
  padding:8px 18px 8px 0; white-space:nowrap; width:230px;}
table.kv td{padding:8px 0; color:var(--text); word-break:break-word;}
table.kv tr+tr th, table.kv tr+tr td{border-top:1px solid var(--border);}

.charts{display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px;}
figure.chart{margin:0; background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:8px; overflow:hidden;}
figure.chart img{display:block; width:100%; height:auto; border-radius:8px;}

details.env{background:var(--panel); border:1px solid var(--border); border-radius:12px; margin-top:14px; overflow:hidden;}
details.env>summary{cursor:pointer; list-style:none; padding:14px 20px; font-weight:600; font-size:15px;
  color:var(--text); display:flex; align-items:center; justify-content:space-between;}
details.env>summary::-webkit-details-marker{display:none;}
details.env>summary::after{content:"+"; color:var(--accent); font-size:20px; line-height:1;}
details.env[open]>summary::after{content:"\2013";}
details.env>summary:hover{color:var(--amber);}
.env-body{padding:2px 20px 20px; max-height:470px; overflow:auto;}
.env-body .table-wrap{border:none; border-radius:0;}

.subhead{margin:26px 0 12px; font-size:13px; color:var(--amber); text-transform:uppercase;
  letter-spacing:.12em; font-weight:600;}
.subhead:first-child{margin-top:6px;}

.foot{margin-top:60px; padding-top:22px; border-top:1px solid var(--border); color:var(--muted);
  font-size:12.5px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; align-items:center;}
.badge{display:inline-block; padding:3px 11px; border:1px solid var(--accent-d); border-radius:999px;
  color:var(--accent); font-size:11px; letter-spacing:.06em; text-transform:uppercase;}

*{scrollbar-color:var(--accent-d) transparent;}
::-webkit-scrollbar{height:10px; width:10px;}
::-webkit-scrollbar-thumb{background:var(--accent-d); border-radius:10px;}
::-webkit-scrollbar-track{background:transparent;}
"""


def vllm_bench_report_html(model, model_name, batches, all_results):
    columns_mapping = {
        'num_prompts': 'Number of Prompts',
        'request_throughput': 'Request Throughput',
        'output_throughput': 'Output Throughput',
        'total_token_throughput': 'Total Token Throughput',
        'max_output_tokens_per_s': 'Max Output Tokens Per Second',
        'mean_ttft_ms': 'Mean TTFT (ms)',
        'median_ttft_ms': 'Median TTFT (ms)',
        'std_ttft_ms': 'Std TTFT (ms)',
        'p99_ttft_ms': 'P99 TTFT (ms)',
        'mean_tpot_ms': 'Mean TPOT (ms)',
        'median_tpot_ms': 'Median TPOT (ms)',
        'std_tpot_ms': 'Std TPOT (ms)',
        'p99_tpot_ms': 'P99 TPOT (ms)',
        'mean_itl_ms': 'Mean ITL (ms)',
        'median_itl_ms': 'Median ITL (ms)',
        'std_itl_ms': 'Std ITL (ms)',
        'p99_itl_ms': 'P99 ITL (ms)',
        'duration': 'Duration (s)',
        'completed': 'Completed',
        'failed': 'Failed',
        'total_input_tokens': 'Total Input Tokens',
        'total_output_tokens': 'Total Output Tokens',
        'request_goodput': 'Request Goodput',
        'max_concurrent_requests': 'Max Concurrent Requests',
        'rtfx': 'RTFX',
    }
    report_path = None
    try:
        import io
        import math
        import base64
        import html as _html
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # AMD-inspired dark palette (no blue accents anywhere)
        C = {
            'panel':   '#17171b',
            'panel2':  '#1e1e24',
            'border':  '#2c2c34',
            'text':    '#e9e9ec',
            'muted':   '#9b9ba4',
            'accent':  '#e23c3c',
            'accent_d':'#8f1d1d',
            'amber':   '#f0a500',
        }

        report_datetime = datetime.datetime.now()

        def esc(v):
            return _html.escape('' if v is None else str(v))

        def fmt(v):
            if v is None:
                return '&mdash;'
            if isinstance(v, bool):
                return 'Yes' if v else 'No'
            if isinstance(v, int):
                return f'{v:,}'
            if isinstance(v, float):
                if math.isnan(v):
                    return '&mdash;'
                av = abs(v)
                if av == 0:
                    return '0'
                if av >= 1000:
                    return f'{v:,.1f}'
                if av >= 1:
                    return f'{v:,.2f}'
                return f'{v:.4f}'
            return esc(v)

        # Normalize each result line onto the vLLM schema before reporting
        all_results = {b: normalize_bench_result(all_results.get(b)) for b in batches}

        valid_batches = [b for b in batches if all_results.get(b) is not None]
        for b in batches:
            if all_results.get(b) is None:
                print(f'{{ "Warning": "No results for batch {b}" }},')

        # ---------- Charts (matplotlib -> base64 PNG data URIs) ----------
        def make_chart_datauri(key, label):
            ys = []
            for b in valid_batches:
                val = all_results[b].get(key)
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    ys.append(float('nan'))
                else:
                    ys.append(float(val))
            if not ys or all(math.isnan(y) for y in ys):
                return None
            xs = [str(b) for b in valid_batches]
            fig, ax = plt.subplots(figsize=(4.9, 3.05), dpi=110)
            fig.patch.set_facecolor(C['panel'])
            ax.set_facecolor(C['panel2'])
            ax.plot(range(len(xs)), ys, color=C['accent'], linewidth=2.2, marker='o',
                    markersize=6, markerfacecolor=C['amber'], markeredgecolor=C['accent'], zorder=3)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs)
            ax.set_xlabel('Batch', color=C['muted'], fontsize=9)
            ax.set_title(label, color=C['text'], fontsize=10.5, pad=10, fontweight='bold')
            ax.grid(True, color=C['border'], linewidth=0.8, alpha=0.7)
            ax.margins(x=0.06, y=0.16)
            for spine in ax.spines.values():
                spine.set_color(C['border'])
            ax.tick_params(colors=C['muted'], labelsize=8)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
            plt.close(fig)
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

        # ---------- HTML building helpers ----------
        def kv_table(rows):
            body = ''.join(f'<tr><th>{esc(k)}</th><td>{v}</td></tr>' for k, v in rows)
            return f'<table class="kv">{body}</table>'

        def two_col_table(h1, h2, rows):
            if not rows:
                rows = [('&mdash;', '')]
            body = ''.join(f'<tr><th>{esc(a)}</th><td>{esc(b)}</td></tr>' for a, b in rows)
            return (f'<div class="table-wrap"><table class="data lt"><thead><tr>'
                    f'<th>{esc(h1)}</th><th>{esc(h2)}</th></tr></thead><tbody>{body}</tbody></table></div>')

        def details_block(title, inner, is_open=False):
            attr = ' open' if is_open else ''
            return (f'<details class="env"{attr}><summary>{esc(title)}</summary>'
                    f'<div class="env-body">{inner}</div></details>')

        # ---------- 3. Benchmarking results table (metrics x batches) ----------
        th_cols = ''.join(f'<th>Batch {esc(b)}</th>' for b in valid_batches)
        result_rows = []
        for key, label in columns_mapping.items():
            cells = ''.join(f'<td>{fmt(all_results[b].get(key))}</td>' for b in valid_batches)
            result_rows.append(f'<tr><th>{esc(label)}</th>{cells}</tr>')
        if valid_batches:
            results_table = (f'<div class="table-wrap"><table class="data"><thead><tr>'
                             f'<th>Metric</th>{th_cols}</tr></thead><tbody>'
                             f'{"".join(result_rows)}</tbody></table></div>')
        else:
            results_table = '<div class="card">No benchmark results available.</div>'

        # ---------- 4. Charts ----------
        chart_figs = []
        for key, label in columns_mapping.items():
            uri = make_chart_datauri(key, label)
            if uri:
                chart_figs.append(f'<figure class="chart"><img alt="{esc(label)}" src="{uri}"></figure>')
        charts_section = (f'<div class="charts">{"".join(chart_figs)}</div>' if chart_figs
                          else '<div class="card">No numeric data available for charts.</div>')

        # ---------- 5. Environment settings ----------
        overview_rows = [
            ('Model', esc(model_name)),
            ('Description', esc(str(model))),
            ('Run Command', esc(' '.join(sys.argv))),
            ('Report Date', esc(report_datetime.strftime('%Y-%m-%d %H:%M:%S'))),
            ('Batches', esc(', '.join(str(b) for b in batches))),
            ('Total Inference Runs', fmt(getattr(model, 'total_inference_runs', None))),
        ]

        system_rows = []
        try:
            system_rows.append(('Hostname', esc(platform.node())))
            system_rows.append(('OS', esc(platform.system())))
            system_rows.append(('OS Version', esc(platform.version())))
            system_rows.append(('OS Release', esc(platform.release())))
        except Exception as e:
            system_rows.append(('OS Information Error', esc(e)))
        system_rows.append(('Python Version', esc(sys.version)))
        try:
            system_rows.append(('CPU', esc(platform.processor())))
        except Exception as e:
            system_rows.append(('CPU Error', esc(e)))

        accel_rows = []
        try:
            accelerators = enumerate_accelerators()
            for item in accelerators.get('gpu', []):
                accel_rows.append(('GPU', item.get('name')))
            for item in accelerators.get('npu', []):
                accel_rows.append(('NPU', item.get('name')))
        except Exception as e:
            accel_rows.append(('Accelerators Error', str(e)))

        pkg_rows = []
        try:
            result = subprocess.run(['pip', 'list', '--format', 'columns'], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                if set(line.strip()) <= set('- '):
                    continue
                tokens = line.split()
                if tokens[0].lower() == 'package' and len(tokens) > 1 and tokens[1].lower() == 'version':
                    continue
                name = tokens[0]
                version = tokens[1] if len(tokens) > 1 else ''
                pkg_rows.append((name, version))
        except Exception as e:
            pkg_rows.append(('Error', str(e)))

        module_rows = []
        try:
            for item in sorted(list_loaded_modules()['modules'], key=lambda x: x['name']):
                module_rows.append((item.get('name'), item.get('path')))
        except Exception as e:
            module_rows.append(('Error', str(e)))

        env_rows = []
        try:
            for key, value in sorted(os.environ.items(), key=lambda x: x[0]):
                env_rows.append((key, value))
        except Exception as e:
            env_rows.append(('Error', str(e)))

        env_parts = []
        env_parts.append('<h3 class="subhead">Overview</h3>')
        env_parts.append(f'<div class="card">{kv_table(overview_rows)}</div>')
        env_parts.append('<h3 class="subhead">System</h3>')
        env_parts.append(f'<div class="card">{kv_table(system_rows)}</div>')
        if accel_rows:
            env_parts.append('<h3 class="subhead">Accelerators</h3>')
            env_parts.append(two_col_table('Type', 'Device', accel_rows))
        env_parts.append(details_block(f'Installed Packages ({len(pkg_rows)})',
                                       two_col_table('Package', 'Version', pkg_rows)))
        env_parts.append(details_block(f'Loaded Modules ({len(module_rows)})',
                                       two_col_table('Module', 'Path', module_rows)))
        env_parts.append(details_block(f'Environment Variables ({len(env_rows)})',
                                       two_col_table('Variable', 'Value', env_rows)))
        environment_section = ''.join(env_parts)

        # ---------- Assemble page ----------
        parts = []
        parts.append('<!DOCTYPE html>')
        parts.append('<html lang="en"><head><meta charset="utf-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
        parts.append(f'<title>vLLM Benchmark Report &middot; {esc(model_name)}</title>')
        parts.append(f'<style>{HTML_REPORT_CSS}</style></head><body><div class="wrap">')

        # 1. Title
        parts.append(
            '<header class="hero" id="top">'
            '<div class="hero-kicker">vLLM Benchmark Report</div>'
            f'<h1 class="hero-title">{esc(model_name)}</h1>'
            f'<div class="hero-sub">{esc(str(model))}</div>'
            '<div class="hero-meta">'
            f'<div><span>Report Date</span>{esc(report_datetime.strftime("%Y-%m-%d %H:%M:%S"))}</div>'
            f'<div><span>Host</span>{esc(platform.node())}</div>'
            f'<div><span>Batches</span>{esc(", ".join(str(b) for b in batches))}</div>'
            f'<div><span>Total Inference Runs</span>{fmt(getattr(model, "total_inference_runs", None))}</div>'
            '</div></header>'
        )

        # 2. Table of contents
        parts.append(
            '<section id="toc">'
            '<div class="sec-head"><div class="sec-num">2</div><h2>Table of Contents</h2></div>'
            '<nav class="card toc"><ol>'
            '<li><a href="#results">Benchmarking Results</a></li>'
            '<li><a href="#charts">Charts</a></li>'
            '<li><a href="#environment">Environment Settings</a></li>'
            '</ol></nav></section>'
        )

        # 3. Benchmarking results
        parts.append(
            '<section id="results">'
            '<div class="sec-head"><div class="sec-num">3</div><h2>Benchmarking Results</h2></div>'
            f'{results_table}</section>'
        )

        # 4. Charts
        parts.append(
            '<section id="charts">'
            '<div class="sec-head"><div class="sec-num">4</div><h2>Charts</h2></div>'
            f'{charts_section}</section>'
        )

        # 5. Environment settings
        parts.append(
            '<section id="environment">'
            '<div class="sec-head"><div class="sec-num">5</div><h2>Environment Settings</h2></div>'
            f'{environment_section}</section>'
        )

        parts.append(
            '<div class="foot">'
            f'<span>Generated {esc(report_datetime.strftime("%Y-%m-%d %H:%M:%S"))} on {esc(platform.node())}</span>'
            '<span class="badge">vLLM Performance</span>'
            '</div>'
        )
        parts.append('</div></body></html>')

        html_doc = ''.join(parts)

        report_path = f"{platform.node().lower()}_" + model_name.replace('/', '_').replace('\\', '_') + f"_{report_datetime.strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)

        reports_path = os.path.join(os.path.dirname(__file__), 'reports', report_datetime.strftime("%Y%m%d"))
        if not os.path.exists(reports_path):
            os.makedirs(reports_path)
        final_path = os.path.join(reports_path, report_path)
        os.replace(report_path, final_path)
        report_path = final_path

        print('{ "HtmlReport": "' + report_path.replace("\\", "/") + '" },')

    except Exception as e:
        print(f'{{ "Error": "Failed to build HTML report {e}" }},')
        print(traceback.format_exc())
    return report_path
