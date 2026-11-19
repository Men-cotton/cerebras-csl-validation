"""Generate only the Direct Link metadata. Tungsten supplies all PE code and routes."""
import argparse
import os
from pathlib import Path

from cerebras.geometry.geometry import IntVector
from cerebras.sdk.runtime.sdkruntimepybind import Color, SdkLayout, SdkTarget, SimfabConfig, get_platform

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
args = p.parse_args()
folder = args.output.resolve()
folder.mkdir(parents=True, exist_ok=False)
os.chdir(folder)
layout = SdkLayout(get_platform(None, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3))
layout.create_input_stream_from_loc(IntVector(0, 0), Color('rx', 0))
layout.create_output_stream_from_loc(IntVector(0, 1), Color('tx', 1))
layout.compile(str(folder/'out'), save_port_map=True)
