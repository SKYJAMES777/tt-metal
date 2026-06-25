# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Segmented ttnn-trace capture across sub-device-manager swaps.

Problem: ttnn forbids loading/clearing a sub-device manager *inside* a trace capture
(`begin_trace_capture`..`end_trace_capture`) — it resets worker state. But the MoE wants to keep the
shared-expert/dispatch overlap, which loads a 2-sub-device manager around that region and clears it
after. So a single trace cannot span the whole forward.

Solution: capture the forward as *several* traces, split exactly at the load/clear points, and perform
the load/clear on the host *between* the trace segments (legal — we are not capturing at that instant).
At replay we walk the recorded program, executing each trace segment and doing the load/clear between
them. Validated bit-exact in scratch PoC (3 chained traces, load/clear between captures and replays).

Usage (capture once, replay many) — the forward runs ONCE during capture; this controller chops it:

    controller = SubDeviceTraceController(mesh_device)
    transformer.set_trace_controller(controller)   # MoE consults it for load/clear
    transformer.forward(...)                         # WARMUP (controller idle) -> compiles programs
    controller.begin_capture()
    transformer.forward(...)                         # captured + auto-split at each load/clear
    controller.end_capture()
    ...                                              # controller.trace_bytes() for memory
    for _ in range(n): controller.replay()           # each replay = full segmented forward
    controller.release()
    transformer.set_trace_controller(None)

Boundary tensors (produced in one segment, read by the next) stay valid because the forward keeps them
referenced across the end/begin split; trace replay reproduces them at the same addresses.
"""

import ttnn


class SubDeviceTraceController:
    """Drives multi-segment trace capture/replay around sub-device-manager load/clear calls.

    MoE.forward calls `sub_device_load(mgr_id)` / `sub_device_clear()` instead of touching the mesh
    device directly. This object decides what those mean based on its mode:
      - idle (not capturing): pass straight through to the real device load/clear (eager behavior).
      - capturing: end the in-progress trace, do the real host load/clear, begin the next trace —
        i.e. split the capture here and record the boundary action.
    """

    # Program step kinds.
    _TRACE = "trace"
    _LOAD = "load"
    _CLEAR = "clear"

    def __init__(self, mesh_device, cq_id=0):
        self.mesh_device = mesh_device
        self.cq_id = cq_id
        self._program = []  # ordered list of (kind, payload): (_TRACE, tid) | (_LOAD, mgr_id) | (_CLEAR, None)
        self._current_tid = None
        self._capturing = False

    # ------------------------------------------------------------------ capture
    def begin_capture(self):
        """Open the first trace segment. The next forward() will be recorded and auto-split."""
        assert not self._capturing, "already capturing"
        self._program = []
        self._capturing = True
        self._current_tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)

    def end_capture(self):
        """Close the final trace segment."""
        assert self._capturing, "begin_capture() was not called"
        ttnn.end_trace_capture(self.mesh_device, self._current_tid, cq_id=self.cq_id)
        self._program.append((self._TRACE, self._current_tid))
        self._current_tid = None
        self._capturing = False

    # ----------------------------------------------------- MoE-facing hooks
    def sub_device_load(self, sd_manager_id):
        """MoE entered the overlap region. Capturing -> split + record + real load; else just load."""
        if self._capturing:
            self._split(self._LOAD, sd_manager_id)
        else:
            self.mesh_device.load_sub_device_manager(sd_manager_id)

    def sub_device_clear(self):
        """MoE left the overlap region. Capturing -> split + record + real clear; else just clear."""
        if self._capturing:
            self._split(self._CLEAR, None)
        else:
            self.mesh_device.clear_loaded_sub_device_manager()

    def _split(self, kind, payload):
        # Close the current segment, perform the real host action, open the next segment.
        ttnn.end_trace_capture(self.mesh_device, self._current_tid, cq_id=self.cq_id)
        self._program.append((self._TRACE, self._current_tid))
        if kind == self._LOAD:
            self.mesh_device.load_sub_device_manager(payload)
        else:
            self.mesh_device.clear_loaded_sub_device_manager()
        self._program.append((kind, payload))
        self._current_tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)

    # ------------------------------------------------------------------ replay
    def replay(self):
        """Run the whole segmented forward: execute each trace, load/clear between segments."""
        assert not self._capturing, "still capturing"
        assert self._program, "nothing captured"
        for kind, payload in self._program:
            if kind == self._TRACE:
                ttnn.execute_trace(self.mesh_device, payload, cq_id=self.cq_id, blocking=True)
            elif kind == self._LOAD:
                self.mesh_device.load_sub_device_manager(payload)
            else:  # _CLEAR
                self.mesh_device.clear_loaded_sub_device_manager()

    # ------------------------------------------------------------------ stats / cleanup
    @property
    def num_segments(self):
        return sum(1 for kind, _ in self._program if kind == self._TRACE)

    def trace_bytes(self):
        """Total device memory used by all captured trace segments (bytes per device)."""
        mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        return mv.total_bytes_allocated_per_bank * mv.num_banks

    def release(self):
        """Release every captured trace. Safe to call repeatedly."""
        for kind, payload in self._program:
            if kind == self._TRACE:
                ttnn.release_trace(self.mesh_device, payload)
        self._program = []
