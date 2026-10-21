"""Run per-camera WAPR inference in persistent worker processes.

Each worker owns its CUDA renderer and estimator on the already-selected GPU.
Each camera scores its three center-update candidates as an independent group.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


_backend = None


def _initialize(args):
    global _backend
    try:
        from .wapr_tracking import WAPRBackend
    except ImportError:
        from wapr_tracking import WAPRBackend
    _backend = WAPRBackend(args)


def _ready():
    return True


def _estimate(task):
    observation, previous_world, center_world = task
    return _backend.estimate(observation, previous_world, center_world)


class ParallelViews:
    def __init__(self, args):
        worker_args = copy.copy(args)
        worker_args.view_workers = 1
        worker_args.cpu_threads = min(args.cpu_threads, 2)
        self.executor = ProcessPoolExecutor(
            max_workers=args.view_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(worker_args,))
        # Load resident workers while the parent registers the first frame.
        self.startup = [self.executor.submit(_ready) for _ in range(args.view_workers)]

    def estimate(self, observations, previous_world, center_world):
        # Masks/center pixels are used in the parent for selection/triangulation,
        # but are not inputs to the official no-mask temporal tracker.
        fields = ("rgb", "depth", "K", "ext", "cam", "frame", "area")
        tasks = [({key: obs[key] for key in fields}, previous_world, center_world) for obs in observations]
        return list(self.executor.map(_estimate, tasks))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
