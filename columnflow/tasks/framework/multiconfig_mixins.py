# coding: utf-8

"""
mixins task which inherit from MultiConfigTask.
"""

from __future__ import annotations

import gc
import time
import itertools
from collections import Counter

import luigi
import law
import order as od

from columnflow.types import Sequence, Any, Iterable
from columnflow.tasks.framework.base import AnalysisTask, MultiConfigTask, RESOLVE_DEFAULT
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.calibration import Calibrator
from columnflow.selection import Selector
from columnflow.production import Producer
from columnflow.weight import WeightProducer
from columnflow.ml import MLModel
from columnflow.inference import InferenceModel
from columnflow.columnar_util import Route, ColumnCollection, ChunkedIOHandler
from columnflow.util import maybe_import, DotDict

ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


class MultiConfigDatasetsProcessesMixin(MultiConfigTask):

    datasets = law.CSVParameter(
        default=(),
        description="comma-separated dataset names or patters to select; can also be the key of a "
        "mapping defined in the 'dataset_groups' auxiliary data of the config; when empty, uses "
        "all datasets registered in the config that contain any of the selected --processes; empty "
        "default",
        brace_expand=True,
        parse_empty=True,
    )
    processes = law.CSVParameter(
        default=(),
        description="comma-separated process names or patterns for filtering processes; can also "
        "be the key of a mapping defined in the 'process_groups' auxiliary data of the config; "
        "uses all processes of the config when empty; empty default",
        brace_expand=True,
        parse_empty=True,
    )

    allow_empty_datasets = False
    allow_empty_processes = False

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        if "config_insts" not in params:
            return params
        config_insts = params["config_insts"]

        # resolve processes
        all_processes = set()
        all_process_insts = set()
        if "processes" in params:

            for config_inst in config_insts:
                if params["processes"]:
                    config_processes = cls.find_config_objects(
                        params["processes"],
                        config_inst,
                        od.Process,
                        config_inst.x("process_groups", {}),
                        deep=True,
                    )
                else:
                    config_processes = config_inst.processes.names()

                all_processes |= set(config_processes)
                all_process_insts |= {config_inst.get_process(p) for p in config_processes}

            # complain when no processes were found
            if not all_processes and not cls.allow_empty_processes:
                raise ValueError(f"no processes found matching {params['processes']}")

            params["processes"] = tuple(all_processes)
            params["process_insts"] = tuple(all_process_insts)

        # resolve datasets
        all_datasets = set()
        all_dataset_insts = set()
        if "datasets" in params:

            for config_inst in config_insts:
                if params["datasets"]:
                    config_datasets = cls.find_config_objects(
                        params["datasets"],
                        config_inst,
                        od.Dataset,
                        config_inst.x("dataset_groups", {}),
                    )
                elif "processes" in params:
                    # pick all datasets that contain any of the requested (sub) processes
                    sub_process_insts = sum((
                        [proc for proc, _, _ in process_inst.walk_processes(include_self=True)]
                        for process_inst in map(config_inst.get_process, params["processes"])
                    ), [])
                    config_datasets = [
                        dataset_inst.name
                        for dataset_inst in config_inst.datasets
                        if any(map(dataset_inst.has_process, sub_process_insts))
                    ]

                all_datasets |= set(config_datasets)
                all_dataset_insts |= {config_inst.get_dataset(d) for d in config_datasets}

            # complain when no datasets were found
            if not all_datasets and not cls.allow_empty_datasets:
                raise ValueError(f"no datasets found matching {params['datasets']}")

            params["datasets"] = tuple(all_datasets)
            params["dataset_insts"] = tuple(all_dataset_insts)

        return params

    @classmethod
    def get_known_shifts(cls, config_inst, params):
        shifts, upstream_shifts = super().get_known_shifts(config_inst, params)

        # add shifts of all datasets to upstream ones
        for dataset_inst in params.get("dataset_insts") or []:
            if dataset_inst.is_mc:
                upstream_shifts |= set(dataset_inst.info.keys())

        return shifts, upstream_shifts

    @property
    def datasets_repr(self):
        if len(self.datasets) == 1:
            return self.datasets[0]

        return f"{len(self.datasets)}_{law.util.create_hash(sorted(self.datasets))}"

    @property
    def processes_repr(self):
        if len(self.processes) == 1:
            return self.processes[0]

        return f"{len(self.processes)}_{law.util.create_hash(self.processes)}"



class MultiConfigHistHookMixin(MultiConfigTask):

    hist_hooks = law.CSVParameter(
        default=(),
        description="names of functions in the config's auxiliary dictionary 'hist_hooks' that are "
        "invoked before plotting to update a potentially nested dictionary of histograms; "
        "default: empty",
    )

    def invoke_hist_hooks(self, hists: dict) -> dict:
        """
        Invoke hooks to update histograms before plotting.
        """
        if not self.hist_hooks:
            return hists

        for hook in self.hist_hooks:
            if hook in (None, "", law.NO_STR):
                continue

            # get the hook from the config instance
            hooks = self.config_inst.x("hist_hooks", {})
            if hook not in hooks:
                raise KeyError(
                    f"hist hook '{hook}' not found in 'hist_hooks' auxiliary entry of config",
                )
            func = hooks[hook]
            if not callable(func):
                raise TypeError(f"hist hook '{hook}' is not callable: {func}")

            # invoke it
            self.publish_message(f"invoking hist hook '{hook}'")
            hists = func(self, hists)

        return hists