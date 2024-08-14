from __future__ import annotations

import luigi
import law
import order as od
from collections import OrderedDict

from columnflow.tasks.framework.base import (
    Requirements, AnalysisTask, DatasetTask,
    wrapper_factory, RESOLVE_DEFAULT, ConfigTask
)
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, VariablesMixin,
    ChunkedIOMixin, DatasetsProcessesMixin, WeightProducerMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase2D,
)

from columnflow.tasks.histograms import CreateHistograms

from columnflow.weight import WeightProducer
from columnflow.production import Producer
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.util import dev_sandbox, dict_add_strict, four_vec, DotDict
from columnflow.types import Any
from columnflow.production.cms.btag import BTagSFConfig

# discontinued and replaced with producer jet_btag that uses
# config.x attribute "default_btagAlgorithm" (algorithm, working point).
# producer jet_btag makes a boolean column Jet.btag


class BTagAlgoritmsMixin(ConfigTask):

    algorithms: list[str] = law.CSVParameter(
        default=(RESOLVE_DEFAULT,),
        description="comma-separated names of algorithms to apply as specified: ALGORITHM-WP(-DISCRIMINATOR). "
                    "The discriminator can be omitted in which case the a default is chosen (see BTagSFConfig) "
                    "default: 'default_btagAlgorithm' config,",
        brace_expand=True,
        parse_empty=True,
    )

    default_variables = ("btag_jet_pt", "btag_jet_eta")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.btag_configs = {algo: self.strtup_to_cfg(algo) for algo in self.algorithms}

    @classmethod
    def resolve_param_values(
        cls,
        params: law.util.InsertableDict[str, Any],
    ) -> law.util.InsertableDict[str, Any]:
        """
        Resolve values *params* and check against possible default values

        Check the values in *params* against the default value ``"default_btagAlgorithm"`` in the current config inst.
        For more information, see
        :py:meth:`~columnflow.tasks.framework.base.ConfigTask.resolve_config_default_and_groups`.

        :param params: Parameter values to resolve
        :return: Dictionary of parameters that contains the list requested
            :py:class:`~columnflow.calibration.Calibrator` instances under the
            keyword ``"calibrator_insts"``. See :py:meth:`~.CalibratorsMixin.get_calibrator_insts`
            for more information.
        """
        redo_default_variables = False
        if "variables" in params:
            # when empty, use the config default
            if not params["variables"]:
                redo_default_variables = True

        params = super().resolve_param_values(params)

        config_inst = params.get("config_inst")
        if not config_inst:
            return params

        if redo_default_variables:
            # when empty, use the config default
            if config_inst.x("default_btag_variables", ()):
                params["variables"] = tuple(config_inst.x.default_btag_variables)
            elif cls.default_variables:
                params["variables"] = tuple(cls.default_variables)
            else:
                raise AssertionError(f"define default btag variables in {cls.__class__} or config {config_inst.name}")

        if "algorithm_insts" not in params and "algorithms" in params:
            algorithms = cls.resolve_config_default_and_groups(
                params,
                params.get("algorithms"),
                container=config_inst,
                default_str="default_btagAlgorithm",
                groups_str="btagAlgorithm_groups",
            )
            btag_configs = []
            for algorithm in algorithms:
                if isinstance(algorithm, (str, tuple)):
                    algorithm = cls.strtup_to_cfg(algorithm)
                else:
                    assert isinstance(algorithm, BTagSFConfig)
                btag_configs.append(algorithm)
            params["algorithm_insts"] = btag_configs
            params["algorithms"] = [cls.cfg_to_str(b_cfg) for b_cfg in btag_configs]
        return params

    @classmethod
    def cfg_to_str(cls, b_cfg: BTagSFConfig):
        return f"{b_cfg.correction_set}-{b_cfg.corrector_kwargs['working_point']}-{b_cfg.discriminator}"

    @classmethod
    def strtup_to_cfg(cls, b_strtup: str | tuple):
        if isinstance(b_strtup, str):
            b_strtup = tuple(b_strtup.split("-"))
        if isinstance(b_strtup, tuple):
            if len(b_strtup) == 2:
                b_strtup = [*b_strtup, None]
            elif len(b_strtup) != 3:
                raise AssertionError(f"{b_strtup} should be of form ALGO-WP(-DISCRIMINATOR)")
        else:
            raise AssertionError(f"expected string or tuple but got {b_strtup}")

        return BTagSFConfig(
            correction_set=b_strtup[0],
            jec_sources=[],
            discriminator=b_strtup[2],
            corrector_kwargs=dict(working_point=b_strtup[1])
        )

    @property
    def algorithms_repr(self):
        if len(self.algorithms) == 1:
            return self.algorithms[0]
        return f"{len(self.algorithms)}_{law.util.create_hash(sorted(self.algorithms))}"


class CreateBTagEfficiencyHistograms(
    BTagAlgoritmsMixin,
    CreateHistograms
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        b_prod_class = WeightProducer.get_cls("fixed_wp_btag_weights")
        b_prod_inst_dct = self.get_producer_kwargs(self) | dict(add_weights=False)
        self.jet_btag_producers: list[WeightProducer] = [b_prod_class.derive(
            cls_name=b_cfg_name,
            cls_dict=dict(add_weights=False, btag_config=self.btag_configs[b_cfg_name], name=self.cfg_to_str)
            )(inst_dict=b_prod_inst_dct) for b_cfg_name in self.btag_configs]

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["jet_btag"] = self.jet_btag_producers[0].run_requires()
        return reqs

    def requires(self):
        reqs = super().requires()
        # requirements don't depend on btag config
        reqs["jet_btag"] = self.jet_btag_producers[0].run_requires()
        return reqs

    def output(self):
        return {
            "hists": self.target(
                f"histograms__algo_{self.algorithms_repr}__var_{self.variables_repr}__{self.branch}.pickle"
            ),
        }

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import Route, update_ak_array, has_ak_column

        # prepare inputs and outputs
        inputs = self.input()
        variable_insts = list(map(self.config_inst.get_variable, self.variables))

        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # run the weight_producer setup
        producer_reqs = self.weight_producer_inst.run_requires()
        reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

        # run the jet tagging setup
        for jet_btag_producer in self.jet_btag_producers:
            jet_btag_producer.run_setup(
                self.requires()["jet_btag"],
                self.input()["jet_btag"],  # input does not matter
            )

        # declare output: dict of histograms
        histograms = {}
        h = (
            hist.Hist.new
            .IntCat([0, 4, 5], name="hadronFlavour")  # Jet hadronFlavour can only be 0, 4, or 5
        )

        # add variables for binning the efficiency
        for var_inst in variable_insts:

            h = h.Var(
                var_inst.bin_edges,
                name=var_inst.name,
                label=var_inst.get_full_x_title(),
            )

        # add two histograms (for numerator and denominator)
        for hist_name in ["incl"] + [b_prod.name for b_prod in self.jet_btag_producers]:
            histograms[hist_name] = h.Weight()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # define columns that need to be read
        read_columns = {"category_ids", "process_id"}
        read_columns |= set(self.weight_producer_inst.used_columns)
        for jet_btag_producer in self.jet_btag_producers:
            read_columns |= jet_btag_producer.uses
        read_columns |= set(map(Route, aliases.values()))

        # add column of the b-tagging algorithm, Jet Genlvl matching & GenJet hadronflavour
        read_columns |= four_vec({"Jet"}, {"hadronFlavour"})
        read_columns = {Route(c) for c in read_columns}

        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        file_targets = [inputs["events"]["events"]]
        if self.producer_insts:
            file_targets.extend([inp["columns"] for inp in inputs["producers"]])
        if self.ml_model_insts:
            file_targets.extend([inp["mlcolumns"] for inp in inputs["ml"]])

        with law.localize_file_targets([*file_targets, *reader_targets.values()], mode="r") as inps:
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.path for inp in inps],
                source_type=len(inps) * ["awkward_parquet"] + len(reader_targets) * [None],
                read_columns=(len(inps) + len(reader_targets)) * [read_columns],
                chunk_size=self.weight_producer_inst.get_min_chunk_size(),
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))

                # add additional columns
                events = update_ak_array(events, *columns)

                # add Jet.btag column
                for jet_btag_producer in self.jet_btag_producers:
                    events = jet_btag_producer(events)

                # build the full event weight
                if not self.weight_producer_inst.skip_func():
                    events, weight = self.weight_producer_inst(events)
                else:
                    weight = ak.Array(np.ones(len(events), dtype=np.float32))

                weight = ak.flatten(ak.broadcast_arrays(weight, events.Jet.hadronFlavour)[0])
                fill_kwargs = {
                    # broadcast event weight and process-id to jet weight
                    "hadronFlavour": ak.flatten(events.Jet.hadronFlavour),
                }
                # loop over Jet variables in which the efficiency is binned
                for var_inst in variable_insts:
                    expr = var_inst.expression
                    if isinstance(expr, str):
                        route = Route(expr)
                        def expr(events, *args, **kwargs):
                            if len(events) == 0 and not has_ak_column(events, route):
                                return empty_f32
                            return route.apply(events, null_value=var_inst.null_value)
                    # apply the variable (flatten to fill histogram)
                    fill_kwargs[var_inst.name] = ak.flatten(expr(events))

                # fill inclusive histogram
                histograms["incl"].fill(**fill_kwargs, weight=weight)

                # fill b-tagged histogram (weight jets by 0 or 1)
                for b_prod in self.jet_btag_producers:
                    histograms[b_prod.name].fill(**fill_kwargs, weight=weight * ak.flatten(events.Jet[b_prod.name]))
            self.output()["hists"].dump(histograms, formatter="pickle")


CreateBTagEfficiencyHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateBTagEfficiencyHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets"],
)


class MergeBTagEfficiencyHistograms(
    BTagAlgoritmsMixin,
    VariablesMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateBTagEfficiencyHistograms=CreateBTagEfficiencyHistograms,
    )

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["hists"] = self.as_branch().requires()

        return reqs

    def requires(self):
        # optional dynamic behavior: determine not yet created variables and require only those

        return self.reqs.CreateBTagEfficiencyHistograms.req(
            self,
            branch=-1,
            _exclude={"branches"},
        )

    def output(self):
        return {"hists": law.SiblingFileCollection({
            algo: self.target(f"hist__{algo}.pickle")
            for algo in self.algorithms
        } | {"incl": self.target(f"hist__incl.pickle")}
        )}

    @law.decorator.log
    def run(self):
        # prepare inputs and outputs
        inputs = self.input()["collection"]
        outputs = self.output()

        # load input histograms
        hists = [
            inp["hists"].load(formatter="pickle")
            for inp in self.iter_progress(inputs.targets.values(), len(inputs), reach=(0, 50))
        ]

        # create a separate file per output variable
        hist_names = list(hists[0].keys())
        for hist_name in self.iter_progress(hist_names, len(hist_names), reach=(50, 100)):

            single_hists = [h[hist_name] for h in hists]
            merged = sum(single_hists[1:], single_hists[0].copy())
            outputs["hists"][hist_name].dump(merged, formatter="pickle")

        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()


MergeBTagEfficiencyHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeBTagEfficiencyHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class BTagEfficiency(
    BTagAlgoritmsMixin,
    VariablesMixin,
    DatasetsProcessesMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    PlotBase2D,
):

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeBTagEfficiencyHistograms=MergeBTagEfficiencyHistograms,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeBTagEfficiencyHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
            )
            for d in self.datasets
        }

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return [DotDict({"algorithm": self.btag_configs[algo]}) for algo in sorted(self.btag_configs)]

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()
        parts["algorithm"] = f"algo_{self.cfg_to_str(self.branch_data.algorithm)}"
        return parts

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):
        return {
            "stats": self.target(".".join(
                self.get_plot_names("btagging_efficiency")[0].split(".")[:-1]
            ) + ".json"),
            "plots": [
                [self.target(name) for name in self.get_plot_names(f"btagging_efficiency__{flav}_hadronflavour")]
                for flav in ["udsg", "c", "b"]
            ],
        }

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", "Processes")
        return params

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        import correctionlib
        import correctionlib.convert

        variable_insts = list(map(self.config_inst.get_variable, self.variables))
        process_insts = list(map(self.config_inst.get_process, self.processes))
        btagAlgorithm = self.branch_data.algorithm.correction_set
        wp = self.branch_data.algorithm.corrector_kwargs["working_point"]
        algo_str = self.cfg_to_str(self.branch_data.algorithm)
        # histogram for the tagged and all jets (combine all datasets)
        hists = {}

        for dataset, inp in self.input().items():
            self.config_inst.get_dataset(dataset)
            h_in = inp["collection"][0]["hists"]

            # copy tagged and inclusive jet histograms
            for key in ["incl", algo_str]:
                h = h_in[key].load(formatter="pickle").copy()

                if key in hists:
                    hists[key] += h
                else:
                    hists[key] = h

        if not hists:
            raise Exception(
                "no histograms found to plot; possible reasons:\n" +
                "  - requested variable requires columns that were missing during histogramming\n" +
                "  - selected --processes did not match any value on the process axis of the input histogram",
            )

        # combine tagged and inclusive histograms to an efficiency histogram
        efficiency_hist = data=hists[algo_str].copy() / hists["incl"].values()

        # save as correctionlib file
        efficiency_hist.name = f"{btagAlgorithm}"
        efficiency_hist.label = "out"
        clibcorr = correctionlib.convert.from_histogram(efficiency_hist)
        clibcorr.description = f"b-tagging efficiency of jets for {btagAlgorithm} algorithm with working point {wp}"

        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description=f"b-tagging efficiency of jets for {btagAlgorithm} algorithm with working point {wp}",
            corrections=[clibcorr],

        )
        self.output()["stats"].dump(cset.dict(exclude_unset=True), indent=4, formatter="json")

        # plot efficiency for each hadronFlavour
        for i, hadronFlavour in enumerate((0, 4, 5)):

            # create a dummy histogram dict for plotting with the first process
            # TODO change process name to the relevant process group
            hist_dict = OrderedDict(((process_insts[-1], efficiency_hist[{"hadronFlavour": hist.loc(hadronFlavour)}]),))

            # create a dummy category for plotting
            cat = od.Category(
                name="hadronFlavour",
                label={0: "udsg flavour", 4: "charm flavour", 5: "bottom flavour"}[hadronFlavour],
            )

            # custom styling:
            label_values = np.around(
                efficiency_hist[{"hadronFlavour": hist.loc(hadronFlavour)}].values() * 100, decimals=1)
            style_config = {"plot2d_cfg": {"cmap": "PiYG", "labels": label_values}}
            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hist_dict,
                config_inst=self.config_inst,
                category_inst=cat.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                style_config=style_config,
                **self.get_plot_parameters(),
            )
            for p in self.output()["plots"][i]:
                p.dump(fig, formatter="mpl")
