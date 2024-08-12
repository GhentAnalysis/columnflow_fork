from __future__ import annotations

import luigi
import law
import order as od
from collections import OrderedDict

from columnflow.tasks.framework.base import Requirements, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, VariablesMixin,
    ChunkedIOMixin, DatasetsProcessesMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase2D,
)

from columnflow.production import Producer
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.tasks.reduction import MergeReducedEvents
from columnflow.util import dev_sandbox, dict_add_strict, four_vec

# discontinued and replaced with producer jet_btag that uses
# config.x attribute "default_btagAlgorithm" (algorithm, working point).
# producer jet_btag makes a boolean column Jet.btag


class CreateBTagEfficiencyHistograms(
    MergeReducedEvents,
    VariablesMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        MergeReducedEvents.reqs,
        RemoteWorkflow.reqs,
        MergeReducedEvents=MergeReducedEvents,
        MergeSelectionStats=MergeSelectionStats)

    # names of columns that contain category ids
    # (might become a parameter at some point)
    category_id_columns = {"category_ids"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # store the normalization weight producer for MC
        self.norm_weight_producer = None
        if self.dataset_inst.is_mc:
            self.norm_weight_producer = Producer.get_cls("normalization_weights")(
                inst_dict=self.get_producer_kwargs(self),
            )
        self.jet_btag_producer = Producer.get_cls("jet_btag")(
            inst_dict=self.get_producer_kwargs(self),
        )

    @law.util.classproperty
    def mandatory_columns(cls) -> set[str]:
        return set(cls.category_id_columns) | {"process_id"}

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["events"] = self.reqs.MergeReducedEvents.req(self, tree_index=-1)
        reqs["selection_stats"] = self.reqs.MergeSelectionStats.req(
            self, tree_index=0, branch=-1, _exclude=MergeSelectionStats.exclude_params_forest_merge)

        if self.dataset_inst.is_mc:
            reqs["normalization"] = self.norm_weight_producer.run_requires()

        reqs["jet_btag"] = self.jet_btag_producer.run_requires()

        return reqs

    def requires(self):
        reqs = {
            "events": self.reqs.MergeReducedEvents.req(self, tree_index=self.branch, _exclude={"branch"}),
            "selection_stats": self.reqs.MergeSelectionStats.req(
                self, tree_index=0, branch=-1, _exclude=MergeSelectionStats.exclude_params_forest_merge),
        }

        if self.dataset_inst.is_mc:
            reqs["normalization"] = self.norm_weight_producer.run_requires()

        reqs["jet_btag"] = self.jet_btag_producer.run_requires()

        return reqs

    def output(self):
        return {
            "hists": self.target(f"histograms__var_{self.variables_repr}__{self.branch}.pickle"),
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
        # setup the normalization weights & jet btag producer
        if self.dataset_inst.is_mc:
            self.norm_weight_producer.run_setup(
                self.requires()["normalization"],
                self.input()["normalization"],
            )
        self.jet_btag_producer.run_setup(
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

        # add two histograms (for nomerator and denominator)
        for hist_name in ("tag", "incl"):
            histograms[hist_name] = h.Weight()

        # define columns that need to be read
        read_columns = {"category_ids", "process_id"} | self.jet_btag_producer.uses
        if self.dataset_inst.is_mc:
            read_columns |= self.norm_weight_producer.used_columns

        # add column of the b-tagging algorithm, Jet Genlvl matching & GenJet hadronflavour
        read_columns |= four_vec({"Jet"}, {"hadronFlavour"})
        read_columns = {Route(c) for c in read_columns}

        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        files = [inputs["events"]["collection"][0]["events"].path]
        for (events, *columns), pos in self.iter_chunked_io(
            files,
            source_type=len(files) * ["awkward_parquet"],
            read_columns=len(files) * [read_columns],
        ):
            # optional check for overlapping inputs
            if self.check_overlapping_inputs:
                self.raise_if_overlapping([events] + list(columns))

            # add additional columns
            events = update_ak_array(events, *columns)

            # add Jet.btag column
            events = self.jet_btag_producer(events)
            # add normalization weight
            if self.dataset_inst.is_mc:
                events = self.norm_weight_producer(events)
            fill_kwargs = {
                # broadcast event weight and process-id to jet weight
                "weight": ak.flatten(ak.broadcast_arrays(events.normalization_weight, events.Jet.hadronFlavour)[0]),
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
            histograms["incl"].fill(**fill_kwargs)

            # fill b-tagged histogram (weight jets by 0 or 1)
            fill_kwargs["weight"] = fill_kwargs["weight"] * ak.flatten(events.Jet.btag)
            histograms["tag"].fill(**fill_kwargs)
        self.output()["hists"].dump(histograms, formatter="pickle")


CreateBTagEfficiencyHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateBTagEfficiencyHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets"],
)


class MergeBTagEfficiencyHistograms(
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
        return {
            "hists": self.target(f"hist__var_{self.variables_repr}.pickle"),
        }

    @law.decorator.log
    def run(self):
        # preare inputs and outputs
        inputs = self.input()["collection"]
        outputs = self.output()

        # load input histograms
        hists = [
            inp["hists"].load(formatter="pickle")
            for inp in self.iter_progress(inputs.targets.values(), len(inputs), reach=(0, 50))
        ]
        merged = {}
        # create a separate file per output variable
        hist_names = list(hists[0].keys())
        for hist_name in self.iter_progress(hist_names, len(hist_names), reach=(50, 100)):

            single_hists = [h[hist_name] for h in hists]
            merged[hist_name] = sum(single_hists[1:], single_hists[0].copy())

        outputs["hists"].dump(merged, formatter="pickle")

        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()


MergeBTagEfficiencyHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeBTagEfficiencyHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class BTagEfficiency(
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
        return {0: None}

    def output(self):
        folder = f"{self.datasets_repr}/alg_{'_'.join(self.config_inst.x.default_btagAlgorithm[:2])}"
        return {
            "stats": self.target(f"{folder}/btagging_efficiency.json"),
            "plots": [
                self.target(f"{folder}/btagging_efficiency__udsg_hadronflavour.pdf"),
                self.target(f"{folder}/btagging_efficiency__c_hadronflavour.pdf"),
                self.target(f"{folder}/btagging_efficiency__b_hadronflavour.pdf"),
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
        btagAlgorithm, wp, _ = self.config_inst.x.default_btagAlgorithm
        # histogram for the tagged and all jets (combine all datasets)
        hists = {}

        for dataset, inp in self.input().items():
            self.config_inst.get_dataset(dataset)
            h_in = inp["collection"][0]["hists"].load(formatter="pickle")

            # copy tagged and inclusive jet histograms
            for key in h_in.keys():
                h = h_in[key].copy()

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
        efficiency_hist = hist.Hist(*hists["tag"].axes[:], data=hists["tag"].values() / hists["incl"].values())

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

            self.output()["plots"][i].dump(fig, formatter="mpl")
