from __future__ import annotations

import law
import order as od
import luigi
from collections import OrderedDict
from itertools import product

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, VariablesMixin,
    DatasetsProcessesMixin, SelectorMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase2D, PlotBase1D,
)

from columnflow.tasks.selection import MergeSelectionStats

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, dict_add_strict
from columnflow.types import Any


class FixedWPEfficiencyBase(
    VariablesMixin,
    DatasetsProcessesMixin,
    SelectorMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    PlotBase2D,
):

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    control_plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_variable_per_process",
        add_default_to_description=True,
    )

    make_plots = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, make plots of efficiencies",
    )

    control_skip_ratio = PlotBase1D.skip_ratio.copy()
    control_density = PlotBase1D.density.copy()
    control_yscale = PlotBase1D.yscale.copy()
    control_shape_norm = PlotBase1D.shape_norm.copy()
    control_hide_errors = PlotBase1D.hide_errors.copy()

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    exclude_index = True

    tag_name = "btag"
    flav_name = "hadronFlavour"
    flavours = {0: "light", 4: "charm", 5: "bottom"}
    wps = ["L", "M", "T"]

    @classmethod
    def resolve_param_values(
            cls,
            params: law.util.InsertableDict[str, Any],
    ) -> law.util.InsertableDict[str, Any]:
        f"""
        Resolve values *params* and check against possible default values

        Check the values in *params* against the default value
        ``"default_{cls.tag_name}_variables"`` in the current config inst.
        For more information, see
        :py:meth:`~columnflow.tasks.framework.base.ConfigTask.resolve_config_default_and_groups`.
        """
        redo_default_variables = False
        # when empty, use the config default
        if not params.get("variables", None):
            redo_default_variables = True

        params = super().resolve_param_values(params)

        config_inst = params.get("config_inst")
        if not config_inst:
            return params

        if redo_default_variables:
            # when empty, use the config default
            if config_inst.x(f"default_{cls.tag_name}_variables", ()):
                params["variables"] = tuple(config_inst.x(f"default_{cls.tag_name}_variables"))
            elif cls.default_variables:
                params["variables"] = tuple(cls.default_variables)
            else:
                raise AssertionError(f"define default {cls.tag_name} variables "
                                     f"in {cls.__class__} or config {config_inst.name}")

        return params

    def workflow_requires(self):
        reqs = super().workflow_requires()
        for d in self.datasets:
            reqs[d] = self.reqs.MergeSelectionStats.req(
                self,
                tree_index=0,
                branch=-1,
                dataset=d,
                _exclude=MergeSelectionStats.exclude_params_forest_merge,
            )
        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeSelectionStats.req(
                self,
                tree_index=0,
                branch=-1,
                dataset=d,
                _exclude=MergeSelectionStats.exclude_params_forest_merge,
            )
            for d in self.datasets
        }

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):
        out = {
            "stats": self.target(".".join(
                self.get_plot_names(f"{self.tag_name}_efficiency")[0].split(".")[:-1],
            ) + ".json"),
        }
        if not self.make_plots:
            return out
        return out | {
            "plots": {dr: [
                [self.target(name)
                 for name in self.get_plot_names(
                    f"{self.tag_name}_eff__{flav}_{self.flav_name}"
                    f"__wp_{wp}" +
                    (f"__err_{dr}" if dr != "central" else ""),
                )]
                for flav in self.flavours.values()
                for wp in self.wps
            ] for dr in ["central", "down", "up"]},
            "controls": {
                vr: [self.target(f"control_{name}") for name in self.get_plot_names(vr)]
                for vr in [
                    self.flav_name,
                    f"{self.tag_name}_wp",
                    *self.variables,
                ]
            },
        }

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", "Processes")
        return params

    def get_control_plot_parameter(self):
        params = PlotBase.get_plot_parameters(self)
        for pp in ["skip_ratio", "density", "shape_norm", "hide_errors"]:
            params[pp] = getattr(self, f"control_{pp}")
        params["yscale"] = None if self.control_yscale == law.NO_STR else self.control_yscale
        return params

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        import correctionlib
        import correctionlib.convert
        from columnflow.plotting.cmsGhent.plot_util import cumulate
        from columnflow.plotting.plot_util import use_flow_bins
        from statsmodels.stats.proportion import proportion_confint

        variable_insts = list(map(self.config_inst.get_variable, self.variables))

        # histogram for the tagged and all jets (combine all datasets)
        histogram = {}
        for dataset, inp in self.input().items():
            dataset_inst = self.config_inst.get_dataset(dataset)
            dt_process_insts: set[od.Process] = {process_inst for process_inst in dataset_inst.processes}
            union_process_name = "_".join([process_inst.name for process_inst in dt_process_insts])
            if self.config_inst.has_process(union_process_name):
                union_process = self.config_inst.get_process(union_process_name)
                xsec = union_process.get_xsec(self.config_inst.campaign.ecm).nominal
            else:
                union_process = od.Process(
                    name="_".join([process_inst.name for process_inst in dt_process_insts]),
                    label=", ".join([process_inst.label for process_inst in dt_process_insts]),
                    id=-1,
                )
                xsec = sum(
                    process_inst.get_xsec(self.config_inst.campaign.ecm).nominal
                    for process_inst in dt_process_insts
                )
            h_in = inp["collection"][0]["hists"].load(formatter="pickle")[f"{self.tag_name}_efficiencies"]
            for variable_inst in variable_insts:
                v_idx = h_in.axes.name.index(variable_inst.name)
                for mi, mj in variable_inst.x("merge_bins", []):
                    merged_bins = h_in[{variable_inst.name: slice(mi, mj + 1, sum)}]
                    new_arr = np.moveaxis(h_in.view(), v_idx, 0)
                    new_arr[mi:mj + 1] = merged_bins.view()[np.newaxis]
                    h_in.view()[:] = np.moveaxis(new_arr, 0, v_idx)

                if any(flows := [
                    getattr(variable_inst, f + "flow", variable_inst.x(f + "flow", False))
                    for f in ["under", "over"]
                ]):
                    h_in = use_flow_bins(h_in, variable_inst.name, *flows)
            norm = inp["collection"][0]["stats"].load()["sum_mc_weight"]
            histogram[union_process] = h_in * xsec / norm * self.config_inst.x.luminosity.nominal

        if not histogram:
            raise Exception(
                "no histograms found to plot; possible reasons:\n" +
                "  - requested variable requires columns that were missing during histogramming\n" +
                "  - selected --processes did not match any value on the process axis of the input histogram",
            )

        # combine tagged and inclusive histograms to an efficiency histogram
        sum_histogram = sum(histogram.values())
        cum_histogram = cumulate(sum_histogram, direction="above", axis=f"{self.tag_name}_wp")
        incl = cum_histogram[{f"{self.tag_name}_wp": slice(0, 1)}]

        axes = OrderedDict(zip(cum_histogram.axes.name, cum_histogram.axes))
        axes[f"{self.tag_name}_wp"] = hist.axis.StrCategory(self.wps, name=f"{self.tag_name}_wp", label="working point")
        efficiency_hist = hist.Hist(
            *axes.values(),
            name=sum_histogram.name,
            storage=hist.storage.Weight(),
        )

        efficiency_hist.view()[:] = cum_histogram[{f"{self.tag_name}_wp": slice(1, None)}].view()
        efficiency_hist = efficiency_hist / incl.values()

        # save as correctionlib file (unfortunately not possible to specify the flow for each variable separately)
        efficiency_hist.label = "out"
        description = f"{self.tag_name} efficiencies of jets for {efficiency_hist.name} algorithm"
        clibcorr = correctionlib.convert.from_histogram(efficiency_hist, flow="clamp")
        clibcorr.description = description

        cset = correctionlib.schemav2.CorrectionSet(schema_version=2, description=description, corrections=[clibcorr])
        self.output()["stats"].dump(cset.dict(exclude_unset=True), indent=4, formatter="json")

        if not self.make_plots:
            return

        selected_counts = cum_histogram[{f"{self.tag_name}_wp": slice(1, None)}]
        eff_sample_size_corr = incl.values() / incl.variances()
        eff_selected = selected_counts.values() * eff_sample_size_corr
        eff_incl = incl.values() * eff_sample_size_corr

        ratio = eff_selected / eff_incl
        error = proportion_confint(eff_selected, eff_incl, alpha=0.35, method="beta")
        err_hists = []
        for err in error:
            err_hists.append(hist.Hist(*axes.values(), storage=hist.storage.Weight()))
            err_hists[-1].values()[:] = err - ratio
            err_hists[-1].variances()[:] = 1
        # plot efficiency for each hadronFlavour and wp
        for i, (flav, wp) in enumerate(product(self.flavours, self.wps)):

            for name, h in [("central", efficiency_hist), *zip(["down", "up"], err_hists)]:
                # create a dummy histogram dict for plotting with the first process
                hist_dict = OrderedDict((
                    (self.config_inst.get_process(self.processes[-1]),
                     h[{
                         self.flav_name: hist.loc(flav),
                         f"{self.tag_name}_wp": wp,
                     }]),),
                )

                # create a dummy category for plotting
                cat = od.Category(name=self.flav_name, label=self.flavours[flav])

                # custom styling:
                label_values = np.around(
                    h[{self.flav_name: hist.loc(flav)}].values() * 100, decimals=1)
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
                for p in self.output()["plots"][name][i]:
                    p.dump(fig, formatter="mpl")

        for control_plot_var in [self.flav_name, f"{self.tag_name}_wp", *self.variables]:
            axis = sum_histogram.axes[control_plot_var]
            if control_plot_var in self.variables:
                variable_inst = self.config_inst.get_variable(control_plot_var)
            elif control_plot_var == self.flav_name:
                n_bins = len(axis.edges)
                variable_inst = od.Variable(
                    name=axis.name, binning=(n_bins, -0.5, n_bins - 0.5), discrete_x=True,
                    x_labels=[axis.bin(i) for i in range(n_bins)],
                )
            else:
                variable_inst = od.Variable(name=f"{self.tag_name} score", binning=axis.edges.tolist())

            fig, _ = self.call_plot_func(
                self.control_plot_function,
                hists={process_inst: h.project(control_plot_var) for process_inst, h in histogram.items()},
                config_inst=self.config_inst,
                category_inst=od.Category(name=f"baseline, excluding\n{self.tag_name} selection"),
                variable_insts=[variable_inst.copy_shallow()],
                **self.get_control_plot_parameter(),
            )
            for p in self.output()["controls"][control_plot_var]:
                p.dump(fig, formatter="mpl")


class BTagEfficiency(FixedWPEfficiencyBase):
    exclude_index = False


class LeptonMVAEfficiency(FixedWPEfficiencyBase):
    exclude_index = False

    tag_name = "lepton_mva"
    flav_name = "pdgId"
    flavours = {11: "electron", 13: "muon"}
    wps = ["VLoose", "Loose", "Medium", "Tight"]
