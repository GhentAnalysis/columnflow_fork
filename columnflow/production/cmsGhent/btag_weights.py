"""
Producer that produces a column Jet.btag based on the default_btag Algorithm provided in the config
"""

from __future__ import annotations

import law
import order as od


from columnflow.production import Producer, producer
from columnflow.weight import weight_producer
from columnflow.selection import SelectionResult

from columnflow.util import maybe_import, InsertableDict, DotDict
from columnflow.columnar_util import set_ak_column, layout_ak_array, Route, has_ak_column
from columnflow.production.cms.btag import BTagSFConfig

ak = maybe_import("awkward")
np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


@weight_producer(
    btag_config=None,
    add_weights=True,
    name=lambda btag_config: f"{btag_config.discriminator}_{btag_config.corrector_kwargs['working_point']}",
    efficiency_variables=None,
)
def fixed_wp_btag_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    btag = events.Jet[self.btag_descriminator] >= self.btag_wp_value
    events = set_ak_column(events, f"Jet.{self.name}", btag)

    if not self.add_weights:
        return events

    # get the total number of jets in the chunk
    n_jets_all = ak.sum(ak.num(events.Jet, axis=1))
    np.ones(n_jets_all, dtype=np.float32)

    # get flat inputs
    flavor = ak.flatten(events.Jet.hadronFlavour, axis=1)
    abs_eta = ak.flatten(abs(events.Jet.eta), axis=1)
    pt = ak.flatten(events.Jet.pt, axis=1)
    btag = ak.flatten(btag, axis=1)

    # helper to create and store the weight
    def add_weight(syst_name, syst_direction, column_name):

        # define a mask that selects the correct flavor to assign to, depending on the systematic
        flavor_mask = Ellipsis
        if syst_name == "light":
            # only apply to light flavor
            flavor_mask = flavor == 0
            btag_sf_corrector = self.btag_sf_incl_corrector
        elif syst_name == "heavy":
            # only apply to heavy flavor
            flavor_mask = flavor != 0
            btag_sf_corrector = self.btag_sf_comb_corrector

        # get the flat scale factors
        sf_flat = btag_sf_corrector(
            syst_direction,
            self.wp,
            flavor[flavor_mask],
            abs_eta[flavor_mask],
            pt[flavor_mask],
        )

        eff_flat = self.btag_eff_corrector(
            flavor[flavor_mask],
            # currently set hard max on pt since overflow could not be changed in correctionlib
            # (could also manually change the flow)
            ak.min([pt[flavor_mask], 999 * ak.ones_like(pt[flavor_mask])], axis=0),
            abs_eta[flavor_mask],
        )

        # calculate the event weight following:
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
        weight_flat = ak.where(btag[flavor_mask],
                               sf_flat, (1. - sf_flat * eff_flat) / (1. - eff_flat))

        # insert them into an array of ones whose length corresponds to the total number of jets
        # (without flavor mask)
        weight_flat_all = np.ones(n_jets_all, dtype=np.float32)
        weight_flat_all[flavor_mask] = weight_flat

        # enforce the correct shape and create the product over all jets per event
        sf = layout_ak_array(weight_flat_all, events.Jet.pt)

        weight = ak.prod(sf, axis=1, mask_identity=False)

        if ak.any((weight == np.inf) | ak.is_none(ak.nan_to_none(weight)) | ak.any(weight < 0)):
            weight = ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
            logger.warning_once(
                "weight column has an infinite, Nan or negative value",
                f"weight column events.{column_name} has an infinite, Nan or negative value and is set to 1." +
                " Make sure the b-tagging efficiency is defined and physical in all bins!",
            )

        return set_ak_column(events, column_name, weight, value_type=np.float32)

    # nominal weight and those of all method intrinsic uncertainties
    for syst_name in self.btag_uncs:
        events = add_weight(syst_name, "central", f"btag_weight_{syst_name}")

        # only calculate up and down variations for nominal shift
        if self.local_shift_inst.is_nominal:
            for direction in ["up", "down"]:
                events = add_weight(
                    syst_name,
                    direction,
                    f"{self.name}_weight_{syst_name}_{direction}",
                )

    return events


@fixed_wp_btag_weights.init
def fixed_wp_btag_weights_init(
    self: Producer,
) -> None:

    if self.btag_config is None:
        self.btag_config = self.config_inst.aux.get(
            "default_btagAlgorithm",
            BTagSFConfig(
                correction_set="DeepJet",
                jec_sources=[],
                corrector_kwargs=dict(working_point="M"),
            ),
        )

    assert "working_point" in self.btag_config.corrector_kwargs, "no working point specified"

    # setup requires the algorithm name
    self.name = self.name(self.btag_config)
    self.btag_algorithm = self.btag_config.correction_set
    self.btag_wp = self.btag_config.corrector_kwargs["working_point"]
    # self requires for btag column calculation
    self.btag_descriminator = self.btag_config.discriminator
    self.uses.add(f"Jet.{self.btag_config.discriminator }")

    if self.dataset_inst.is_data:
        self.add_weights = False

    if self.add_weights:
        self.uses.add("Jet.{pt,eta,hadronFlavour}")
        # depending on the requested shift_inst, there are three cases to handle:
        #   1. when a JEC uncertainty is requested whose propagation to btag weights is known, the
        #      producer should only produce that specific weight column
        #   2. when the nominal shift is requested, the central weight and all variations related to the
        #      method-intrinsic shifts are produced
        #   3. when any other shift is requested, only create the central weight column

        shift_inst = getattr(self, "local_shift_inst", None)
        if not shift_inst:
            return

        if self.variables is None:
            if hasattr(self.config_inst.x, "default_btag_variables"):
                self.variables = self.config_inst.x.default_btag_variables
            else:
                logger.warning_once(
                    "no default btagging efficiency variables defined in config",
                    "Config does not have an attribute x.default_btag_variables that provides default \
                        variables in which to bin b - tagging efficiency.\n \
                        The variables 'btag_jet_pt' & 'btag_jet_eta' are used if defined in the config.",
                )
                self.variables = ("btag_jet_pt", "btag_jet_eta")

        # to handle this efficiently in one spot, store jec information
        self.jec_source = shift_inst.x.jec_source if shift_inst.has_tag("jec") else None
        "" if self.jec_source == "Total" else self.jec_source

        # save names of method-intrinsic uncertainties
        self.btag_uncs = {
            "light",
            "heavy",
        }

        # add uncertainty sources of the method itself
        if shift_inst.is_nominal:
            for name in self.btag_uncs:
                # nominal columns
                self.produces.add(f"btag_weight_{name}")
                # directions
                for direction in ["up", "down"]:
                    self.produces.add(f"btag_weight_{name}_{direction}")
        else:
            # only the nominal columns
            for name in self.btag_uncs:
                self.produces.add(f"btag_weight_{name}")


@fixed_wp_btag_weights.setup
def fixed_wp_btag_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    import correctionlib

    bundle = reqs["external_files"]
    correction_set_btag_wp_corr = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )

    btag_wp_corrector = correction_set_btag_wp_corr[f"{self.btag_algorithm}_wp_values"]
    self.btag_wp_value = btag_wp_corrector.evaluate(self.btag_wp)

    if self.add_weights:
        # fix for change in nomenclature of deepJet scale factors for light hadronFlavour jets
        if self.config_inst.x.run == 2:
            self.btag_sf_incl_corrector = correction_set_btag_wp_corr[f"{self.btag_algorithm}_incl"]
        else:
            self.btag_sf_incl_corrector = correction_set_btag_wp_corr[f"{self.btag_algorithm}_light"]
        self.btag_sf_comb_corrector = correction_set_btag_wp_corr[f"{self.btag_algorithm}_comb"]

        # unpack the b-tagging efficiency
        correction_set_btag_eff_corr = correctionlib.CorrectionSet.from_file(
            reqs["btag_efficiency"].output()["stats"].path,
        )
        if len(correction_set_btag_eff_corr.keys()) != 1:
            raise Exception("Expected exactly one type of btagging efficiencies")

        corrector_name = list(correction_set_btag_eff_corr.keys())[0]
        self.btag_eff_corrector = correction_set_btag_eff_corr[corrector_name]


@fixed_wp_btag_weights.requires
def fixed_wp_btag_weights_requires(self: Producer, reqs: dict) -> None:

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)

    if not self.add_weights:
        return

    from columnflow.tasks.cmsGhent.btagefficiency import BTagEfficiency

    # require btag efficiency to be ran for the btag_dataset_group
    # default value of datasets to calculate the efficiency is the dataset of the produce task
    datasets = [self.dataset_inst.name]
    process = self.dataset_inst.processes.names()[0]

    if hasattr(self.config_inst.x, "btag_dataset_groups"):
        for btag_group in self.config_inst.x.btag_dataset_groups:
            # check if dataset is in data group
            if self.dataset_inst.name in self.config_inst.x.btag_dataset_groups[btag_group]:
                datasets = self.config_inst.x.btag_dataset_groups[btag_group]
                if btag_group in self.config_inst.processes.names():
                    process = btag_group  # only for plotting text
                break
    else:
        logger.warning_once(
            "no default btagging efficiency dataset groups defined in config",
            "Config does not have an attribute 'x.btag_dataset_groups' that provides  \
            default groupings of datasets for b-tagging efficiency calculation.\n"
            f"The dataset {self.dataset_inst.name} is used to calculate but defining one is recommended.\n"
            "example: config.x.btag_dataset_groups = {'ttx': ['ttztollnunu_m10_amcatnlo','tt_sl_powheg']}",
        )

    reqs["btag_efficiency"] = BTagEfficiency.req(
        self.task,
        datasets=datasets,
        variables=self.variables,
        processes=process,
        algorithms=[self.btag_config],
    )


@producer(
    uses={"mc_weight", "Jet.{hadronFlavour,pt,eta}"},
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_btag_file=(lambda self, external_files: external_files.btag_sf_corr),
    # function to determine the btag sf config
    get_btag_config=(lambda self: BTagSFConfig.new(self.config_inst.x.btag_sf)),
    # function to determine the efficiency variables
    get_btag_vars=(lambda self: self.config_inst.x.default_btag_variables),
)
def btag_efficiency_hists(
    self: Producer,
    events: ak.Array,
    results: SelectionResult,
    hists: DotDict | dict = None,
    **kwargs,
) -> ak.Array:

    if hists is None:
        return events

    assert "event_no_btag" in results.aux, "results.aux does not contain mask 'event_no_btag'"

    selected_events = events[results.x.event_no_btag]

    histogram = hist.Hist.new.IntCat([0, 4, 5], name="hadronFlavour")  # Jet hadronFlavour 0, 4, or 5
    # add variables for binning the efficiency
    for var_inst in self.variable_insts:
        histogram = histogram.Var(
            var_inst.bin_edges,
            name=var_inst.name,
            label=var_inst.get_full_x_title(),
        )
    hists["btag_efficiencies"] = histogram.Weight()

    fill_kwargs = {
        # broadcast event weight and process-id to jet weight
        "hadronFlavour": ak.flatten(selected_events.Jet.hadronFlavour),
        "weight": ak.flatten(ak.broadcast_arrays(selected_events.mc_weight, selected_events.Jet.hadronFlavour)[0]),
    }

    # loop over Jet variables in which the efficiency is binned
    for var_inst in self.variable_insts:
        expr = var_inst.expression
        if isinstance(expr, str):
            route = Route(expr)

            def expr(selected_events, *args, **kwargs):
                if len(selected_events) == 0 and not has_ak_column(selected_events, route):
                    return ak.Array(np.array([], dtype=np.float32))
                return route.apply(selected_events, null_value=var_inst.null_value)

        # apply the variable (flatten to fill histogram)
        fill_kwargs[var_inst.name] = ak.flatten(expr(selected_events))

    # fill inclusive histogram
    hists["btag_efficiencies"].fill(**fill_kwargs)
    hists["btag_efficiencies"].name = f"{self.btag_config.correction_set}({self.btag_config.discriminator})"

    return events


@btag_efficiency_hists.init
def btag_efficiency_hists_init(self: Producer) -> None:
    self.btag_config: BTagSFConfig = self.get_btag_config()
    self.uses.add(f"Jet.{self.btag_config.discriminator}")

    self.variable_insts = list(map(self.config_inst.get_variable, self.get_btag_vars()))
    self.uses.update({
        inp
        for variable_inst in self.variable_insts
        for inp in (
            [variable_inst.expression] if isinstance(variable_inst.expression, str) else variable_inst.x("inputs", [])
        )
    })


@btag_efficiency_hists.setup
def btag_efficiency_hists_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    import correctionlib

    bundle = reqs["external_files"]
    correction_set_btag_wp_corr = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )

    btag_wp_corrector = correction_set_btag_wp_corr[f"{self.btag_config.correction_set}_wp_values"]
    self.variable_insts.append(od.Variable(
        name="btag_wp",
        expression=f"Jet.{self.btag_config.discriminator}",
        binning=[0, *(btag_wp_corrector.evaluate(wp) for wp in "LMT"), 1],
        x_labels=["U", "L", "M", "T"],
    ))


@btag_efficiency_hists.requires
def btag_efficiency_hists_requires(self: Producer, reqs: dict) -> None:

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)
