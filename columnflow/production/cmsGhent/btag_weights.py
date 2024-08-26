"""
Producer that produces a column Jet.btag based on the default_btag Algorithm provided in the config
"""

from __future__ import annotations

import law
import order as od
from typing import Iterable
from collections import OrderedDict


from columnflow.production import Producer, producer
from columnflow.weight import WeightProducer, weight_producer
from columnflow.selection import SelectionResult

from columnflow.util import maybe_import, InsertableDict, DotDict
from columnflow.columnar_util import set_ak_column, layout_ak_array, Route, has_ak_column, optional_column
from columnflow.production.cms.btag import BTagSFConfig

ak = maybe_import("awkward")
np = maybe_import("numpy")
hist = maybe_import("hist")
correctionlib = maybe_import("correctionlib")

logger = law.logger.get_logger(__name__)


def init_btag(self: Producer | WeightProducer, add_eff_vars=True):
    if self.btag_config is None:
        self.btag_config = self.config_inst.x(
            "btag_sf",
            BTagSFConfig(
                correction_set="DeepJet",
                jec_sources=[],
            ),
        )
        self.btag_config = BTagSFConfig.new(self.btag_config)

    # setup requires the algorithm name
    self.btag_algorithm = self.btag_config.correction_set
    # self requires for btag column calculation
    self.btag_descriminator = self.btag_config.discriminator
    self.uses.add(f"Jet.{self.btag_config.discriminator }")

    if add_eff_vars:
        if "default_btag_variables" not in self.config_inst.aux:
            logger.warning_once(
                "no default btagging efficiency variables defined in config",
                "Config does not have an attribute x.default_btag_variables that provides default \
                    variables in which to bin b - tagging efficiency.\n \
                    The variables 'btag_jet_pt' & 'btag_jet_eta' are used if defined in the config.",
            )
        self.variables = self.config_inst.x("default_btag_variables", ("btag_jet_pt", "btag_jet_eta"))
        self.variable_insts = list(map(self.config_inst.get_variable, self.variables))
        self.uses.update({
            inp
            for variable_inst in self.variable_insts
            for inp in (
                [variable_inst.expression] if isinstance(variable_inst.expression, str) else variable_inst.x("inputs",
                                                                                                             [])
            )
        })


def setup_btag(self: Producer | WeightProducer, reqs: dict):
    bundle = reqs["external_files"]
    correction_set_btag_wp_corr = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )

    btag_wp_corrector = correction_set_btag_wp_corr[f"{self.btag_algorithm}_wp_values"]
    self.btag_wp_value = OrderedDict([(wp, btag_wp_corrector.evaluate(wp)) for wp in "LMT"])
    return correction_set_btag_wp_corr


def req_btag(self: Producer | WeightProducer, reqs: dict):
    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@producer(
    btag_config=None,
    produces={optional_column("Jet.btag_{LMT}")},
)
def jet_btag(
    self: Producer,
    events: ak.Array,
    working_points: Iterable[str],
    jet_mask: ak.Array[bool] = None,
    **kwargs,
) -> ak.Array:

    for wp in working_points:
        tag = events.Jet[self.btag_descriminator] >= self.btag_wp_value[wp]
        if jet_mask is not None:
            tag = tag & jet_mask
        events = set_ak_column(events, f"Jet.btag_{wp}", tag)

    return events


@jet_btag.init
def jet_btag_init(self: Producer):
    init_btag(self, add_eff_vars=False)


@jet_btag.setup
def jet_btag_setup(self: Producer, reqs: dict, *args, **kwargs) -> None:
    setup_btag(self, reqs)


@jet_btag.requires
def jet_btag_requires(self: Producer, reqs: dict) -> None:
    req_btag(self, reqs)


@weight_producer(
    uses={"Jet.{pt,eta,hadronFlavour}", jet_btag},
    btag_config=None,
    mc_only=True,
)
def fixed_wp_btag_weights(
    self: Producer,
    events: ak.Array,
    working_points: Iterable[str],
    jet_mask: ak.Array[bool] = None,
    **kwargs,
) -> ak.Array:
    # get the total number of jets in the chunk
    jets = events.Jet[jet_mask] if jet_mask is not None else events.Jet

    n_jets_all = ak.sum(ak.num(jets, axis=1))
    np.ones(n_jets_all, dtype=np.float32)

    # get flat inputs
    flavor = ak.flatten(jets.hadronFlavour, axis=1)
    abs_eta = ak.flatten(abs(jets.eta), axis=1)
    pt = ak.flatten(jets.pt, axis=1)
    btags = {wp: ak.flatten(jets[f"btag_{wp}"], axis=1) for wp in working_points}

    # helper to create and store the weight
    def add_weight(syst_name, syst_direction, working_point):

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
            working_point,
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
            working_point,
        )

        # calculate the event weight following:
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
        weight_flat = ak.where(btags[working_point][flavor_mask],
                               sf_flat, (1. - sf_flat * eff_flat) / (1. - eff_flat))

        # insert them into an array of ones whose length corresponds to the total number of jets
        # (without flavor mask)
        weight_flat_all = np.ones(n_jets_all, dtype=np.float32)
        weight_flat_all[flavor_mask] = weight_flat

        # enforce the correct shape and create the product over all jets per event
        sf = layout_ak_array(weight_flat_all, jets.pt)

        weight = ak.prod(sf, axis=1, mask_identity=False)

        column_name = f"btag_{working_point}_weight_{syst_name}"
        if syst_direction != "central":
            column_name += "_" + syst_direction

        if ak.any((weight == np.inf) | ak.is_none(ak.nan_to_none(weight)) | ak.any(weight < 0)):
            weight = ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
            logger.warning_once(
                "weight column has an infinite, Nan or negative value",
                f"weight column events.{column_name} has an infinite, Nan or negative value and is set to 1." +
                " Make sure the b-tagging efficiency is defined and physical in all bins!",
            )

        return set_ak_column(events, column_name, weight, value_type=np.float32)

    for wp in working_points:
        # nominal weight and those of all method intrinsic uncertainties
        for syst_name in self.btag_uncs:
            events = add_weight(syst_name, "central", wp)

            # only calculate up and down variations for nominal shift
            if self.local_shift_inst.is_nominal:
                for direction in ["up", "down"]:
                    events = add_weight(syst_name, direction, wp)

    return events


@fixed_wp_btag_weights.init
def fixed_wp_btag_weights_init(
    self: Producer,
) -> None:
    init_btag(self)

    # depending on the requested shift_inst, there are three cases to handle:
    #   1. when a JEC uncertainty is requested whose propagation to btag weights is known, the
    #      producer should only produce that specific weight column
    #   2. when the nominal shift is requested, the central weight and all variations related to the
    #      method-intrinsic shifts are produced
    #   3. when any other shift is requested, only create the central weight column

    shift_inst = getattr(self, "local_shift_inst", None)
    if not shift_inst:
        return

    # to handle this efficiently in one spot, store jec information
    self.jec_source = shift_inst.x.jec_source if shift_inst.has_tag("jec") else None
    btag_sf_jec_source = "" if self.jec_source == "Total" else self.jec_source  # noqa

    # save names of method-intrinsic uncertainties
    self.btag_uncs = {
        "light",
        "heavy",
    }

    # add uncertainty sources of the method itself
    produces = set()
    for name in self.btag_uncs:
        # nominal columns
        produces.add(f"weight_{name}")
        if shift_inst.is_nominal:
            produces.update({f"weight_{name}_{direction}" for direction in ["up", "down"]})
    for c in produces:
        self.produces.add(optional_column("btag_{LMT}" + c))


@fixed_wp_btag_weights.setup
def fixed_wp_btag_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    correction_set_btag_wp_corr = setup_btag(self, reqs)

    # fix for change in nomenclature of deepJet scale factors for light hadronFlavour jets
    if self.config_inst.x.year <= 2018:
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
    req_btag(self, reqs)

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
    )


@producer(
    uses={"mc_weight", "Jet.{hadronFlavour,pt,eta}"},
    btag_config=None,
    # only run on mc
    mc_only=True,
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

    # jet selection and event selection
    jets = events.Jet[results.objects.Jet.Jet][results.x.event_no_btag]
    selected_events = ak.Array({
        "Jet": jets,
        "mc_weight": events.mc_weight[results.x.event_no_btag],
    })

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
        "hadronFlavour": ak.flatten(jets.hadronFlavour),
        "weight": ak.flatten(ak.broadcast_arrays(selected_events.mc_weight, jets.hadronFlavour)[0]),
    }

    # loop over Jet variables in which the efficiency is binned
    for var_inst in self.variable_insts:
        expr = var_inst.expression
        if isinstance(expr, str):
            route = Route(expr)

            def expr(evs, *args, **kwargs):
                if len(evs) == 0 and not has_ak_column(evs, route):
                    return ak.Array(np.array([], dtype=np.float32))
                return route.apply(evs, null_value=var_inst.null_value)

        # apply the variable (flatten to fill histogram)
        fill_kwargs[var_inst.name] = ak.flatten(expr(selected_events))

    # fill inclusive histogram
    hists["btag_efficiencies"].fill(**fill_kwargs)
    hists["btag_efficiencies"].name = f"{self.btag_config.correction_set}({self.btag_config.discriminator})"

    return events


@btag_efficiency_hists.init
def btag_efficiency_hists_init(self: Producer) -> None:
    init_btag(self)


@btag_efficiency_hists.setup
def btag_efficiency_hists_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    setup_btag(self, reqs)
    self.variable_insts.append(od.Variable(
        name="btag_wp",
        expression=f"Jet.{self.btag_config.discriminator}",
        binning=[0, *self.btag_wp_value.values(), 1],
        x_labels=["U", "L", "M", "T"],
    ))


@btag_efficiency_hists.requires
def btag_efficiency_hists_requires(self: Producer, reqs: dict) -> None:
    req_btag(self, reqs)
