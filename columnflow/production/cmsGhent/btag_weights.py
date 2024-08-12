"""
Producer that produces a column Jet.btag based on the default_btag Algorithm provided in the config
"""

import law


from columnflow.production import Producer, producer
from columnflow.weight import WeightProducer, weight_producer

from columnflow.util import maybe_import, four_vec, InsertableDict
from columnflow.columnar_util import set_ak_column, layout_ak_array

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


@producer(
    uses=(four_vec("Jet")),
    produces=("Jet.btag"),
)
def jet_btag(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    events = set_ak_column(events, "Jet.btag", events.Jet[self.btag_descriminator] >= self.btag_wp_value)

    return events


@jet_btag.init
def jet_btag_init(
    self: Producer,
) -> None:

    if hasattr(self.config_inst.x, "default_btagAlgorithm"):
        assert len(self.config_inst.x.default_btagAlgorithm) == 3, \
            "btag algorithm requires tuple specifying the algorith, the wp, and the discriminator"
        btagAlgorithm, wp, btag_descriminator = self.config_inst.x.default_btagAlgorithm

    else:
        logger.warning_once(
            "no default btagging algorithm and working point defined in config",
            "Config does not have an attribute x.default_btagAlgorithm that provides default algorithm \
                and working point with which to identify b-jets.\n\
            The algorithm 'deepJet with medium working point is used if defined in the config.",
        )
        btagAlgorithm, wp, btag_descriminator = ("deepJet", "M", "btagDeepFlavB")

    self.btag_algorithm = btagAlgorithm  # setup requires the algorithm name
    self.btag_wp = wp
    self.btag_descriminator = btag_descriminator  # producer requires for btag column calculation
    self.uses.add(f"Jet.{btag_descriminator}")


@jet_btag.setup
def jet_btag_setup(
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


@jet_btag.requires
def jet_btag_requires(self: Producer, reqs: dict) -> None:

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@weight_producer(
    # make sure to add the default btagging algorithm used in the analysis to the config
    # example: "cfg.x.default_btagAlgorithm = ... (notation as in events.Jet.{})"
    uses={"Jet.pt", "Jet.eta", "Jet.hadronFlavour"} | {jet_btag},
    mc_only=True,
)
def btag_weights_fixed_wp(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # get the total number of jets in the chunk
    n_jets_all = ak.sum(ak.num(events.Jet, axis=1))
    events = self[jet_btag](events, **kwargs)
    np.ones(n_jets_all, dtype=np.float32)

    # get flat inputs
    flavor = ak.flatten(events.Jet.hadronFlavour, axis=1)
    abs_eta = ak.flatten(abs(events.Jet.eta), axis=1)
    pt = ak.flatten(events.Jet.pt, axis=1)
    btag = ak.flatten(events.Jet.btag, axis=1)

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
                    f"btag_weight_{syst_name}_{direction}",
                )

    return events


@btag_weights_fixed_wp.requires
def btag_weights_fixed_wp_requires(self: Producer, reqs: dict) -> None:

    from columnflow.tasks.external import BundleExternalFiles
    from columnflow.tasks.cmsGhent.btagefficiency import BTagEfficiency

    reqs["external_files"] = BundleExternalFiles.req(self.task)

    # require btag efficiency to be ran for the btag_dataset_group
    # default value of datasets to calculate the efficiency is the dataset of the produce task
    datasets = [self.dataset_inst.name]
    process = self.dataset_inst.processes.names()[0]

    if hasattr(self.config_inst.x, "default_btag_variables"):
        variables = self.config_inst.x.default_btag_variables
    else:
        logger.warning_once(
            "no default btagging efficiency variables defined in config",
            "Config does not have an attribute x.default_btagAlgorithm that provides default \
                variables in which to bin b - tagging efficiency.\n \
                The variables 'btag_jet_pt' & 'btag_jet_eta' are used if defined in the config.",
        )
        variables = ("btag_jet_pt", "btag_jet_eta")

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
        self.task, datasets=datasets, variables=variables, processes=process)


@btag_weights_fixed_wp.init
def btag_weights_fixed_wp_init(self: Producer) -> None:
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
    btag_sf_jec_source = "" if self.jec_source == "Total" else self.jec_source

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


@btag_weights_fixed_wp.setup
def btag_weights_fixed_wp_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    B-tag scale factor weight producer for fixed working points. Requires an external file in the config as under
    ``btag_sf_corr``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "btag_sf_corr": "PATH/TO/POG/BTV/2017_UL/btagging.json.gz",  # noqa
        })


    The name of the algorithm, a list of JEC uncertainty sources which should be
    propagated through the weight calculation, and the column used for b-tagging should
    be given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.btag_sf = BTagSFConfig(
            correction_set="deepJet_shape",
            jec_sources=["Absolute", "FlavorQCD", ...],
            discriminator="btagDeepFlavB",
            corrector_kwargs={...},
        )

    Resources:

        - https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration?rev=26
        - https://indico.cern.ch/event/1096988/contributions/4615134/attachments/2346047/4000529/Nov21_btaggingSFjsons.pdf
    """
    import correctionlib

    # unpack the b-tagging scalefactors
    bundle = reqs["external_files"]
    correction_set_btag_sf_corr = correctionlib.CorrectionSet.from_string(
        bundle.files.btag_sf_corr.load(formatter="gzip").decode("utf-8"),
    )
    # check if btagging scale factor name is given and is available in the correction set
    btagAlgorithm, wp, _ = self.config_inst.x.default_btagAlgorithm
    self.wp = wp  # wp saved for input to correctors
    # fix for change in nomenclature of deepJet scale factors for light hadronFlavour jets
    if self.config_inst.x.run == 2:
        self.btag_sf_incl_corrector = correction_set_btag_sf_corr[f"{btagAlgorithm}_incl"]
    else:
        self.btag_sf_incl_corrector = correction_set_btag_sf_corr[f"{btagAlgorithm}_light"]
    self.btag_sf_comb_corrector = correction_set_btag_sf_corr[f"{btagAlgorithm}_comb"]

    # unpack the b-tagging efficiency
    correction_set_btag_eff_corr = correctionlib.CorrectionSet.from_file(
        reqs["btag_efficiency"].output()["stats"].path,
    )
    if len(correction_set_btag_eff_corr.keys()) != 1:
        raise Exception("Expected exactly one type of btagging efficiencies")

    corrector_name = list(correction_set_btag_eff_corr.keys())[0]
    self.btag_eff_corrector = correction_set_btag_eff_corr[corrector_name]
