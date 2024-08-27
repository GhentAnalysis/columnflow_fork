from columnflow.production.cmsGhent.fixed_wp_weights import (
    FixedWpConfig,
    fixed_wp_weights,
    fixed_wp_tag,
    fixed_wp_efficiency_hists,
)
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def efficiency_task_import(self):
    from columnflow.tasks.cmsGhent.fixed_wp_efficiency import BTagEfficiency
    return BTagEfficiency


BTagConfigRun2 = FixedWpConfig(
    "btag",
    correction_sets=("deepJet_incl", "deepJet_comb"),
    object="Jet",
    objects=["light", "heavy"],
    object_mapping=[lambda obj: obj[obj.hadronFlavour == 0], lambda obj: obj[obj.hadronFlavour != 0]],
    wp_set="deepJet_wp_values",
    wps="L/M/T",
    discriminator="btagDeepFlavB",
    algorithm="deepJet",
    get_sf_file=lambda self, external_files: external_files.btag_sf_corr,
    get_eff_task=efficiency_task_import,
    flavour_input="hadronFlavour",
    flavour_binning=[0, 4, 5],
)

jet_btag = fixed_wp_tag.derive(
    "jet_btag",
    cls_dict=dict(config=BTagConfigRun2),
)

btag_fixed_wp_weights = fixed_wp_weights.derive(
    "btag_fixed_wp_weights",
    cls_dict=dict(
        config=BTagConfigRun2,
        tag_producer=None,
        sf_inputs=lambda syst_variation, wp, flat_input: [
            syst_variation,
            wp,
            flat_input.hadronFlavour,
            flat_input.pt,
            abs(flat_input.eta)
        ],
        eff_inputs=lambda wp, flat_input: [
            abs(flat_input.pdgId),
            # currently set hard max on pt since overflow could not be changed in correctionlib
            # (could also manually change the flow)
            ak.min([flat_input.pt, 999 * ak.ones_like(flat_input.pt)], axis=0),
            abs(flat_input.eta),
            wp,
        ],
    )
)

btag_efficiency_hists = fixed_wp_efficiency_hists.derive(
    "btag_efficiency_hists",
    cls_dict=dict(config=BTagConfigRun2),
)