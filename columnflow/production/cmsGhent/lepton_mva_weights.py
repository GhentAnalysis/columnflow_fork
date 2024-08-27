from columnflow.production.cmsGhent.fixed_wp_weights import (
    FixedWpConfig,
    fixed_wp_weights,
    fixed_wp_tag,
    fixed_wp_efficiency_hists,
)
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def efficiency_task_import(self):
    from columnflow.tasks.cmsGhent.fixed_wp_efficiency import LeptonMVAEfficiency
    return LeptonMVAEfficiency


LeptonMVAConfig = FixedWpConfig(
    "lepton_mva",
    correction_sets=("sf_Electron", "sf_Muon"),
    object="Lepton",
    objects=["Muon", "Electron"],
    wp_set="wp_values",
    wps="VLoose/Loose/Medium/Tight",
    discriminator="mvaTOP",
    get_sf_file=lambda self, external_files: external_files.lepton_mva.sf,
    get_eff_task=efficiency_task_import,
    flavour_input="pdgId",
    flavour_transform=abs,
    flavour_binning=[11, 13],
)

lepton_mva_id = fixed_wp_tag.derive(
    "jet_lepton_mva",
    cls_dict=dict(config=LeptonMVAConfig),
)

lepton_mva_fixed_wp_weights = fixed_wp_weights.derive(
    "lepton_mva_fixed_wp_weights",
    cls_dict=dict(
        config=LeptonMVAConfig,
        tag_producer=None,
        sf_inputs=lambda syst_variation, wp, flat_input: [
            syst_variation,
            wp,
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

lepton_mva_efficiency_hists = fixed_wp_efficiency_hists.derive(
    "lepton_mva_efficiency_hists",
    cls_dict=dict(config=LeptonMVAConfig),
)