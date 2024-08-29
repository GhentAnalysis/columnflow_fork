from columnflow.production.cmsGhent.fixed_wp_weights import (
    FixedWpConfig,
    fixed_wp_weights,
    fixed_wp_tag,
    fixed_wp_efficiency_hists,
)
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def efficiency_task_import():
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
    get_sf_file=lambda external_files: external_files.lepton_mva.sf,
    get_eff_task=efficiency_task_import,
    flavour_input="pdgId",
    flavour_transform=abs,
    flavour_binning=[11, 13],
)

lepton_mva_id = fixed_wp_tag.derive(
    "lepton_mva_id",
    cls_dict=dict(wp_config=LeptonMVAConfig),
)

lepton_mva_fixed_wp_weights = fixed_wp_weights.derive(
    "lepton_mva_fixed_wp_weights",
    cls_dict=dict(
        wp_config=LeptonMVAConfig,
        tag_producer=lepton_mva_id,
        syst_corr_name="syst",
        syst_uncorr_name="stat",
        sf_inputs=lambda self, syst_variation, wp, flat_input: [
            syst_variation,
            wp,
            flat_input.pt,
            flat_input.eta,
        ],
        eff_inputs=lambda self, wp, flat_input: [
            self.flavour_transform(flat_input[self.flavour_input]),
            wp,
            flat_input.pt,
            flat_input.eta,
        ],
    )
)

lepton_mva_efficiency_hists = fixed_wp_efficiency_hists.derive(
    "lepton_mva_efficiency_hists",
    cls_dict=dict(wp_config=LeptonMVAConfig),
)

pass_single_producers = [
    producer_inst.derive(
        producer_inst.cls_name + "_pass_single",
        cls_dict=dict(wp_config=LeptonMVAConfig.copy(single_wp=True, pass_only=True)),
    ) for producer_inst in [
        lepton_mva_id,
        lepton_mva_fixed_wp_weights,
        lepton_mva_efficiency_hists
    ]
]

lepton_mva_id_pass_single = pass_single_producers[0]
lepton_mva_fixed_wp_weights_pass_single = pass_single_producers[1]
lepton_mva_efficiency_hists_pass_single = pass_single_producers[2]

