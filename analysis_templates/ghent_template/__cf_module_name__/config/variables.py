import order as od

from columnflow.util import maybe_import, call_once_on_config

np = maybe_import("numpy")
ak = maybe_import("awkward")

from columnflow.columnar_util import EMPTY_FLOAT


@call_once_on_config()
def add_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `features` producer.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(6, 0.5, 6.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        "genTop_pt",
        expression="genTop.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 300),
        x_title="Gen-level Top pT",
        unit="GeV",
    )
    config.add_variable(
        "jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 300),
        x_title=r"Leading Jet $p_T$",
        unit="GeV",
    )
