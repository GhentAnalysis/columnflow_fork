from __future__ import annotations

import law
import order as od
from typing import Iterable, Sequence, Callable
import time
from collections import OrderedDict
import dataclasses

from columnflow.production import Producer, producer
from columnflow.weight import WeightProducer, weight_producer
from columnflow.selection import SelectionResult

from columnflow.util import maybe_import, InsertableDict, DotDict
from columnflow.columnar_util import set_ak_column, layout_ak_array, Route, has_ak_column, optional_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
hist = maybe_import("hist")
correctionlib = maybe_import("correctionlib")

logger = law.logger.get_logger(__name__)


@dataclasses.dataclass
class FixedWpConfig:
    tag_name: str
    correction_sets: Sequence[str] | str
    object: str
    wp_set: str
    wps: str | list[str]
    discriminator: str
    get_sf_file: Callable
    get_eff_task: Callable
    flavour_input: str
    flavour_binning: Sequence[int]
    flavour_transform: Callable = lambda x: x
    algorithm: str | None = None
    objects: Sequence[str] | None = None
    object_mapping: Sequence[Callable] | dict[str, Callable] | None = None
    default_eff_variables: Sequence[str] | None = None
    discriminator_range: tuple[float, float] = (0, 1)
    single_wp: bool = False
    pass_only: bool = False
    nano_objects: list[str] = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.correction_sets, str):
            self.correction_sets = [self.correction_sets]
        if self.objects is None:
            self.objects = [self.object]
        assert len(self.correction_sets) == len(self.objects), "need correction set for each object and v.v."
        self.nano_objects = self.objects if self.object_mapping is None else [self.object]
        if isinstance(self.wps, str):
            self.wps = self.wps.split("/")
        if self.algorithm is None:
            self.algorithm = self.discriminator
        if self.default_eff_variables is None:
            var_prefix = f"{self.tag_name}_{self.object.lower()}"
            self.default_eff_variables = (f"{var_prefix}_pt", f"{var_prefix}_eta")
        if isinstance(self.object_mapping, dict):
            assert set(self.objects) == set(self.object_mapping)
        elif self.object_mapping is not None:
            self.object_mapping = dict(zip(self.objects, self.object_mapping))
        if self.pass_only:
            self.pass_only &= self.single_wp
            if not self.pass_only:
                logger.warning_once("pass_only option is ignored since single_wp option is disabled")

    def copy(self, /, **changes):
        return dataclasses.replace(self, **changes)


def init_fixed_wp(self: Producer | WeightProducer, add_weight_inputs_vars=True):
    if self.wp_config is None:
        return
    for key, value in dataclasses.asdict(self.wp_config).items():
        setattr(self, key, value)

    self.uses.update([f"{obj}.{self.discriminator}" for obj in self.nano_objects])

    if add_weight_inputs_vars:
        self.uses.update({f"{obj}.{self.flavour_input}" for obj in self.nano_objects})
        if f"default_{self.tag_name}_variables" not in self.config_inst.aux:
            logger.warning_once(
                f"no default {self.tag_name} efficiency variables defined in config",
                f"Config does not have an attribute x.default_{self.tag_name}_variables that provides default \
                variables in which to bin {self.tag_name} efficiency.\n \
                The variables '{' & '.join(self.default_eff_variables)}' are used if defined in the config.",)
        self.variables = self.config_inst.x(f"default_{self.tag_name}_variables", self.default_eff_variables)
        self.variable_insts = list(map(self.config_inst.get_variable, self.variables))
        self.uses.update({
            inp.replace(self.object, obj)
            for obj in self.nano_objects
            for variable_inst in self.variable_insts
            for inp in (
                [variable_inst.expression]
                if isinstance(variable_inst.expression, str)
                else variable_inst.x("inputs", [])
            )
        })


def setup_fixed_wp(self: Producer | WeightProducer, reqs: dict, *args, **kwargs):
    if self.wp_config is not None:
        bundle = reqs["external_files"]
        correction_set = correctionlib.CorrectionSet.from_string(
            self.get_sf_file(bundle.files).load(formatter="gzip").decode("utf-8"),
        )
        wp_set = correction_set[self.wp_set]
        self.wp_values = OrderedDict([(wp, wp_set.evaluate(wp)) for wp in self.wps])
        return correction_set


def req_fixed_wp(self: Producer | WeightProducer, reqs: dict):
    if self.wp_config is not None:
        from columnflow.tasks.external import BundleExternalFiles
        reqs["external_files"] = BundleExternalFiles.req(self.task)


@producer(
    wp_config=None,
    setup_func=setup_fixed_wp,
    requires_func=req_fixed_wp,
)
def fixed_wp_tag(
    self: Producer,
    events: ak.Array,
    working_points: Iterable[str] = None,
    object_mask: ak.Array[bool] = None,
    **kwargs,
) -> ak.Array:
    if self.wp_config is None:
        logger.warning_once(self.cls_name + " no config",
            f"no {self.cls_name} config defined. Not doing anything"
        )
        return events
    if working_points is None:
        working_points = self.wps
    for wp in working_points:
        for obj in self.nano_objects:
            tag = events[obj][self.discriminator] >= self.wp_values[wp]
            if object_mask is not None:
                tag = tag & object_mask
            events = set_ak_column(events, f"{obj}.{self.tag_name}_{wp}", tag)

    return events


@fixed_wp_tag.init
def fixed_wp_tag_init(self: Producer):
    if self.wp_config is not None:
        init_fixed_wp(self, add_weight_inputs_vars=False)
        self.produces = optional_column({f"{obj}.{self.tag_name}_{wp}" for obj in self.objects for wp in self.wps})


@weight_producer(
    wp_config=None,
    tag_producer=fixed_wp_tag,
    mc_only=True,
    get_weight_name=lambda self: f"{self.tag_name}_weight",
    syst_corr_name="syst",
    syst_uncorr_name="stat",
    get_dataset_groups=lambda self: self.config_inst.x(f"{self.tag_name}_dataset_groups", None),
    sf_inputs=lambda self, syst_variation, wp, flat_input: [syst_variation, wp, flat_input.pt, abs(flat_input.eta)],
    eff_inputs=lambda self, wp, flat_input: [
        self.flavour_transform(flat_input[self.flavour_input]),
        wp,
        flat_input.pt,
        abs(flat_input.eta),
    ],
)
def fixed_wp_weights(
    self: Producer,
    events: ak.Array,
    working_points: Iterable[str] | str = None,
    object_mask: ak.Array[bool] | dict[ak.Array[bool]] = None,
    **kwargs,
) -> ak.Array:
    if self.wp_config is None:
        logger.warning_once(self.cls_name + " no config",
            f"no {self.cls_name} config defined. Not doing anything"
        )
        return events

    if working_points is None:
        working_points = self.wps
    else:
        working_points = sorted(law.util.make_list(working_points), key=lambda x: self.wps.index(x))

    if self.object_mapping is not None or len(self.objects) == 1:
        assert self.object in events.fields, f"cannot find {self.object} in events array"
        if object_mask is None:
            object_mask = Ellipsis
        else:
            assert isinstance(object_mask, ak.Array), "require one mask for {self.object}"
        object_data = {obj: obj_map(events[self.object][object_mask]) for obj, obj_map in self.object_mapping.items()}
    else:
        if object_mask is None:
            object_mask = {obj: Ellipsis for obj in self.objects}
        else:
            assert isinstance(object_mask, dict) and set(object_mask) == set(self.objects), \
                f"need an object mask for each object in {self.objects}"
        object_data = {obj: events[obj][object_mask[obj]] for obj in self.objects}

    # helper to create and store the weight
    def add_weight(obj, syst_variation, wps):
        # define a mask that selects the correct flavor to assign to, depending on the systematic

        sf_corrector = self.correctors[obj]
        data = object_data[obj]

        weight = np.ones(len(data))

        wps = law.util.make_list(wps)

        if not self.pass_only:
            wps = [None, *wps]
        for i, wp in enumerate(wps):
            next_wp = wps[i + 1] if wp != wps[-1] else None

            if wp is None:
                wp_mask = ~data[f"{self.tag_name}_{next_wp}"]
            else:
                wp_mask = data[f"{self.tag_name}_{wp}"]
                if next_wp is not None:
                    wp_mask = wp_mask & (~data[f"{self.tag_name}_{next_wp}"])

            wp_data = data[wp_mask]
            shape_reference = wp_data[self.flavour_input]
            flat_wp_data = ak.flatten(wp_data, axis=1)

            # get efficiencies and scale factors for this and next working point
            def sf_eff_wp(working_point, none_value=0.):
                if working_point is None:
                    return (none_value,) * 2
                sf_inputs = self.sf_inputs(syst_variation, working_point, flat_wp_data)
                sf = sf_corrector.evaluate(*sf_inputs)
                if self.pass_only:
                    return sf, none_value
                eff_inputs = self.eff_inputs(working_point, flat_wp_data)
                eff = self.eff_corrector.evaluate(*eff_inputs)
                return sf, eff

            sf_this_wp, eff_this_wp = sf_eff_wp(wp, none_value=1.)
            sf_next_wp, eff_next_wp = sf_eff_wp(next_wp, none_value=0.)

            # calculate the event weight following:
            # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
            weight_flat = (sf_this_wp * eff_this_wp - sf_next_wp * eff_next_wp) / (eff_this_wp - eff_next_wp)

            # enforce the correct shape and create the product over all objects per event
            weight = weight * ak.prod(layout_ak_array(weight_flat, shape_reference), axis=1, mask_identity=False)

        column_name = f"{self.weight_name}{f'_{wps[0]}' if self.single_wp else ''}_{obj}"
        if syst_variation != "central":
            column_name += "_" + syst_variation.replace(self.syst_uncorr_name, str(self.config_inst.x.year))

        if ak.any((weight == np.inf) | ak.is_none(ak.nan_to_none(weight)) | ak.any(weight < 0)):
            weight = ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
            logger.warning_once(
                "weight column has an infinite, Nan or negative value",
                f"weight column events.{column_name} has an infinite, Nan or negative value and is set to 1. " +
                f"Make sure the {self.tag_name} efficiency is defined and physical in all bins!",
            )

        return set_ak_column(events, column_name, weight, value_type=np.float32)

    # nominal weight and those of all method intrinsic uncertainties
    for wps in (working_points if self.single_wp else [working_points]):
        for obj in self.objects:
            events = add_weight(obj, "central", wps)
            # only calculate up and down variations for nominal shift
            if self.local_shift_inst.is_nominal:
                for direction in ["up", "down"]:
                    for corr in ["", self.syst_uncorr_name, self.syst_corr_name]:
                        variation = direction if not corr else f"{direction}_{corr}"
                        events = add_weight(obj, variation, wps)

        # nominal weights:
        weight_name = self.weight_name + (f'_{wps}' if self.single_wp else '')
        nominal = np.prod([events[f"{weight_name}_{fg}"] for fg in self.objects], axis=0)
        events = set_ak_column(events, self.weight_name, nominal)

    return events


@fixed_wp_weights.init
def fixed_wp_weights_init(
        self: Producer,
) -> None:
    if self.wp_config is None:
        return
    init_fixed_wp(self)

    self.uses.add(self.tag_producer)
    self.weight_name = self.get_weight_name()

    # depending on the requested shift_inst, there are three cases to handle:
    #   1. when the nominal shift is requested, the central weight and all variations related to the
    #      method-intrinsic shifts are produced: year-uncorrelated and -correlated varations, and combination.
    #   3. when any other shift is requested, only create the central weight column

    shift_inst = getattr(self, "local_shift_inst", None)
    if not shift_inst:
        return

    produces = {self.weight_name}
    prod_weight_name = self.weight_name
    if self.single_wp:
        prod_weight_name += "_{%s}" % ",".join(self.wps)

    for obj in self.objects:
        # nominal columns
        produces.add(f"{prod_weight_name}_{obj}")
        if shift_inst.is_nominal:
            produces.update({
                f"{prod_weight_name}_{obj}_{direction}" + ("" if not corr else f"_{corr}")
                for direction in ["up", "down"]
                for corr in ["", self.syst_corr_name, self.config_inst.x.year]
            })
    if self.single_wp:
        produces = optional_column(produces)
    self.produces.update(produces)


@fixed_wp_weights.setup
def fixed_wp_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    if self.wp_config is None:
        return
    all_correction_sets = setup_fixed_wp(self, reqs)

    self.correctors = {obj: all_correction_sets[c_set] for obj, c_set in zip(self.objects, self.correction_sets)}
    if self.pass_only:
        return
    # unpack the b-tagging efficiency
    correction_set_eff = correctionlib.CorrectionSet.from_file(
        reqs["efficiency"].output()["stats"].path,
    )
    if len(correction_set_eff.keys()) != 1:
        raise Exception("Expected exactly one type of efficiencies")

    self.eff_corrector = correction_set_eff[list(correction_set_eff.keys())[0]]


@fixed_wp_weights.requires
def fixed_wp_weights_requires(self: Producer, reqs: dict) -> None:
    if self.wp_config is None:
        return
    req_fixed_wp(self, reqs)

    if self.pass_only:
        return
    efficiency_task = self.get_eff_task()

    # require efficiency to be ran for the dataset groups
    # default value of datasets to calculate the efficiency is the dataset of the produce task
    datasets = [self.dataset_inst.name]
    process = self.dataset_inst.processes.names()[0]

    if (data_set_groups := self.get_dataset_groups()) is None:
        logger.warning_once(
            f"no default {self.tag_name} efficiency dataset groups defined in config",
            f"Config does not have an attribute 'x.{self.tag_name}_dataset_groups' that provides  \
                    default groupings of datasets for {self.tag_name} efficiency calculation.\n"
            f"The dataset {self.dataset_inst.name} is used to calculate but defining one is recommended.\n"
            f"e.g.: config.x.{self.tag_name}_dataset_groups" " = {'ttx': ['ttztollnunu_m10_amcatnlo','tt_sl_powheg']}",
        )
    else:
        for group in data_set_groups:
            # check if dataset is in data group
            if self.dataset_inst.name in data_set_groups[group]:
                datasets = data_set_groups[group]
                if group in self.config_inst.processes.names():
                    process = group  # only for plotting text
                break

    reqs["efficiency"] = efficiency_task.req(
        self.task,
        datasets=datasets,
        variables=self.variables,
        processes=process,
    )


@producer(
    wp_config=None,
    uses={"mc_weight"},
    # only run on mc
    mc_only=True,
    get_no_tag_selection=lambda self, results: results.x(f"event_no_{self.tag_name}", None),
    requires_func=req_fixed_wp,
)
def fixed_wp_efficiency_hists(
    self: Producer,
    events: ak.Array,
    results: SelectionResult,
    hists: DotDict | dict = None,
    **kwargs,
) -> ak.Array:
    if hists is None:
        logger.warning_once(self.cls_name + " no config",
            f"no {self.cls_name} config defined. Not doing anything"
        )
        return events
    if self.wp_config is None:
        logger.warning_once(self.cls_name + " did not get any histograms")
        return events

    no_tag_selection = self.get_no_tag_selection(results)
    assert no_tag_selection is not None, f"results does not contain mask without {self.tag_name} selection"

    # jet selection
    if self.object_mapping is None:
        object_data = ak.concatenate(
            [events[obj][results.objects[obj][obj]] for obj in self.objects],
            axis=1
        )
    else:
        object_data = events[self.object][results.objects[self.object][self.object]]

    # event selection
    object_data = object_data[no_tag_selection]

    selected_events = ak.Array({
        self.object: object_data,
        "mc_weight": events.mc_weight[no_tag_selection],
    })

    histogram = hist.Hist.new.IntCat(self.flavour_binning, name=self.flavour_input)  # Jet hadronFlavour 0, 4, or 5
    # add variables for binning the efficiency
    for var_inst in self.variable_insts:
        histogram = histogram.Var(
            var_inst.bin_edges,
            name=var_inst.name,
            label=var_inst.get_full_x_title(),
        )
    hists[f"{self.tag_name}_efficiencies"] = histogram.Weight()

    fill_kwargs = {
        # broadcast event weight and process-id to jet weight
        self.flavour_input: ak.flatten(self.flavour_transform(object_data[self.flavour_input])),
        "weight": ak.flatten(ak.broadcast_arrays(selected_events.mc_weight, object_data[self.flavour_input])[0]),
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
    hists[f"{self.tag_name}_efficiencies"].fill(**fill_kwargs)
    hists[f"{self.tag_name}_efficiencies"].name = f"{self.algorithm}({self.discriminator})"

    return events


@fixed_wp_efficiency_hists.init
def fixed_wp_efficiency_hists_init(self: Producer) -> None:
    init_fixed_wp(self)


@fixed_wp_efficiency_hists.setup
def fixed_wp_efficiency_hists_setup(
        self: Producer,
        reqs: dict,
        inputs: dict,
        reader_targets: InsertableDict,
) -> None:
    if self.wp_config is None:
        return
    setup_fixed_wp(self, reqs)
    self.variable_insts.insert(0, od.Variable(
        name=f"{self.tag_name}_wp",
        expression=f"{self.object}.{self.discriminator}",
        binning=[self.discriminator_range[0], *self.wp_values.values(), self.discriminator_range[1]],
        x_labels=["U", *self.wps],
    ))

