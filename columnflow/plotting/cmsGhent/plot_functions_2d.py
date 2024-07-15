import law
from collections import OrderedDict
from columnflow.util import maybe_import

plt = maybe_import("matplotlib.pyplot")
np = maybe_import("numpy")
od = maybe_import("order")
mtrans = maybe_import("matplotlib.transforms")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")

from columnflow.plotting.plot_all import make_plot_2d
from columnflow.plotting.plot_util import (
    apply_variable_settings,
    remove_residual_axis,
    prepare_plot_config_2d,
    prepare_style_config_2d,
    fix_cbar_minor_ticks,
)

def merge_migration_bins(h):
    '''
    binning both axes in equal bins
    '''

    x_edges = h.axes[0].edges
    y_edges = h.axes[1].edges

    # check if edges are subsets of each other
    x_subset_of_y = np.all([x in y_edges for x in x_edges])
    y_subset_of_x = np.all([y in x_edges for y in y_edges])

    # if they are both, no rebinning is needed
    # so return the original histogram
    if x_subset_of_y and y_subset_of_x:
        return h

    if not (x_subset_of_y or y_subset_of_x):
        raise ValueError(
            f"Bin edges of 2D histograms not compatible:\n"
            f"x: {x_edges}\n"
            f"y: {y_edges}"
        )

    # get the indices of the common bin edges and create index tuples
    if x_subset_of_y:
        rebin_y = np.array([list(y_edges).index(x) for x in x_edges])
        rebin_tuples = [slice(int(lo), int(hi)) for lo, hi in zip(rebin_y[:-1], rebin_y[1:])]
        new_edges = x_edges
    if y_subset_of_x:
        rebin_x = np.array([list(x_edges).index(y) for y in y_edges])
        rebin_tuples = [slice(int(lo), int(hi)) for lo, hi in zip(rebin_x[:-1], rebin_x[1:])]
        new_edges = y_edges

    # create a new 2d array with merged bins
    if x_subset_of_y:
        new_h = np.array([ h[:,slc].values().sum(axis=1) for slc in rebin_tuples ])
    if y_subset_of_x:
        new_h = np.array([ h[slc,:].values().sum(axis=0) for slc in rebin_tuples ])

    # initialize a new boost histogram with updated axes
    h_eq_ax = hist.Hist( 
        hist.axis.Variable( new_edges, name=h.axes[0].name, label=h.axes[0].label ), 
        hist.axis.Variable( new_edges, name=h.axes[1].name, label=h.axes[1].label ) 
    )

    # update the bin contents
    h_eq_ax[:,:] = new_h

    # return the new histogram
    return h_eq_ax

def plot_migration_matrices(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Process,  # per process plots
    variable_insts: list[od.Variable],
    style_config: dict = None,
    shape_norm: bool = False,
    yscale: str = None,
    hide_errors: bool = None,
    variable_settings: dict = None,
    initial: str = "Initial",
    label_numbers: bool = False,
    colormap: str = "Blues",
    cms_label: str = "wip",
    **kwargs,
):
    plt.style.use(mplhep.style.CMS)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(10, 10),
        gridspec_kw=dict(width_ratios=[1, 4, 0.5], height_ratios=[4, 1], hspace=0, wspace=0),
    )
    axes[0, 1].sharex(axes[1, 1])
    axes[0, 1].sharey(axes[0, 0])

    remove_residual_axis(hists, "shift")
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    initial_hist = hists.pop(initial)
    [(category, hist_2d)] = hists.items()

    # forcing histograms to have equal bins on gen and reco axis
    hist_2d_eq_ax = merge_migration_bins(hist_2d)
    initial_hist_eq_ax = merge_migration_bins(initial_hist)

    # add all processes into 1 histogram
    projections = [hist_2d.project(v.name) for v in variable_insts]
    projections_eq_ax = [hist_2d_eq_ax.project(v.name) for v in variable_insts]

    migrations = hist_2d / projections[1].values(flow=True)[None]
    migrations_eq_ax = hist_2d_eq_ax / projections_eq_ax[1].values(flow=True)[None]

    plot_config = prepare_plot_config_2d(
        {category_inst: migrations},
        shape_norm=shape_norm,
        zscale="linear",
        colormap=colormap,
    )

    # will add cbar separately!
    plot_config["kwargs"]["cbar"] = False
    plot_config["cbar_kwargs"] |= dict(
        cax=axes[0, 2],
        fraction=1,
    )

    default_style_config = prepare_style_config_2d(
        config_inst=config_inst,
        category_inst=config_inst.get_category(category),
        process_insts=[category_inst],
        variable_insts=variable_insts,
        cms_label=cms_label,
    )

    del default_style_config["legend_cfg"]
    default_style_config["annotate_cfg"]["bbox"] = dict(alpha=0.5, facecolor="white")

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # make main central migration plot
    make_plot_2d(plot_config, style_config, figaxes=(fig, axes[0, 1]))
    if label_numbers:
        for i, x in enumerate(migrations_eq_ax.axes[0].centers):
            for j, y in enumerate(migrations_eq_ax.axes[1].centers):
                if abs(i - j) <= 1:
                    axes[0, 1].text(x, y, f"{migrations_eq_ax[i, j].value:.2f}", ha="center", va="center", size="large")

    cbar = plt.colorbar(axes[0, 1].collections[0], **plot_config["cbar_kwargs"])
    fix_cbar_minor_ticks(cbar)


    # make purity plot
    purity = hist_2d_eq_ax / projections_eq_ax[0].values(flow=True)[:, None]
    purity_diagonal = purity * np.eye(*[len(a) for a in hist_2d_eq_ax.axes[1:]])
    purity_diagonal = purity_diagonal[:, sum]
    purity_diagonal.plot1d(ax=axes[1, 1])
    trans = mtrans.blended_transform_factory(axes[1, 1].transData, axes[1, 1].transAxes)
    if label_numbers:
        for i, x in enumerate(purity_diagonal.axes[0].centers):
            axes[1, 1].text(x, 0.5, f"{purity_diagonal[i].value * 100:.1f}%", rotation="vertical",
                            ha="center", va="center", size="medium", transform=trans)
    axes[1, 1].set_xlabel(axes[0, 1].get_xlabel(), size="medium")
    axes[1, 1].set_ylabel("purity", size="small", loc="bottom")
    axes[1, 1].tick_params(labelleft=False)

    # make efficiency plot
    efficiency = projections_eq_ax[1] / initial_hist_eq_ax.project(variable_insts[1].name).values()
    trans = mtrans.Affine2D().scale(1, -1).rotate_deg(90) + axes[0, 0].transData
    efficiency.plot1d(ax=axes[0, 0], transform=trans)
    trans = mtrans.blended_transform_factory(axes[0, 0].transAxes, axes[0, 0].transData)
    if label_numbers:
        for i, x in enumerate(efficiency.axes[0].centers):
            axes[0, 0].text(0.5, x, f"{efficiency[i].value * 100:.1f}%", rotation="horizontal",
                            ha="center", va="center", size="medium", transform=trans)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 0].set_ylabel(axes[0, 1].get_ylabel(), size="medium")
    axes[0, 0].set_xlabel("efficiency", size="small", loc="left")

    # condition number
    cond = np.linalg.cond(migrations.values())
    axes[1, 0].text(
        0.05, 0.05,
        f"condition\nnumber\n{cond:.1f}",
        transform=axes[1, 0].transAxes,
        size="small",
        va="bottom",
        ha="left",
        color="red",
    )

    # finally remove redundant stuff
    for i in [0, 2]:
        axes[1, i].set_axis_off()
    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    axes[0, 1].set_ylabel(None)
    axes[0, 1].set_xlabel(None)
    return fig, axes
