from collections import OrderedDict
import law

from columnflow.tasks.cutflow import PlotCutflowVariables1D
from luigi.parameter import ParameterVisibility


class PlotInclusiveVariables1D(PlotCutflowVariables1D):
    def run_postprocess(self, hists, category_inst, variable_insts):
        # resolve plot function
        if self.plot_function == law.NO_STR:
            self.plot_function = (
                self.plot_function_processes if self.per_plot == "processes" else self.plot_function_steps
            )

        import hist

        if len(variable_insts) != 1:
            raise Exception(f"task {self.task_family} is only viable for single variables")

        outputs = self.output()["plots"]
        step_hists = OrderedDict(
            (process_inst.copy_shallow(), h[{"step": hist.loc(self.initial_step)}])
            for process_inst, h in hists.items()
        )

        # call the plot function
        fig, _ = self.call_plot_func(
            self.plot_function,
            hists=step_hists,
            config_inst=self.config_inst,
            category_inst=category_inst.copy_shallow(),
            variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
            style_config={},
            **self.get_plot_parameters(),
        )

        # save the plot
        for outp in outputs[self.initial_step]:
            outp.dump(fig, formatter="mpl")


PlotCutflowVariables1D.only_final_step = PlotCutflowVariables1D.only_final_step.copy(
    visibility=ParameterVisibility.PRIVATE,
)
PlotCutflowVariables1D.per_plot = PlotCutflowVariables1D.per_plot.copy(
    visibility=ParameterVisibility.PRIVATE,
)
PlotCutflowVariables1D.selector_steps = PlotCutflowVariables1D.selector_steps.copy(
    default=tuple(),
    visibility=ParameterVisibility.PRIVATE,
)
