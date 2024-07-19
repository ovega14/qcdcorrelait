from typing import TypeVar, Optional
Fitter = TypeVar('Fitter')


class Table:
    """
    Table object for collecting results and displaying a table in LaTeX.
    """
    def __init__(self):
        pass

    @staticmethod
    def print_param_line(
        reg_method: str,
        dict_fits: dict[str, Fitter],
        filename: str,
        tag: str,
        outfile: str,
        num_states: Optional[int] = 2
    ) -> None:
        """
        Prints a single line of a table in LaTeX comparing fit parameters across inference methods.

        The table line shows the resulting fit parameters and fit qualities for the chosen 
        regression method for the fits supplied to the function. Simply call this function inside
        of a file with write permissions to send the line of TeX code to the file.

        Args:
            reg_method: Name of the regression method being used
            dict_fits: Dictionary of `corfitter.CorrFitter` objects that have been fit to data
            filename: Name of the specific file inside the fitters to examine
            tag: Keyword for which regime to tabulate, e.g. `corr_o_pred_corrected`
            outfile: Name of the file in which to print the text
            num_states: Number of fit parameters to display, starting with the ground state
        """
        fit = dict_fits[tag]
        print(tag + ':\n', file=outfile)

        a = fit.p[filename + ':a']
        dE = fit.p[filename + ':dE']
        chi2_dof = fit.chi2 / fit.dof
        Q = fit.Q

        string = f'{reg_method} & '
        for i in range(num_states):
            string += f'{a[i]} & {dE[i]} & '
        string += f'{round(chi2_dof, 4)} & {round(Q, 4)} \\\\'
        print(string, file=outfile)
