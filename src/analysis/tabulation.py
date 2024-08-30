from typing import TypeVar, Optional, List
Fitter = TypeVar('Fitter')


class Table:
    """
    Table object for collecting results and displaying a table in LaTeX.

    Abstract base class.
    """
    def write_line(*args):
        raise NotImplementedError('Table must be implemented.')


class FitParamsTable(Table):
    """
    Table for comparing fit parameters across different ML models within the same regime.

    Args:
        dict_fits:
        filename:
        tag:
        num_states:
        reg_methods:
        include_truth:
    """
    def __init__(
        self, 
        dict_fits: dict[str, dict[str, Fitter]],
        reg_methods: List[str],
        tag: str,
        filename: str,
        num_states: Optional[int] = 2, 
        include_truth: bool = True,
    ):
        self.dict_fits = dict_fits  # {reg_method: {tag: fit}}
        self.filename = filename
        self.tag = tag
        self.num_states = num_states
        self.include_truth = include_truth
        self.reg_methods = reg_methods

    @staticmethod
    def write_line(
        reg_method: str,
        dict_fits: dict[str, Fitter],
        filename: str,
        tag: str,
        num_states: Optional[int] = 2
    ) -> str:
        """
        Writes a single line of a table in LaTeX comparing fit parameters across inference methods.

        The table line shows the resulting fit parameters and fit qualities for the chosen 
        regression method for the fits supplied to the function. Simply call this function and 
        print its result inside of a file with write permissions to send the line of TeX code to 
        the file.

        Args:
            reg_method: Name of the regression method being used
            dict_fits: Dictionary of `corfitter.CorrFitter` objects that have been fit to data
            filename: Name of the specific file inside the fitters to examine
            tag: Keyword for which regime to tabulate, e.g. `corr_o_pred_corrected`
            outfile: Name of the file in which to print the text
            num_states: Number of fit parameters to display, starting with the ground state
        """
        fit = dict_fits[tag]

        a = fit.p[filename + ':a']
        dE = fit.p[filename + ':dE']
        chi2_dof = fit.chi2 / fit.dof
        Q = fit.Q

        line = f'{reg_method} & '
        for i in range(num_states):
            line += f'{a[i]} & {dE[i]} & '
        line += f'{round(chi2_dof, 4)} & {round(Q, 4)} \\\\'
        return line

    def make_header(self) -> str:
        """
        Constructs the table header.
        """
        num_cols = 2 * self.num_states + 2
        header: str = '\\hline\n& \\multicolumn{%s}{c|}{\\textbf{Fit Parameters}} \\\\\n\\hline\n' % num_cols
        header += '\\textbf{Regression Method} & '
        for i in range(self.num_states):
            header += f'$a_{i}$ & $dE_{i}$ & '
        header += '$\\chi^2 / \\mathrm{dof}$ & $Q$ \\\\'
        return header
    
    def __str__(self):
        table = self.make_header()
        if self.include_truth:
            fit_truth = self.dict_fits[self.reg_methods[0]]['corr_o_truth']  # same for all methods
            a_truth = fit_truth.p[self.filename + ':a']
            dE_truth = fit_truth.p[self.filename + ':dE']
            chi2_dof_truth = fit_truth.chi2 / fit_truth.dof
            Q_truth = fit_truth.Q
            
            table += '\n\\hline\n\\rowcolor{green!40}\n'
            table += 'TRUTH & '
            for i in range(self.num_states):
                table +=f'{a_truth[i]} & {dE_truth[i]} & '
            table += f'{round(chi2_dof_truth, 4)} & {round(Q_truth, 4)} \\\\'
            table += '\n'

        for method in self.reg_methods:
            dict_fit = self.dict_fits[method]
            table += self.write_line(method, dict_fit, self.filename, self.tag, self.num_states)
            table += '\n'
        table += '\\hline'
        return table

