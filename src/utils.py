import matplotlib.pyplot as plt

def save_plot(
    fig,
    path='./',
    filename=None,
    save_as_pdf=True,
    save_as_png=True,
    save_as_svg=True,
    dpi=800.,
    tight_layout=True,
    show=False,
):
    """
    Args:
        fig: matplotlib figure object
        path: str, the path where the figure is saved, ending with ``/``.
        filename: str, the file name, not including the file type suffix, ending with no ``.``.
    """
    if tight_layout:
        plt.tight_layout()
    if save_as_pdf:
        fig.savefig(
            path+filename+'.pdf',
            dpi=dpi,
        )
    if save_as_png:
        fig.savefig(
            path+filename+'.png',
            dpi=dpi,
        )
    if save_as_svg:
        fig.savefig(
            path+filename+'.svg',
        )
    if show:
        plt.show()
    plt.close()
    return