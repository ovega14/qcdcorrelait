import gvar as gv


def make_prior(
    filename: str,
    *,
    ne: int,
    no: int
) -> dict[str, gv.GVar]:
    """
    Constructs priors for a given file of correlator data to be used in fitting.

    *Note: Generally want to use a higher number of odd states than even states.

    Args:
        filename: The name of the subfile containing correlator data
        ne: Number of 'even' or 'non-oscillating' states to include
        no: Number of 'odd' or 'oscillating' states to include
    
    Returns:
        Dictionary of fit parameters and their corresponding priors as Gvar objects.
    """
    # TODO: fix numbers of states for other mass combinations
    prior = gv.BufferDict()

    #prior[filename + ':a'] = gv.gvar(ne*['0.0(0.5)'])
    #prior[filename + ':ao'] = gv.gvar(no*['0.0(0.5)'])
    #prior[filename + ':a'] = gv.gvar(ne*['0.50(20)'])
    #prior[filename + ':ao'] = gv.gvar(no*['0.1(1.0)'])
    prior[filename + ':a'] = gv.gvar(ne*['0.5(1.0)'])
    prior[filename + ':ao'] = gv.gvar(no*['0.0(0.1)'])

    if filename.endswith('P5-P5_RW_RW_d_d_m0.164_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.400(5)', '0.20(5)', '0.28(5)', '0.6(2)', '1.0(2)'][:ne])
        prior[filename + ':dEo'] = gv.gvar(['0.25(5)', '0.25(5)', '0.10(10)', '0.10(10)', '0.20(10)'][:no])
        #prior[filename + ':a'][0] = gv.gvar('0.050(5)')
        #if ne >= 5:
        #    prior[filename +':a'][4] = gv.gvar('0.5(5)')
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.1827_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.40(5)', '0.20(5)', '0.20(10)'])
        prior[filename + ':dEo'] = gv.gvar(['0.30(5)', '0.20(5)', '0.20(10)'])
    
    elif filename.endswith('P5-P5_RW_RW_d_d_m0.365_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['0.65(25)', '0.27(5)', '0.62(5)'])
        prior[filename + ':dEo'] = gv.gvar(['0.65(25)', '0.06(10)', '0.28(5)'])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.548_m0.01555_p000'):
        prior[filename + ':dEo'] = gv.gvar(['0.85(05)', '0.10(5)', '0.25(10)', '0.25(30)', '0.25(30)'])
        prior[filename + ':dE'] = gv.gvar(['0.85(05)', '0.20(5)', '0.30(10)', '0.30(20)', '0.30(20)'])

    elif filename.endswith('P5-P5_RW_RW_d_d_m0.843_m0.01555_p000'):
        prior[filename + ':dE'] = gv.gvar(['1.15(2)', '0.11(5)', '0.25(10)'])
        prior[filename + ':dEo'] = gv.gvar(['1.2(2)', '0.160(5)', '0.19(10)'])

    return prior
