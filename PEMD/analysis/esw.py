"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


from typing import Union, List, Tuple, Optional
from PEMD.constants import (
    FARADAY_CONSTANT,
)

def esw(
    G_init: float,
    *,
    G_oxid: Optional[float] = None,
    G_red: Optional[float] = None,
    output: Union[str, List[str]] = 'all',
    unit: str = 'hartree',
) -> Union[float, Tuple[float, float, float], List[float]]:

    # ΔG unit conversion factor
    if unit == 'hartree':
        uc = 2.6255E3 * 1e3 # Hartree to J/mol
    elif unit == 'eV':
        uc = 96.48534 * 1e3 # eV to J/mol
    elif unit == 'kcal/mol':
        uc = 4.184043 * 1e3 # kcal/mol to J/mol
    else:
        raise ValueError("unit must be 'hartree','eV' or 'kcal/mol'")

    # 规范输出 key
    if isinstance(output, str):
        keys = [output]
    else:
        keys = output
    keys = [k.lower() for k in keys]
    need_all = 'all' in keys

    values = {}
    if need_all or 'eox' in keys:
        if G_oxid is None:
            raise ValueError("Missing G_oxid, cannot calculate Eox")
        deltaG_ox = (G_init - G_oxid) * uc
        values['eox'] = -deltaG_ox / FARADAY_CONSTANT - 1.46

    if need_all or 'ered' in keys:
        if G_red is None:
            raise ValueError("Missing G_red, cannot calculate Ered")
        deltaG_red = (G_red - G_init) * uc
        values['ered'] = -deltaG_red / FARADAY_CONSTANT - 1.46

    if need_all or 'esw' in keys:
        if 'eox' not in values:
            if G_oxid is None:
                raise ValueError("Missing G_oxid, cannot calculate ESW")
            deltaG_ox = (G_oxid - G_init) * uc
            values['eox'] = -deltaG_ox / FARADAY_CONSTANT - 1.46
        if 'ered' not in values:
            if G_red is None:
                raise ValueError("Missing G_red, cannot calculate ESW")
            deltaG_red = (G_init - G_red) * uc
            values['ered'] = -deltaG_red / FARADAY_CONSTANT - 1.46
        values['esw'] = values['eox'] - values['ered']

    if need_all:
        return values['eox'], values['ered'], values['esw']
    if len(keys) == 1:
        return values[keys[0]]
    return [values[k] for k in keys]


















