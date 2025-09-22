"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""

import os
import pandas as pd
from cclib.io import ccopen
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


def homo_lumo_energy(work_dir, log_filename):

    log_file_path = os.path.join(work_dir, log_filename)
    if not os.path.isfile(log_file_path):
        raise FileNotFoundError(f"Log not found: {log_file_path}")

    # 解析量化日志
    try:
        data = ccopen(log_file_path).parse()
    except Exception as e:
        raise RuntimeError(f"Failed to parse {log_file_path}: {e}")

    # 轨道能（通常单位 eV）
    moenergies = getattr(data, "moenergies", None)
    if not moenergies or len(moenergies) == 0:
        raise RuntimeError(f"{log_file_path}: cclib did not provide 'moenergies'.")

    # HOMO 索引（每自旋一个）；无则兜底
    homos = getattr(data, "homos", None)
    if not homos:
        ne = getattr(data, "nelectrons", None)
        if ne is None:
            raise RuntimeError(f"{log_file_path}: missing 'homos' and 'nelectrons'.")
        homos = [int(round(ne / 2.0)) - 1]

    nspin = len(moenergies)

    def _safe(arr, idx):
        try:
            return float(arr[idx])
        except Exception:
            return None

    # α 通道
    homo_a = lumo_a = None
    homo_idx_a = homos[0]
    homo_a = _safe(moenergies[0], homo_idx_a)
    lumo_a = _safe(moenergies[0], homo_idx_a + 1)

    # β 通道（若存在）
    homo_b = lumo_b = None
    if nspin >= 2:
        homo_idx_b = homos[1] if len(homos) > 1 else homos[0]
        homo_b = _safe(moenergies[1], homo_idx_b)
        lumo_b = _safe(moenergies[1], homo_idx_b + 1)

    # 合并：总体 HOMO 取两自旋 HOMO 的最大，总体 LUMO 取两自旋 LUMO 的最小
    homo_candidates = [x for x in (homo_a, homo_b) if x is not None]
    lumo_candidates = [x for x in (lumo_a, lumo_b) if x is not None]
    homo_energy = max(homo_candidates) if homo_candidates else None
    lumo_energy = min(lumo_candidates) if lumo_candidates else None

    unit = "eV"
    result_df = pd.DataFrame({
        "Name": ["HOMO", "LUMO"],
        "Energy ": [
            f"{homo_energy:.6f} {unit}" if homo_energy is not None else f"NA {unit}",
            f"{lumo_energy:.6f} {unit}" if lumo_energy is not None else f"NA {unit}",
        ],
    })

    return result_df















