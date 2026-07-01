"""
Optical properties calculation module for generalized transfer matrix method.

This module provides functions to calculate reflection and transmission coefficients
and their corresponding reflectance and transmittance for multilayered optical structures
using the transfer matrix method.

Functions:
    reflection_coeffs: Calculate complex reflection coefficients from wavelength, angle, and structure.
    reflection_coeffs_from_m: Calculate complex reflection coefficients from transfer matrix.
    transmission_coeffs: Calculate complex transmission coefficients from wavelength, angle, and structure.
    transmission_coeffs_from_m: Calculate complex transmission coefficients from transfer matrix.
    calculate_reflection: Calculate reflectance (intensity reflection) for a given wavelength and angle.
    calculate_transmission: Calculate transmittance (intensity transmission) for a given wavelength and angle.

Note:
    All functions support both 'linear' and 'circular' polarization bases for calculations.
    Reflection and transmission coefficients are computed for both s-polarized (TE) and
    p-polarized (TM) waves, as well as their cross-polarization components.
"""

from __future__ import annotations
import numpy as np

from .helpers import basis_selector, calculate_cos_alpha_t, convert_to_wavelength
from .tmm import calculate_structure_m


def reflection_coeffs(wavelength, alpha, structure, basis: str = "linear"):
    wavelength_m = convert_to_wavelength(wavelength)
    m = calculate_structure_m(wavelength_m, alpha, structure)
    return reflection_coeffs_from_m(m, basis=basis)


def reflection_coeffs_from_m(m: np.ndarray, basis: str = "linear"):
    denom = m[3, 3] * m[2, 2] - m[3, 2] * m[2, 3]
    # print("denom:", denom)
    # Evitar warnings por división por cero o NaN
    if denom == 0 or np.isnan(denom):
        r_pp = r_ss = r_ps = r_sp = np.nan
    else:
        r_pp = (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]) / denom
        r_ss = (m[3, 0] * m[2, 3] - m[2, 0] * m[3, 3]) / denom
        r_ps = (m[3, 1] * m[2, 3] - m[3, 3] * m[2, 1]) / denom
        r_sp = (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]) / denom
    # print("r_pp:", r_pp, "\n r_ss:", r_ss, "\n r_ps:", r_ps, "\n r_sp:", r_sp)
    # print("abs r:", [abs(x) for x in (r_pp, r_ss, r_ps, r_sp)])
    # print("\n")
    return basis_selector((r_pp, r_ss, r_ps, r_sp), basis)


def transmission_coeffs(wavelength, alpha, structure, basis: str = "linear"):
    wavelength_m = convert_to_wavelength(wavelength)
    m = calculate_structure_m(wavelength_m, alpha, structure)
    return transmission_coeffs_from_m(m, basis=basis)


def transmission_coeffs_from_m(m: np.ndarray, basis: str = "linear"):
    r_pp, r_ss, r_ps, r_sp = reflection_coeffs_from_m(m)
    t_ss = m[0, 0] + m[0, 2] * r_ss + m[0, 3] * r_sp
    t_sp = m[1, 0] + m[1, 2] * r_ss + m[1, 3] * r_sp
    t_ps = m[0, 1] + m[0, 2] * r_ps + m[0, 3] * r_pp
    t_pp = m[1, 1] + m[1, 2] * r_ps + m[1, 3] * r_pp
    return basis_selector((t_pp, t_ss, t_ps, t_sp), basis)


def calculate_reflection(wavelength, alpha, structure, basis: str = "linear"):
    r = reflection_coeffs(wavelength, alpha, structure, basis=basis)
    return tuple(np.abs(np.asarray(r)) ** 2)


def calculate_transmission(wavelength, alpha, structure, basis: str = "linear"):
    wavelength_m = convert_to_wavelength(wavelength)
    cos_alpha_t, n_inc, n_out, _ = calculate_cos_alpha_t(wavelength_m, alpha, structure)
    t = transmission_coeffs(wavelength, alpha, structure, basis=basis)
    scale = np.real(n_out * cos_alpha_t) / np.real(n_inc * np.cos(alpha))
    if np.isclose(scale, 0.0) or not np.isfinite(scale):
        # Evitar resultados nulos por errores numéricos en el factor de escala
        scale = np.real(n_out * cos_alpha_t / (n_inc * np.cos(alpha)))
    return tuple(scale * (np.abs(np.asarray(t)) ** 2))


# CD reflection
def calculate_circular_dichroism_ref(wavelength, alpha, structure):
    r = calculate_reflection(wavelength, alpha, structure, basis="circular")
    r_rr, r_ll, r_rl, r_lr = r
    # Incidencia R: rr + lr. Incidencia L: ll + rl
    R_r = r_rr + r_lr
    R_l = r_ll + r_rl
    cd_reflection = R_r - R_l
    Refectance = R_r + R_l
    # cd_reflection = abs(R_r - R_l)
    # Refectance = abs(R_r + R_l)
    cd_reflection_norm = cd_reflection / Refectance if Refectance != 0 else 0
    return cd_reflection, cd_reflection_norm, Refectance, R_r, R_l


# CD transmission
def calculate_circular_dichroism_tr(wavelength, alpha, structure):
    t = calculate_transmission(wavelength, alpha, structure, basis="circular")
    t_rr, t_ll, t_rl, t_lr = t
    # Incidencia R: rr + lr. Incidencia L: ll + rl
    T_r = t_rr + t_lr
    T_l = t_ll + t_rl
    cd_transmission = abs(T_r - T_l)
    Transmitance = T_r + T_l
    # cd_transmission = abs(T_r - T_l)
    # Transmitance = abs(T_r + T_l)
    cd_transmission_norm = (
        abs(cd_transmission / Transmitance) if Transmitance != 0 else 0
    )
    return cd_transmission, cd_transmission_norm, Transmitance, T_r, T_l


# LD reflection
def calculate_linear_dichroism_ref(wavelength, alpha, structure):
    r = calculate_reflection(wavelength, alpha, structure, basis="linear")
    r_pp, r_ss, r_ps, r_sp = r
    # Incidencia p: pp + sp. Incidencia s: ss + ps
    R_p = r_pp + r_sp
    R_s = r_ss + r_ps
    LD_reflection = R_p - R_s
    Refectance = R_p + R_s
    # LD_reflection = abs(R_x - R_y)
    # Refectance = abs(R_x + R_y)
    LD_reflection_norm = LD_reflection / Refectance if Refectance != 0 else 0
    return LD_reflection, LD_reflection_norm, Refectance, R_p, R_s


def calculate_linear_dichroism_tr(wavelength, alpha, structure):
    t = calculate_transmission(wavelength, alpha, structure, basis="linear")
    t_pp, t_ss, t_ps, t_sp = t
    # Incidencia p: pp + sp. Incidencia s: ss + ps
    T_p = t_pp + t_sp
    T_s = t_ss + t_ps
    LD_transmission = T_p - T_s
    Transmitance = T_p + T_s
    # LD_transmission = abs(T_x - T_y)
    # Transmitance = abs(T_x + T_y)
    LD_transmission_norm = LD_transmission / Transmitance if Transmitance != 0 else 0
    return LD_transmission, LD_transmission_norm, Transmitance, T_p, T_s
