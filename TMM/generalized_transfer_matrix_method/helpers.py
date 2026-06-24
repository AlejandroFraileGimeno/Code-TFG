from __future__ import annotations

import numpy as np


def convert_to_wavelength(omega: float) -> float:
    wavelengths = 1.0 / (omega * 100.0)
    return float(wavelengths)


def basis_change(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.linalg.inv(t) @ m @ t


def euler_mat(theta: float, phi: float, psi: float) -> np.ndarray:
    return np.array(
        [
            [
                np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi),
                np.sin(theta) * np.sin(phi),
            ],
            [
                np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
                -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
                -np.sin(theta) * np.cos(phi),
            ],
            [np.sin(theta) * np.sin(psi), np.sin(theta) * np.cos(psi), np.cos(theta)],
        ],
        dtype=complex,
    )


def circ_coeffs(c_pp: complex, c_ss: complex, c_ps: complex, c_sp: complex):
    t_mat = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]], dtype=complex)
    c_linear = np.array([[c_pp, c_ps], [c_sp, c_ss]], dtype=complex)
    c_rr, c_rl, c_lr, c_ll = (basis_change(c_linear, np.linalg.inv(t_mat))).ravel()
    return c_rr, c_ll, c_rl, c_lr


def basis_selector(coeffs, basis: str):
    if basis == "linear":
        return coeffs
    if basis == "circular":
        return circ_coeffs(*coeffs)
    raise ValueError(f"Unknown basis={basis!r}. Use 'linear' or 'circular'.")


def calculate_cos_alpha_t(wavelength: float, alpha: float, structure):
    n_inc = np.sqrt(
        complex(
            structure.superstrate.eps(wavelength)[0, 0]
            * structure.superstrate.mu(wavelength)[0, 0]
        )
    )
    n_out = np.sqrt(
        complex(
            structure.substrate.eps(wavelength)[0, 0]
            * structure.substrate.mu(wavelength)[0, 0]
        )
    )
    q_x = n_inc * np.sin(alpha)

    sin_alpha_t = (n_inc / n_out) * np.sin(alpha)
    cos_alpha_t = np.sqrt(1 - sin_alpha_t**2)

    ret = np.real(n_out * cos_alpha_t)
    imt = np.imag(n_out * cos_alpha_t)

    if np.real(n_out) > 0:
        if np.imag(n_out) > 0:
            if imt < 0:
                cos_alpha_t *= -1
        elif np.isclose(np.imag(n_out), 0, atol=1e-8):
            if np.isclose(imt, 0, atol=1e-14):
                if ret < 0:
                    cos_alpha_t *= -1
            elif imt < 0:
                cos_alpha_t *= -1
    elif np.real(n_out) < 0 and np.imag(n_out) < 0:
        if imt < 0:
            cos_alpha_t *= -1

    return cos_alpha_t, n_inc, n_out, q_x
