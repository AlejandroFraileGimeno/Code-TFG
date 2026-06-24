from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .helpers import (
    basis_change,
    calculate_cos_alpha_t,
    euler_mat,
)

c0 = 299792458.0
eps0 = 8.8541878188e-12
mu0 = 1.25663706127e-6
J = np.ones((4, 4), dtype=complex)


def nu_mats(q_x, eps, mu, xi, zeta):
    denom = eps[2, 2] * mu[2, 2] - xi[2, 2] * zeta[2, 2]

    nu_ee_zx = -((mu[2, 2] * eps[2, 0] - xi[2, 2] * zeta[2, 0]) / denom)
    nu_ee_zy = -((mu[2, 2] * eps[2, 1] - xi[2, 2] * (zeta[2, 1] - q_x / c0)) / denom)
    nu_eh_zx = (xi[2, 2] * mu[2, 0] - mu[2, 2] * xi[2, 0]) / denom
    nu_eh_zy = (xi[2, 2] * mu[2, 1] - mu[2, 2] * (xi[2, 1] + q_x / c0)) / denom
    nu_he_zx = (zeta[2, 2] * eps[2, 0] - eps[2, 2] * zeta[2, 0]) / denom
    nu_he_zy = (zeta[2, 2] * eps[2, 0] - eps[2, 2] * (zeta[2, 1] - q_x / c0)) / denom
    nu_hh_zx = -((eps[2, 2] * mu[2, 0] - zeta[2, 2] * xi[2, 0]) / denom)
    nu_hh_zy = -((eps[2, 2] * mu[2, 1] - zeta[2, 2] * (xi[2, 1] + q_x / c0)) / denom)

    nu_e = np.diag([nu_ee_zx, nu_ee_zy, nu_eh_zx, nu_eh_zy])
    nu_h = np.diag([nu_he_zx, nu_he_zy, nu_hh_zx, nu_hh_zy])
    return nu_e, nu_h


def p_mat(omega, q_x, eps, mu, xi, zeta):
    nu_e, nu_h = nu_mats(q_x, eps, mu, xi, zeta)

    p1 = np.array(
        [
            [zeta[1, 0], zeta[1, 1], mu[1, 0], mu[1, 1]],
            [-zeta[0, 0], -zeta[0, 1], -mu[0, 0], -mu[0, 1]],
            [-eps[1, 0], -eps[1, 1], -xi[1, 0], -xi[1, 1]],
            [eps[0, 0], eps[0, 1], xi[0, 0], xi[0, 1]],
        ],
        dtype=complex,
    )
    p2 = np.diag([zeta[1, 2] + q_x / c0, -zeta[0, 2], -eps[1, 2], eps[0, 2]])
    p3 = np.diag([mu[1, 2], -mu[0, 2], xi[1, 2] + q_x / c0, xi[0, 2]])
    return omega * (p1 + p2 @ J @ nu_e + p3 @ J @ nu_h)


def m_mat(d, omega, q_x, eps, mu, xi, zeta):
    return expm(-1j * p_mat(omega, q_x, eps, mu, xi, zeta) * d)


def k_mat(n, cos_theta):
    n_imp = n * np.sqrt(eps0 / mu0)
    return np.array(
        [
            [0, cos_theta, 0, -cos_theta],
            [1, 0, 1, 0],
            [n_imp * cos_theta, 0, -n_imp * cos_theta, 0],
            [0, -n_imp, 0, -n_imp],
        ],
        dtype=complex,
    )


def calculate_layer_m(wavelength, q_x, layer):
    omega = 2 * np.pi * c0 / wavelength
    eul = euler_mat(layer.theta, layer.phi, layer.psi)

    eps = eps0 * basis_change(layer.eps(wavelength), np.linalg.inv(eul))
    mu = mu0 * basis_change(layer.mu(wavelength), np.linalg.inv(eul))
    # print("eps:", eps)
    # print("wavelength:", wavelength)
    # print("\n")
    xi = basis_change(layer.xi(wavelength), np.linalg.inv(eul))
    zeta = basis_change(layer.zeta(wavelength), np.linalg.inv(eul))
    return m_mat(layer.d, omega, q_x, eps, mu, xi, zeta)


def calculate_structure_m(wavelength, alpha, structure):
    sup, sub = structure.superstrate, structure.substrate

    if not (
        np.allclose(np.diag(sup.eps(wavelength)), sup.eps(wavelength)[0, 0])
        and np.allclose(np.diag(sub.eps(wavelength)), sub.eps(wavelength)[0, 0])
        and np.allclose(np.diag(sup.mu(wavelength)), sup.mu(wavelength)[0, 0])
        and np.allclose(np.diag(sub.mu(wavelength)), sub.mu(wavelength)[0, 0])
    ):
        raise ValueError("The superstrate and substrate must be isotropic")

    cos_alpha_t, n_inc, n_out, q_x = calculate_cos_alpha_t(wavelength, alpha, structure)
    k_sup = k_mat(n_inc, np.cos(alpha))
    k_sub = k_mat(n_out, cos_alpha_t)

    if structure.layers:
        lay_ms = [calculate_layer_m(wavelength, q_x, lay) for lay in structure.layers]
        if len(structure.layers) > 1:
            return (
            np.linalg.inv(k_sub) @ np.linalg.multi_dot(list(reversed(lay_ms))) @ k_sup
        )
        else:
            return np.linalg.inv(k_sub) @ lay_ms[0] @ k_sup 
    return np.linalg.inv(k_sub) @ k_sup
