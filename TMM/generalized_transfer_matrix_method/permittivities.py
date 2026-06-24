from __future__ import annotations

import numpy as np

import pathlib
from .types import Layer
from scipy.interpolate import interp1d

c0 = 299792458.0


def lorentz_osc(f: float, f_lo: float, f_to: float, gamma: float) -> complex:
    num = f_lo**2 - f**2 - 1j * gamma * f
    den = f_to**2 - f**2 - 1j * gamma * f
    return num / den


def eps_drude(f: float, f_p: float, gamma: float, eps_inf: float = 1.0) -> complex:
    return eps_inf - f_p**2 / (f**2 + 1j * f * gamma)


def eps_vacuum(_: float) -> np.ndarray:
    return np.eye(3, dtype=complex)


def mu_vacuum(_: float) -> np.ndarray:
    return np.eye(3, dtype=complex)


def xi_vacuum(_: float) -> np.ndarray:
    return np.zeros((3, 3), dtype=complex)


def zeta_vacuum(_: float) -> np.ndarray:
    return np.zeros((3, 3), dtype=complex)


def Air() -> Layer:
    return Layer()


def Air_gap(d: float, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> Layer:
    """
    Capa de aire con espesor finito (gap).
    d en metros.
    """
    return Layer(d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# # Glass
# =============================================================================


def eps_glass(eps_val: float = 1.23) -> np.ndarray:
    return np.eye(3, dtype=complex) * eps_val


def Glass(eps_val: float = 1.23) -> Layer:
    return Layer(eps=lambda _: eps_glass(eps_val))


# =============================================================================
# # Ag
# =============================================================================
def eps_Ag(wavelength: float) -> np.ndarray:
    f = c0 / wavelength
    val = eps_drude(f, 2.152e15, 1 / 17e-15, 5.0)
    return np.eye(3, dtype=complex) * val


def Ag(d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> Layer:
    return Layer(eps=eps_Ag, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# # Au
# =============================================================================
def eps_Au(wavelength: float) -> np.ndarray:
    f = c0 / wavelength
    val = eps_drude(f, 2.183e15, 1.7410e13, 9.84)
    # print("eps_Au:", val)
    # print(f"Wavelength (m): {wavelength:.2e}")
    # print("eps_Au (Drude):", val)
    return np.eye(3, dtype=complex) * val


def Au(d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> Layer:
    return Layer(eps=eps_Au, d=d, theta=theta, phi=phi, psi=psi)


# # Au (visible, n/k tabulados)

_Au_vis_interps = None


def _get_Au_visible_interps():
    global _Au_vis_interps
    if _Au_vis_interps is not None:
        return _Au_vis_interps

    data_dir = pathlib.Path(__file__).parent / "Permittivities/Au_visible"
    n_data = np.loadtxt(data_dir / "n_gold.txt")
    k_data = np.loadtxt(data_dir / "k_gold.txt")

    x_n, y_n = _ensure_increasing(n_data[:, 0], n_data[:, 1])
    x_k, y_k = _ensure_increasing(k_data[:, 0], k_data[:, 1])

    n_interp = interp1d(
        x_n,
        y_n,
        kind="linear",
        bounds_error=False,
        fill_value=(y_n[0], y_n[-1]),
    )
    k_interp = interp1d(
        x_k,
        y_k,
        kind="linear",
        bounds_error=False,
        fill_value=(y_k[0], y_k[-1]),
    )

    _Au_vis_interps = (n_interp, k_interp)
    return _Au_vis_interps


def eps_Au_visible(wavelength: float) -> np.ndarray:
    """
    Permittivity of Au in visible/NIR using tabulated n and k.
    The txt uses wavelength in microns.
    """
    # Convert m -> microns
    w_um = wavelength * 1e6
    n_interp, k_interp = _get_Au_visible_interps()
    n = n_interp(w_um)
    k = k_interp(w_um)
    eps = (n + 1j * k) ** 2
    # print(eps)
    return np.eye(3, dtype=complex) * eps


def Au_visible(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_Au_visible, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# # Sic2
# =============================================================================


def eps_SiC(wavelength: float) -> np.ndarray:
    f = 1.0 / (wavelength * 1e2)
    val = 6.5 * lorentz_osc(f, 972, 796, 3.75)
    return np.eye(3, dtype=complex) * val


def SiC(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_SiC, d=d, theta=theta, phi=phi, psi=psi)


def eps_SiO2_exp(wavelength: float) -> np.ndarray:
    # Nanomaterials 2021, 11(1), 120; https://doi.org/10.3390/nano11010120
    eps = 1.0
    eps_inf = 2
    gamma = [51, 10, 10]
    wTO = [450, 800, 1045]
    wLO = [505, 830, 1240]

    eps = eps_inf
    f = 1.0 / (wavelength * 1e2)
    for i in range(0, len(wTO)):
        eps = eps * lorentz_osc(f, wLO[i], wTO[i], gamma[i])
    return np.eye(3, dtype=complex) * eps


def SiO2(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_SiO2_exp, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# # MoO3
# =============================================================================
def eps_x_MoO3(wavelength: float) -> complex:
    f = 1.0 / (wavelength * 1e2)
    return (
        5.78
        * lorentz_osc(f, 534.3, 506.7, 49.1)
        * lorentz_osc(f, 963.0, 821.4, 6.0)
        * lorentz_osc(f, 999.2, 998.7, 0.35)
    )


def eps_y_MoO3(wavelength: float) -> complex:
    f = 1.0 / (wavelength * 1e2)
    return 6.07 * lorentz_osc(f, 850.1, 544.6, 9.5)


def eps_z_MoO3(wavelength: float) -> complex:
    f = 1.0 / (wavelength * 1e2)
    return 4.47 * lorentz_osc(f, 1006.9, 956.7, 1.5)


def eps_MoO3(wavelength: float) -> np.ndarray:
    return np.diag(
        [eps_x_MoO3(wavelength), eps_y_MoO3(wavelength), eps_z_MoO3(wavelength)]
    )


def MoO3(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_MoO3, d=d, theta=theta, phi=phi, psi=psi)



# =============================================================================
# # hBN
# =============================================================================

def lorentz_osc_hBN(f: float, wLO: float, wTO: float, gamma: float, eps_inf: float) -> complex:
    return eps_inf + eps_inf * (wLO**2 - wTO**2) / (wTO**2 - f**2 - 1j * gamma * f)


def eps_perp_hBN(wavelength: float) -> complex:
    f = 1.0 / (wavelength * 1e2)

    eps_inf = 4.87
    wLO = 1610.0
    wTO = 1372.0
    gamma = 5.0

    return lorentz_osc_hBN(f, wLO, wTO, gamma, eps_inf)


def eps_par_hBN(wavelength: float) -> complex:
    f = 1.0 / (wavelength * 1e2)

    eps_inf = 2.95
    wLO = 819.0
    wTO = 746.0
    gamma = 4.0

    return lorentz_osc_hBN(f, wLO, wTO, gamma, eps_inf)


def eps_x_hBN(wavelength: float) -> complex:
    return eps_perp_hBN(wavelength)


def eps_y_hBN(wavelength: float) -> complex:
    return eps_perp_hBN(wavelength)


def eps_z_hBN(wavelength: float) -> complex:
    return eps_par_hBN(wavelength)


def eps_hBN_tensor(wavelength: float) -> np.ndarray:
    return np.diag(
        [eps_x_hBN(wavelength), eps_y_hBN(wavelength), eps_z_hBN(wavelength)]
    )


def hBN(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_hBN_tensor, d=d, theta=theta, phi=phi, psi=psi)
    

# =============================================================================
# # MoOCl2
# =============================================================================
_MoOCl2_interps = None


def _get_MoOCl2_interps():
    global _MoOCl2_interps
    if _MoOCl2_interps is not None:
        return _MoOCl2_interps

    data_dir = pathlib.Path(__file__).parent / "Permittivities/MoOCl2"

    eps_MoOCl2_xx_real = np.loadtxt(data_dir / "MoOCl2_permittivity_xx_real.txt")
    eps_MoOCl2_yy_real = np.loadtxt(data_dir / "MoOCl2_permittivity_yy_real.txt")
    eps_MoOCl2_zz_real = np.loadtxt(data_dir / "MoOCl2_permittivity_zz_real.txt")
    eps_MoOCl2_xx_imag = np.loadtxt(data_dir / "MoOCl2_permittivity_xx_imag.txt")
    eps_MoOCl2_yy_imag = np.loadtxt(data_dir / "MoOCl2_permittivity_yy_imag.txt")
    eps_MoOCl2_zz_imag = np.loadtxt(data_dir / "MoOCl2_permittivity_zz_imag.txt")

    def _interp(x, y):
        return interp1d(
            x,
            y,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

    _MoOCl2_interps = (
        _interp(eps_MoOCl2_xx_real[:, 0], eps_MoOCl2_xx_real[:, 1]),
        _interp(eps_MoOCl2_xx_imag[:, 0], eps_MoOCl2_xx_imag[:, 1]),
        _interp(eps_MoOCl2_yy_real[:, 0], eps_MoOCl2_yy_real[:, 1]),
        _interp(eps_MoOCl2_yy_imag[:, 0], eps_MoOCl2_yy_imag[:, 1]),
        _interp(eps_MoOCl2_zz_real[:, 0], eps_MoOCl2_zz_real[:, 1]),
        _interp(eps_MoOCl2_zz_imag[:, 0], eps_MoOCl2_zz_imag[:, 1]),
    )
    return _MoOCl2_interps


def eps_MoOCl2(wavelength: float) -> np.ndarray:
    """
    Permittivity of MoOCl2 for a given axis, using tabulated data files.
    """
    # Los datos de permitividad están tabulados en nm; el código usa m.
    w_nm = wavelength * 1e9

    (
        xx_re,
        xx_im,
        yy_re,
        yy_im,
        zz_re,
        zz_im,
    ) = _get_MoOCl2_interps()

    eps_x_MoOCl2 = xx_re(w_nm) + 1j * xx_im(w_nm)
    eps_y_MoOCl2 = yy_re(w_nm) + 1j * yy_im(w_nm)
    eps_z_MoOCl2 = zz_re(w_nm) + 1j * zz_im(w_nm)
    return np.diag([eps_x_MoOCl2, eps_y_MoOCl2, eps_z_MoOCl2])


def MoOCl2(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_MoOCl2, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# # Si
# =============================================================================
_Si_interps = None
_BaF2_interps = None


def _ensure_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x[0] > x[-1]:
        return x[::-1], y[::-1]
    return x, y


def _get_Si_interps():
    global _Si_interps
    if _Si_interps is not None:
        return _Si_interps

    data_dir = pathlib.Path(__file__).parent / "Permittivities/Si"
    eps_Si_real = np.loadtxt(data_dir / "real_Si.txt")
    eps_Si_imag = np.loadtxt(data_dir / "imag_Si.txt")

    x_re, y_re = _ensure_increasing(eps_Si_real[:, 0], eps_Si_real[:, 1])
    x_im, y_im = _ensure_increasing(eps_Si_imag[:, 0], eps_Si_imag[:, 1])

    re_interp = interp1d(
        x_re,
        y_re,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    im_interp = interp1d(
        x_im,
        y_im,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )

    _Si_interps = (re_interp, im_interp)
    return _Si_interps


def eps_Si(wavelength: float) -> np.ndarray:
    """
    Permittivity of Si for a given axis, using tabulated data files.
    """
    # Los datos de permitividad están tabulados en cm-1
    w_cm = 1.0 / (wavelength * 100.0)

    re_interp, im_interp = _get_Si_interps()
    eps_Si = re_interp(w_cm) + 1j * im_interp(w_cm)

    return np.eye(3, dtype=complex) * eps_Si


def Si(d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> Layer:
    return Layer(eps=eps_Si, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# BaF2
# =============================================================================
def _get_BaF2_interps():
    global _BaF2_interps
    if _BaF2_interps is not None:
        return _BaF2_interps

    data_dir = pathlib.Path(__file__).parent / "Permittivities/BaF2"
    eps_BaF2_R_I = np.loadtxt(data_dir / "epsbaf2.txt", delimiter=",", dtype=np.float64)

    x_re, y_re = _ensure_increasing(eps_BaF2_R_I[:, 0], eps_BaF2_R_I[:, 1])
    x_im, y_im = _ensure_increasing(eps_BaF2_R_I[:, 0], eps_BaF2_R_I[:, 2])

    # Evitar oscilaciones no fÃ­sicas del cÃºbico: usar lineal y clamp en bordes.
    re_interp = interp1d(
        x_re,
        y_re,
        kind="linear",
        bounds_error=False,
        fill_value=(y_re[0], y_re[-1]),
    )
    im_interp = interp1d(
        x_im,
        y_im,
        kind="linear",
        bounds_error=False,
        fill_value=(y_im[0], y_im[-1]),
    )

    _BaF2_interps = (re_interp, im_interp)
    return _BaF2_interps


def eps_BaF2(wavelength: float) -> np.ndarray:
    """
    Permittivity of BaF2 for a given axis, using tabulated data files.
    """
    # Los datos de permitividad están tabulados en cm-1
    w_cm = 1.0 / (wavelength * 100.0)

    re_interp, im_interp = _get_BaF2_interps()
    eps_BaF2 = re_interp(w_cm) + 1j * im_interp(w_cm)
    return np.eye(3, dtype=complex) * eps_BaF2


def BaF2(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_BaF2, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
# V2O5
# =============================================================================


# Lorentz oscillator model
def epsilon_1phonon(w, wT, wL, gT, gL):
    eps = (w**2 - wL**2 + 1j * gL * w) / (w**2 - wT**2 + 1j * gT * w)
    return eps


def eps_A_V2O5_4parOsc(w):
    # data from from P. Clauws  J. Vennik, \"Lattice Vibrations of $V_2 O_5$. Determination of TO and LO Frequencies from Infrared Reflection and Transmission\" in cm^-1
    f = 1.0 / (w * 1e2)
    epsAinf = 5.5
    eps = 1.0
    GTOA = [3.6, 13, 15, 5, 30, 10]
    GLOA = [4.2, 8, 12.2, 30, 50, 15]
    wTOA = [72.4, 261, 303, 411, 767.5, 980.5]
    wLOA = [76.2, 265.5, 390.5, 586, 959, 982]
    for i in range(0, len(wTOA)):
        eps = eps * epsilon_1phonon(f, wTOA[i], wLOA[i], GTOA[i], GLOA[i])
    eps = epsAinf * eps
    return eps


def eps_B_V2O5_4parOsc(w):
    # data from from P. Clauws  J. Vennik, \"Lattice Vibrations of $V_2 O_5$. Determination of TO and LO Frequencies from Infrared Reflection and Transmission\" in cm^-1
    f = 1.0 / (w * 1e2)
    epsBinf = 4.77
    eps = 1.0
    GTOB = [18, 2.5]
    GLOB = [15, 2.5]
    wTOB = [473, 975.5]
    # wTOB = [473,  960];
    wLOB = [490, 1038]
    for i in range(0, len(wTOB)):
        eps = eps * epsilon_1phonon(f, wTOB[i], wLOB[i], GTOB[i], GLOB[i])
    eps = epsBinf * eps
    return eps


def eps_C_V2O5_4parOsc(w):
    # data from from P. Clauws  J. Vennik, \"Lattice Vibrations of $V_2 O_5$. Determination of TO and LO Frequencies from Infrared Reflection and Transmission\" in cm^-1
    f = 1.0 / (w * 1e2)
    epsCinf = 4.49
    eps = 1.0
    GTOC = [10.5, 7.8, 21.0]
    GLOC = [7.5, 10.2, 18.0]
    wTOC = [212.0, 284.0, 506.5]
    wLOC = [225.0, 315.5, 842.5]
    for i in range(0, len(wTOC)):
        eps = eps * epsilon_1phonon(f, wTOC[i], wLOC[i], GTOC[i], GLOC[i])
    eps = epsCinf * eps
    return eps


def eps_V2O5(wavelength: float) -> np.ndarray:
    return np.diag(
        [
            eps_A_V2O5_4parOsc(wavelength),
            eps_C_V2O5_4parOsc(wavelength),
            eps_B_V2O5_4parOsc(wavelength),
        ]
    )


def V2O5(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_V2O5, d=d, theta=theta, phi=phi, psi=psi)


# =============================================================================
#  MgTeMoO6
# =============================================================================
def epsilon_phonon_Q(w, wi, fi, gi):
    eps = np.zeros_like(w, dtype=complex)
    for i in range(len(wi)):
        aux = fi[i] ** 2 / (wi[i] ** 2 - w**2 - 1j * gi[i] * w)
        eps += aux
    return eps


def eps_XYZ_MgTeMoO6(w, axis):
    """
    Data: https://doi.org/10.1038/s41565-024-01628-y
    """
    f = 1.0 / (w * 1e2)
    eps_inf = [3.0, 3.1, 0.7]

    wix = np.array([698.47, 749.47, 895.50])
    fix = np.array([570.04, 90.30, 420.63])
    gix = np.array([6.09, 36.71, 1.81])

    wiy = np.array([684.54, 749.08, 902.98])
    fiy = np.array([710.71, 140.53, 316.50])
    giy = np.array([2.81, 31.80, 1.99])

    wiz = np.array([950.0])
    fiz = np.array([103.37])
    giz = np.array([7.0])

    axis = axis.upper()

    if axis == "X":
        eps = eps_inf[0] + epsilon_phonon_Q(f, wix, fix, gix)
    elif axis == "Y":
        eps = eps_inf[1] + epsilon_phonon_Q(f, wiy, fiy, giy)
    elif axis == "Z":
        eps = eps_inf[2] + epsilon_phonon_Q(f, wiz, fiz, giz)
    else:
        raise ValueError("Eje no válido. Usa 'X', 'Y' o 'Z'.")

    return eps


def eps_MgTeMoO6(wavelength: float) -> np.ndarray:
    return np.diag(
        [
            eps_XYZ_MgTeMoO6(wavelength, "X"),
            eps_XYZ_MgTeMoO6(wavelength, "Y"),
            eps_XYZ_MgTeMoO6(wavelength, "Z"),
        ]
    )


def MgTeMoO6(
    d: float = 0.0, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0
) -> Layer:
    return Layer(eps=eps_MgTeMoO6, d=d, theta=theta, phi=phi, psi=psi)
