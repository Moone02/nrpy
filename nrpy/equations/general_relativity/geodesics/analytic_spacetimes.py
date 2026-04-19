"""
Construct symbolic expressions for analytic spacetime metrics.

This module provides a class-based structure for generating the symbolic
metric tensor for the Kerr-Schild analytic solution to Einstein's equations.
It is designed to integrate with nrpy's CodeParameter system.


Author: Dalton J. Moone
"""

# Step 0.a: Import standard Python modules
import logging
from typing import Dict, List, Tuple

# Step 0.b: Import third-party modules
import sympy as sp

# Step 0.c: Import NRPy core modules
import nrpy.indexedexp as ixp
import nrpy.params as par
import nrpy.validate_expressions.validate_expressions as ve
from nrpy.equations.general_relativity.g4munu_conversions import ADM_to_g4DD
from nrpy.equations.general_relativity.InitialData_Cartesian import (
    InitialData_Cartesian,
)


class AnalyticSpacetimes:
    """
    Generate and store symbolic expressions for analytic spacetime metrics.

    This class is instantiated with a specific spacetime name. It then calls
    the appropriate recipe to generate the 4-metric g_munu and the underlying
    coordinate system symbols, storing them as instance attributes.
    """

    # mypy --strict requires class attributes to be declared.
    spacetime_name: str
    g4DD: List[List[sp.Expr]]
    xx: List[sp.Symbol]

    def __init__(self, spacetime_name: str) -> None:
        """
        Initialize and generate the symbolic metric for a given spacetime.

        :param spacetime_name: The name of the spacetime to generate
                               (e.g., "KerrSchild_Cartesian", "Schwarzschild_Cartesian_Isotropic").
        :raises ValueError: If the requested spacetime is not supported.
        """
        self.spacetime_name = spacetime_name

        if self.spacetime_name == "KerrSchild_Cartesian":
            self.g4DD, self.xx = self._define_kerr_metric_Cartesian_Kerr_Schild()
        elif self.spacetime_name == "BrillLindquist_Cartesian":
            self.g4DD, self.xx = self._define_from_InitialData_Cartesian(
                "BrillLindquist"
            )
        else:
            raise ValueError(f"Spacetime '{self.spacetime_name}' is not supported.")

    @staticmethod
    def _define_kerr_metric_Cartesian_Kerr_Schild() -> (
        Tuple[List[List[sp.Expr]], List[sp.Symbol]]
    ):
        """
        Define the Kerr metric in Cartesian Kerr-Schild coordinates.

        The metric is constructed as g_munu = eta_munu + 2H * l_mu * l_nu.
        This form is regular everywhere, including the horizon.

        Reference:
        Wikipedia: Kerr-Schild coordinates
        Permanent Link: https://en.wikipedia.org/w/index.php?title=Kerr_metric&oldid=1318460406
        (See section on Kerr–Schild coordinates)

        :return: A tuple (g4DD, xx), where g4DD is the symbolic 4x4 metric tensor
                 and xx is the list of symbolic coordinate variables (t, x, y, z).
        """
        # Step 1.a: Define generic symbolic coordinates.
        t, x, y, z = sp.symbols("t x y z", real=True)
        xx = [t, x, y, z]

        # Step 1.b: Register physical parameters (G=c=1; M_scale = ADM mass)
        M_scale = par.register_CodeParameter(
            "REAL", __name__, "M_scale", 1.0, commondata=True
        )
        a_spin = par.register_CodeParameter(
            "REAL", __name__, "a_spin", 0.0, commondata=True
        )

        # Step 2: Define intermediate geometric quantities.
        # The Kerr-Schild radius 'r' is not the Euclidean radius. It is solved
        # for implicitly from the Cartesian coordinates (x, y, z) and spin a.
        # rho2 is the squared Euclidean distance from the origin.
        rho2 = x**2 + y**2 + z**2
        a_spin2 = a_spin**2

        # This is the solution to the quartic equation for r:
        # r^4 - (rho^2 - a^2)r^2 - a^2 z^2 = 0
        r2 = sp.Rational(1, 2) * (
            rho2 - a_spin2 + sp.sqrt((rho2 - a_spin2) ** 2 + 4 * a_spin2 * z**2)
        )
        r = sp.sqrt(r2)

        # Step 3: Define the Kerr-Schild null vector l_mu.
        l_down = ixp.zerorank1(dimension=4)
        l_down[0] = sp.sympify(1)
        l_down[1] = (r * x + a_spin * y) / (r2 + a_spin2)
        l_down[2] = (r * y - a_spin * x) / (r2 + a_spin2)
        l_down[3] = z / r

        # Step 4: Define the scalar function H.
        H = (M_scale * r**3) / (r**4 + a_spin2 * z**2)

        # Step 5: Construct the Kerr-Schild metric g_munu = eta_munu + 2H * l_mu * l_nu.
        eta4DD = ixp.zerorank2(dimension=4)
        eta4DD[0][0] = sp.sympify(-1)
        eta4DD[1][1] = eta4DD[2][2] = eta4DD[3][3] = sp.sympify(1)
        g4DD = ixp.zerorank2(dimension=4)
        for mu in range(4):
            for nu in range(4):
                g4DD[mu][nu] = eta4DD[mu][nu] + 2 * H * l_down[mu] * l_down[nu]

        return g4DD, xx

    @staticmethod
    def _define_from_InitialData_Cartesian(
        IDtype: str,
    ) -> Tuple[List[List[sp.Expr]], List[sp.Symbol]]:
        """
        Define 4-metric from InitialData_Cartesian.

        Physical Note: Injecting time-dependence into Brill-Lindquist t=0 initial data
        violates time symmetry, ignores frame dragging, misses gravitational radiation,
        and violates constraint equations; this creates an ad-hoc, geometric approximation
        (a "flipbook" of static universes) for visualization purposes rather than a
        rigorous binary black hole merger.

        :param IDtype: The type of initial data (e.g., "BrillLindquist").
        :return: A tuple (g4DD, xx).
        """
        # Step 1: Initialize generic coordinates and standard initial data.
        t = sp.symbols("t", real=True)
        ID = InitialData_Cartesian(IDtype)
        g4DD = ADM_to_g4DD(ID.gammaDD, ID.betaU, ID.alpha)
        xx = [t, ID.x, ID.y, ID.z]

        # Step 2: Inject dynamic time dependence for Brill-Lindquist trajectories.
        if IDtype == "BrillLindquist":
            # Step 2.a: Target the static CodeParameter symbols by exact name.
            # These now represent the *initial* (t=0) positions and masses.
            BH1_posn_z0 = sp.Symbol("BH1_posn_z", real=True)
            BH2_posn_z0 = sp.Symbol("BH2_posn_z", real=True)
            BH1_mass = sp.Symbol("BH1_mass", real=True)
            BH2_mass = sp.Symbol("BH2_mass", real=True)
            M_tot = BH1_mass + BH2_mass

            # Step 2.b: Construct the [2/2] Padé approximant for the separation distance.
            # We assume BH1 is initially above BH2 (BH1_posn_z0 > BH2_posn_z0).
            # This uses a [2/2] Padé approximant (see: https://en.wikipedia.org/wiki/Pad%C3%A9_approximant).
            r0 = BH1_posn_z0 - BH2_posn_z0
            Z_cm = (BH1_mass * BH1_posn_z0 + BH2_mass * BH2_posn_z0) / M_tot

            A = r0
            C = -M_tot / (6 * r0**3)
            B = -(2 * M_tot) / (3 * r0**2)

            # r_pade(t) extends the 4th-order Taylor series for Newtonian infall (assuming G=c=1).
            r_pade = (A + B * t**2) / (sp.sympify(1) + C * t**2)

            # Step 2.c: Define the new dynamic z-axis positions over time.
            # Assuming G=c=1, this approximates the infalling black holes using a purely Newtonian force,
            # mapped via the [2/2] Padé approximant derived from the Taylor series of separation distance.
            BH1_posn_z_t = Z_cm + (BH2_mass / M_tot) * r_pade
            BH2_posn_z_t = Z_cm - (BH1_mass / M_tot) * r_pade

            # Step 2.d: Map static parameters to the dynamic expressions.
            dynamic_subs = {
                BH1_posn_z0: BH1_posn_z_t,
                BH2_posn_z0: BH2_posn_z_t,
            }

            # Step 2.e: Apply substitution and explicitly enforce symmetry.
            # Note: .subs() is restricted by coding_style.md for pattern matching,
            # but is allowed here as it performs an exact evaluation/mapping of
            # static coordinate parameters to time-dependent functions.
            for mu in range(4):
                for nu in range(mu, 4):
                    g4DD[mu][nu] = g4DD[mu][nu].subs(dynamic_subs)
                    g4DD[nu][mu] = g4DD[mu][nu]

        return g4DD, xx


class AnalyticSpacetimes_dict(Dict[str, "AnalyticSpacetimes"]):
    """A caching dictionary for AnalyticSpacetimes instances."""

    def __getitem__(self, key: str) -> "AnalyticSpacetimes":
        """
        Get or create an AnalyticSpacetimes instance for a given configuration.

        :param key: A string key identifying the spacetime (e.g., "KerrSchild").
        :return: An AnalyticSpacetimes instance for the specified configuration.
        """
        if key not in self:
            # If the spacetime is not cached, generate it and add it to the cache.
            logging.getLogger(__name__).info(
                "Setting up analytic spacetime: '%s'...", key
            )
            self[key] = AnalyticSpacetimes(spacetime_name=key)
        return super().__getitem__(key)


Analytic_Spacetimes = AnalyticSpacetimes_dict()


if __name__ == "__main__":
    import doctest
    import os
    import sys

    # Configure logging to output to the console.
    logging.basicConfig(level=logging.INFO)

    results = doctest.testmod()
    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    # Use a distinct loop variable name to avoid pylint redefined-outer-name warnings.
    for spacetime_name_str in ["KerrSchild_Cartesian", "BrillLindquist_Cartesian"]:
        spacetimes = Analytic_Spacetimes[spacetime_name_str]
        results_dict = ve.process_dictionary_of_expressions(
            spacetimes.__dict__, fixed_mpfs_for_free_symbols=True
        )
        # Break long line to satisfy pylint line-too-long.
        ve.compare_or_generate_trusted_results(
            os.path.abspath(__file__),
            os.getcwd(),
            f"{os.path.splitext(os.path.basename(__file__))[0]}_{spacetime_name_str}",
            results_dict,
        )
