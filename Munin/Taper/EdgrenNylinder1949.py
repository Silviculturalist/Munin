from Munin.Volume.Timber import Timber
from Munin.Volume.Naslund1947 import NaslundFormFactor
from typing import Optional
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

class EdgrenNylinder1949:
    @staticmethod
    def get_constants(species: str, north: bool, form_factor: float):
        const_spruce_north = np.array([
            [0.5, np.nan, 1.671, 16.104, np.nan, 103.06],
            [0.55, 0.62, 1.422, 14.883, 286.36, 140.91],
            [0.6, 1.594, 1.976, 13.784, 151.04, 127.89],
            [0.65, 3.240, 2.906, 12.906, 102.7, 110.68],
            [0.7, 6.320, 3.759, 12.099, 76.543, 105.15],
            [0.75, 13.056, 4.026, 11.321, 59.096, 112.59],
            [0.8, 32.012, 3.595, 10.540, 45.754, 134.76],
        ], dtype=float)

        const_spruce_south = np.array([
            [0.5, np.nan, 0.892, 15.765, np.nan, 176.40],
            [0.55, 0.620, 0.923, 14.818, 287.44, 202.65],
            [0.6, 1.594, 1.093, 14.032, 148.97, 202.51],
            [0.65, 3.240, 2.164, 13.479, 99.532, 132.69],
            [0.7, 6.320, 3.324, 13.040, 72.736, 108.43],
            [0.75, 13.059, 4.463, 12.775, 54.618, 97.49],
            [0.8, 33.208, 5.586, 12.578, 40.509, 91.77],
        ], dtype=float)

        const_pine_north = np.array([
            [0.5, np.nan, 1.513, 14.233, np.nan, 123.91],
            [0.55, 0.620, 1.228, 13.321, 311.68, 172.85],
            [0.6, 1.594, 1.506, 12.657, 160.29, 167.68],
            [0.65, 3.240, 2.493, 12.177, 106.67, 128.16],
            [0.7, 6.320, 4.488, 11.880, 77.416, 94.947],
            [0.75, 13.056, 6.602, 11.759, 57.767, 81.725],
            [0.8, 32.307, 7.594, 11.753, 42.808, 80.776],
        ], dtype=float)

        const_pine_south = np.array([
            [0.5, np.nan, 0.8409, 15.970, np.nan, 183.44],
            [0.55, 0.620, 0.3694, 14.948, 285.28, 458.59],
            [0.6, 1.594, 0.4251, 14.214, 147.44, 463.07],
            [0.65, 3.240, 1.529, 13.646, 98.601, 171.70],
            [0.7, 6.320, 3.974, 13.240, 71.915, 95.286],
            [0.75, 13.070, 5.510, 12.951, 54.050, 84.904],
            [0.8, 33.502, 6.445, 12.755, 39.982, 83.659],
        ], dtype=float)

        # Species and region logic
        if north and species == 'picea abies':
            constants = const_spruce_north
        elif not north and species == 'picea abies':
            constants = const_spruce_south
        elif north: #and species == 'pinus sylvestris': #Other species
            constants = const_pine_north
        elif not north:# and species == 'pinus sylvestris': #Other species
            constants =  const_pine_south
        else:
            #Should never occur.
            raise ValueError(f"Unsupported species or region: {species}, North={north}")
    
            # Step 1: Round form factor to the nearest 0.5
        rounded_form_factor = round(form_factor * 2) / 2

        # Step 2: Enforce limits between 0.5 and 0.8
        if rounded_form_factor < 0.5:
            rounded_form_factor = 0.5
        elif rounded_form_factor > 0.8:
            rounded_form_factor = 0.8

        # Step 3: Select the corresponding row
        row = constants[np.isclose(constants[:, 0], rounded_form_factor)]

        if row.size == 0:
            raise ValueError(f"No matching row for form factor: {rounded_form_factor}")

        return row

    @staticmethod
    def get_inflexion_point(species: str, north: bool, form_quotient: float) -> float:
        if north:
            if species == "picea abies":
                return 0.08631 / (1 - form_quotient) ** 0.5
            else: #species == "pinus sylvestris":
                return 0.05270 / (1 - form_quotient) ** 0.9
        else:
            if species == "picea abies":
                return 0.06731 / (1 - form_quotient) ** 0.8
            else: #species == "pinus sylvestris":
                return 0.06873 / (1 - form_quotient) ** 0.8

        #raise ValueError(f"Unsupported species: {species}")


class Pettersson1949:
    @staticmethod
    def form_quotient(
        species: str, north: bool, height: float, dbh_ub: float, form_factor_ub: float
    ) -> float:
        if north:
            if species == "picea abies":
                return 0.239 + 0.01046 * height - 0.004407 * dbh_ub + 0.6532 * form_factor_ub
            else:# species == "pinus sylvestris":
                return 0.293 + 0.00669 * height - 0.001384 * dbh_ub + 0.6348 * form_factor_ub
        else:
            if species == "picea abies":
                return 0.209 + 0.00859 * height - 0.003157 * dbh_ub + 0.7385 * form_factor_ub
            else:
                return 0.372 + 0.008742 * height - 0.003263 * dbh_ub + 0.4929 * form_factor_ub


class TimberEdgrenDiameter:
    @staticmethod
    def get_base_diameter(timber: Timber) -> float:
        """
        Calculate the base diameter (DBAS) of the tree using the diameter at breast height (DBH).
        
        :param timber: Timber object containing tree data.
        :return: The base diameter (DBAS) in cm.
        """
        # Calculate relative diameter at breast height (1.3m)
        dbh_relative = TimberEdgrenDiameter.get_relative_diameter(
            rel_height=1.3 / timber.height_m, timber=timber
        )
    
        # Validate relative diameter
        if dbh_relative is None or dbh_relative <= 0:
            print(f"Warning: Invalid relative diameter at breast height: {dbh_relative}")
            return None
    
        # Compute base diameter (DBAS)
        base_diameter = 100 * (timber.diameter_cm / dbh_relative)
        if base_diameter <= 0:
            print(f"Warning: Invalid base diameter: {base_diameter}")
            return None
    
        return base_diameter

    
    @staticmethod
    def get_relative_diameter(rel_height: float, timber: Timber) -> float:
        """
        Calculate the relative diameter at a given relative height using the taper function.

        :param rel_height: The relative height (height / total height).
        :param timber: Timber object containing tree data.
        :return: The relative diameter (unitless).
        """
        # Retrieve constants and inflexion point
        form_factor = NaslundFormFactor.calculate(
            species=timber.species,
            height_m=timber.height_m,
            diameter_cm=timber.diameter_cm,
            double_bark_mm=timber.double_bark_mm,
            crown_base_height_m=timber.crown_base_height_m,
            over_bark=timber.over_bark,
            region=timber.region,
        )

        form_quotient = Pettersson1949.form_quotient(
            species=timber.species,
            north=(timber.region == "northern"),
            height=timber.height_m,
            dbh_ub=timber.diameter_cm,
            form_factor_ub=form_factor,
        )

        inflexion_point = EdgrenNylinder1942.get_inflexion_point(
            species=timber.species, north=(timber.region == "northern"), form_quotient=form_quotient
        )

        constants = EdgrenNylinder1942.get_constants(
            species=timber.species, north=(timber.region == "northern"), form_factor=form_quotient
        )

        # Constants used in taper equations
        const_F, const_beta, const_Gamma, const_q, const_Q, const_R = constants[0]

        # Compute relative diameter based on height regions
        if rel_height <= inflexion_point:
            return 100 - const_q * np.log10(1 + 10000 * rel_height)
        elif inflexion_point < rel_height <= 0.6:
            return const_Q * np.log10(1 + (1 - rel_height) * const_beta)
        elif 0.6 < rel_height < 1:
            return const_R * np.log10(1 + (1 - rel_height) * const_Gamma)
        else:
            print(f"Warning: Unexpected value for relative height: {rel_height}")
            return None

    @staticmethod
    def get_diameter_at_height(timber: Timber, height: float) -> Optional[float]:
        """
        Calculate the diameter at a specific height using the taper function.

        :param timber: Timber object containing tree data.
        :param height: Height (in meters) for which to calculate the diameter.
        :return: Diameter (in cm) at the specified height.
        """
        timber.validate()

        # Validate height range
        if height < 0 or height > timber.height_m:
            print(f"Warning: Invalid requested height: {height}")
            return None

        # Get base diameter
        base_diameter = TimberEdgrenDiameter.get_base_diameter(timber)
        if base_diameter is None:
            return None

        # Calculate relative height
        rel_height = height / timber.height_m

        # Get relative diameter at the specified height
        relative_diameter = TimberEdgrenDiameter.get_relative_diameter(rel_height, timber)
        if relative_diameter is None or relative_diameter <= 0:
            print(f"Warning: Invalid relative diameter at requested height: {relative_diameter}")
            return None

        # Compute actual diameter at height
        return (base_diameter * relative_diameter) / 100

    

    @staticmethod
    def get_height(timber: Timber, minDiameter: float) -> float:
        """
        Find the height corresponding to the specified diameter.

        :param timber: Timber object containing the tree's data.
        :param minDiameter: Target diameter (cm).
        :return: Height (m) corresponding to the target diameter.
        """
        # Validate inputs
        if minDiameter <= 0 or minDiameter > TimberEdgrenDiameter.get_base_diameter(timber):
            print(f"Invalid minDiameter: {minDiameter}. Must be between 0 and {TimberEdgrenDiameter.get_base_diameter(timber)}.")
            return None

        # Objective function: Minimize |calculated_diameter - minDiameter|
        def objective(height):
            diameter_at_height = TimberEdgrenDiameter.get_diameter_at_height(timber, height)
            return abs(diameter_at_height - minDiameter)

        # Use Newton's method or a bounded scalar minimizer
        result = minimize_scalar(
            objective,
            bounds=(0, timber.height_m),
            method='bounded'
        )

        # Check if optimization was successful
        if result.success:
            return result.x  # The height where minDiameter is matched
        else:
            print(f"Optimization failed for minDiameter: {minDiameter}")
            return None    




class TimberVolumeIntegrator:
    """
    A utility class for calculating the integrated volume of a timber log between two heights using a taper function.
    """

    @staticmethod
    def cylinder_volume_integrand(height, timber, taper_class):
        """
        Calculate the cross-sectional area (πr^2) at a given height.

        :param height: Height at which to compute the diameter and area.
        :param timber: The Timber object (e.g., timber_birch).
        :param taper_class: The Taper class (e.g., TimberEdgrenDiameter).
        :return: Cross-sectional area at the given height.
        """
        diameter = taper_class.get_diameter_at_height(timber, requested_height=height)
        radius = diameter / 200 #cm to m
        return np.pi * (radius ** 2)

    @staticmethod
    def integrate_volume(height1, height2, timber, taper_class):
        """
        Integrate the volume of the cylinder between two heights.

        :param height1: The lower bound of integration.
        :param height2: The upper bound of integration.
        :param timber: The Timber object (e.g., timber_birch).
        :param taper_class: The Taper class (e.g., TimberEdgrenDiameter).
        :return: The integrated volume (m3)
        """
        volume, _ = quad(
            TimberVolumeIntegrator.cylinder_volume_integrand, height1, height2,
            args=(timber, taper_class)
        )
        return volume
    