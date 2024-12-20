import numpy as np
class PriceList:
    def __init__(
        self,
        species: str,
        source: str = "MellanSkog",
        diameter_classes: Optional[list] = None,
        quality_price_list: Optional[list] = None,
        quality_percentages: Optional[dict] = None,
        downgrading_percentages: Optional[dict] = None,
        pulpwood_price: float = 265,
        bucker_config: Optional[dict] = None,
    ):
        """
        Initializes the PriceList class with pricing, quality, and bucker configuration details.

        :param species: Tree species (e.g., "picea abies").
        :param source: Source of the price list (default: "MellanSkog").
        :param diameter_classes: List of diameter classes for logs.
        :param quality_price_list: Nested list of prices for each diameter and quality.
        :param quality_percentages: Quality percentages for different log parts.
        :param downgrading_percentages: Downgrading percentages for different log parts.
        :param pulpwood_price: Price for pulpwood (default: 265).
        :param bucker_config: Dictionary containing log constraints and bucker settings.
        """
        if not species:
            raise ValueError("Species must be named.")

        # Default values
        if diameter_classes is None:
            diameter_classes = [13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
        if quality_price_list is None:
            quality_price_list = [
                [300, 425, 450, 485, 510, 535, 555, 575, 590, 605, 620, 625, 525],
                [300, 375, 375, 400, 400, 400, 400, 400, 425, 425, 425, 425, 350],
            ]
        if quality_percentages is None:
            quality_percentages = {
                "ButtLog": [86, 14],
                "MiddleLog": [10, 0, 2],
                "TopLog": [10, 0, 2],
            }
        if downgrading_percentages is None:
            downgrading_percentages = {
                "ButtLog": [10, 0, 2],
                "MiddleLog": [10, 0, 2],
                "TopLog": [10, 0, 2],
            }
        if bucker_config is None:
            bucker_config = {
                "max_tree_height": 45,  # Maximum tree height in meters
                "min_length_sawlog": 3.4,  # Minimum sawlog length in meters
                "max_length_sawlog": 5.5,  # Maximum sawlog length in meters
                "min_diameter_sawlog": 5,  # Minimum sawlog diameter in cm
                "max_diameter_sawlog": 60,  # Maximum sawlog diameter in cm
                "min_length_pulpwood": 2.7,  # Minimum pulpwood length in meters
                "max_length_pulpwood": 5.5,  # Maximum pulpwood length in meters
                "top_diameter": 5,  # Minimum top diameter for logs
                "log_cull_price": 0,  # Price for culled logs
                "pulpwood_cull_proportion": 0.05,  # Cull proportion for pulpwood
                "fuelwood_proportion": 0,  # Proportion of logs as fuelwood
                "harvest_residue_price": 380,  # Harvest residue price
                "fuelwood_log_price": 200,  # Fuelwood price
                "stump_price": 280,  # Stump price
                "high_stump_height": 4,  # High stump height in meters
            }

        self.species = species
        self.source = source
        self.diameter_classes = diameter_classes
        self.quality_price_list = quality_price_list
        self.quality_percentages = quality_percentages
        self.downgrading_percentages = downgrading_percentages
        self.pulpwood_price = pulpwood_price
        self.bucker_config = bucker_config

    def get_bucker_config(self):
        """
        Retrieve the bucker configuration for this price list.
        :return: Dictionary containing bucker settings.
        """
        return self.bucker_config
