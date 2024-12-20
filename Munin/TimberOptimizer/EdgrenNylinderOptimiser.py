import numpy as np
from Munin.PriceList import PriceList
from Munin.Taper import EdgrenNylinder1949, TimberEdgrenDiameter, TimberVolumeIntegrator
from Munin.Volume import Timber

class TimberOptimizer:
    def __init__(self, price_list: PriceList):
        """
        Initializes the TimberOptimizer with the given price list.
        :param price_list: Instance of PriceList containing species-specific pricing and bucker configuration.
        """
        self.price_list = price_list
        self.bucker_config = price_list.get_bucker_config()

    def optimize(self, timber):
        """
        Optimize log cutting for a given timber tree using the associated price list and bucker configuration.
        
        :param timber: Timber object containing tree data (species, height, DBH, etc.).
        :return: List of log sections with optimal value.
        """
        # Extract bucker constraints
        min_length_sawlog = self.bucker_config["min_length_sawlog"]
        max_length_sawlog = self.bucker_config["max_length_sawlog"]
        min_diameter_sawlog = self.bucker_config["min_diameter_sawlog"]
        max_diameter_sawlog = self.bucker_config["max_diameter_sawlog"]
        downgrading_percentages = self.price_list.downgrading_percentages
        quality_price_list = self.price_list.quality_price_list

        # Dynamic programming setup
        height_step = self.bucker_config["height_step_length"]
        heights = np.arange(0, timber.height_m + height_step, height_step)
        n_segments = len(heights)
        values = np.zeros(n_segments)
        cut_positions = np.zeros(n_segments, dtype=int)
        qualities = [None] * n_segments

        # Optimization loop
        for start in range(n_segments):
            for length in range(1, n_segments - start):
                end = start + length
                if end >= n_segments:
                    break

                # Calculate log properties
                h1, h2 = heights[start], heights[end]
                log_length = h2 - h1
                log_diameter = EdgrenNylinder1949.get_diameter_at_height(timber, h2)
                if log_diameter is None or log_length < min_length_sawlog or log_length > max_length_sawlog:
                    continue

                if log_diameter < min_diameter_sawlog or log_diameter > max_diameter_sawlog:
                    continue

                # Calculate log volume
                log_volume = TimberVolumeIntegrator.integrate_volume(h1, h2, timber, TimberEdgrenDiameter)
                if log_volume <= 0:
                    continue

                # Determine log quality and adjust value
                quality, adjusted_value = self._get_log_quality_and_value(
                    log_diameter, log_volume, quality_price_list, downgrading_percentages
                )

                # Update dynamic programming arrays
                if values[start] + adjusted_value > values[end]:
                    values[end] = values[start] + adjusted_value
                    cut_positions[end] = start
                    qualities[end] = quality

        # Backtrack to find optimal cuts
        sections = self._backtrack_cuts(heights, cut_positions, qualities, values)
        return sections

    def _get_log_quality_and_value(self, diameter, volume, quality_price_list, downgrading_percentages):
        """
        Determine the quality and adjusted value of a log based on its diameter and volume.
        :param diameter: Diameter of the log (cm).
        :param volume: Volume of the log (m³).
        :param quality_price_list: Nested list of prices for each diameter and quality.
        :param downgrading_percentages: Dictionary of downgrading percentages for different log parts.
        :return: Tuple (quality, adjusted_value).
        """
        quality = None
        max_value = 0

        # Iterate through qualities to find the best value
        for q_idx, price_list in enumerate(quality_price_list):
            if diameter >= len(price_list):  # Skip if diameter exceeds price list range
                continue

            base_value = price_list[int(diameter)] * volume
            downgrading_factor = 1 - sum(downgrading_percentages["ButtLog"]) / 100  # Adjust for downgrading
            adjusted_value = base_value * downgrading_factor

            if adjusted_value > max_value:
                max_value = adjusted_value
                quality = q_idx

        return quality, max_value

    def _backtrack_cuts(self, heights, cut_positions, qualities, values):
        """
        Backtrack through the dynamic programming arrays to determine optimal cuts.
        :param heights: Array of segment heights.
        :param cut_positions: Array of cut positions for each segment.
        :param qualities: Array of log qualities for each segment.
        :param values: Array of log values for each segment.
        :return: List of log sections with optimal cuts.
        """
        sections = []
        end = np.argmax(values)
        while end > 0:
            start = cut_positions[end]
            h1, h2 = heights[start], heights[end]
            sections.append({
                "start_height": h1,
                "end_height": h2,
                "length": h2 - h1,
                "quality": qualities[end],
                "value": values[end] - values[start],
            })
            end = start

        return sections[::-1]
