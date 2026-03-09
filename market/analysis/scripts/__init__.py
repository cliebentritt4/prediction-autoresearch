"""Registry of available analysis scripts."""

from market.analysis.scripts.price_distribution import PriceDistributionAnalysis
from market.analysis.scripts.volume_analysis import VolumeAnalysis

ANALYSES = {
    "price_distribution": PriceDistributionAnalysis,
    "volume_analysis": VolumeAnalysis,
}
