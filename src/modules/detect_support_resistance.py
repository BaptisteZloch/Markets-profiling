from typing import Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy import stats, signal


# TODO: Create a function that generate a range level around each levels with a certain interval in % -> identify supply and demand zone.


def detect_levels(
    price_history: pd.Series,
    volume_history: Optional[pd.Series] = None,
    kde_factor: float = 0.075,
    total_levels: str | int = "all",
) -> pd.DataFrame:
    """Detect the support and resistance level over an historical time period using price and volume.

    Args:
        price_history (pd.Series[float | int]): The price history usually closing price.
        volume_history (pd.Series[float | int], optional): The volume history. Defaults to None.
        kde_factor (float, optional):  The coefficient used to calculate the estimator bandwidth. The higher coefficient is the strongest levels will only be detected. Defaults to 0.075.
        total_levels (str | int, optional): The total number of levels to detect. If "all" is provided, all levels will be detected. Defaults to "all".

    Returns:
        pd.DataFrame: The DataFrame containing the levels [min price, max price] and weights [0, 1] associated with each.
    """
    if volume_history is not None:
        assert len(price_history) == len(
            volume_history
        ), "Error, provide same size price and volume Series."

    # Generate a number of sample of the complete price history range in order to apply density estimation.
    xr = np.linspace(
        start=price_history.min(), stop=price_history.max(), num=len(price_history)
    )

    # Generate the kernel density estimation of the price weighted by volume over a certain number of sample xr.
    # It's possible to interpolate less precisely with decreasing the num parameter above.
    if volume_history is not None:
        estimated_density = stats.gaussian_kde(
            dataset=price_history, weights=volume_history, bw_method=kde_factor
        )(xr)
    else:
        estimated_density = stats.gaussian_kde(
            dataset=price_history, bw_method=kde_factor
        )(xr)

    def min_max_scaling(
        to_scale_array: npt.NDArray[np.float64],
        min_limit: int = 0,
        max_limit: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Min max scaling between 0 and 1.

        Args:
            to_scale_array (npt.NDArray[np.float64]): The array to scale.
            min_limit (int, optional): The lower limit of the range. Defaults to 0.
            max_limit (int, optional): The higher limit of the range. Defaults to 1.

        Returns:
            npt.NDArray[np.float64]: The scaled array.
        """
        return (to_scale_array - to_scale_array.min(axis=0)) / (
            to_scale_array.max(axis=0) - to_scale_array.min(axis=0)
        ) * (max_limit - min_limit) + min_limit

    # Find the index of the peaks over on a signal, here the estimated density.
    peaks, _ = signal.find_peaks(estimated_density)

    levels = xr[peaks]
    weights = min_max_scaling(estimated_density[peaks])
    
    df = pd.DataFrame({"levels": levels, "weights": weights})

    if isinstance(total_levels, int):
        assert (
            total_levels > 0
        ), "Error, provide a positive not null value for the total_levels parameter."

        if total_levels < len(levels):
            return df.sort_values(by="weights", ascending=False).head(total_levels)
    
    return df
