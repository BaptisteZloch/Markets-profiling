from typing import Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy import stats, signal


# TODO: Create a function that generate a range level around each levels with a certain interval in % -> identify supply and demand zone.


def detect_windowing_support_resistance(high: pd.Series, low: pd.Series) -> list[float]:
    """Detects the supports and resistances according to the windowing method

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.

    Returns:
        list[float]: The supports and resistances levels.
    """
    pivots: list[float] = []
    max_list = []
    min_list = []
    for i in range(5, len(high) - 5):
        # taking a window of 9 candles
        high_range = high.iloc[i - 5 : i + 4]
        current_max = high_range.max()
        # if we find a new maximum value, empty the max_list
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        # if the maximum value remains the same after shifting 5 times
        if len(max_list) == 5 and is_far_from_level(
            current_max,
            pivots,
            high,
            low,
        ):
            pivots.append(float(current_max))

        low_range = low[i - 5 : i + 5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(
            current_min,
            pivots,
            high,
            low,
        ):
            pivots.append(float(current_min))
    return pivots


def detect_fractal_support_resistance(high: pd.Series, low: pd.Series) -> list[float]:
    """Detects the supports and resistances according to the fractal method.

    Args:
        high (pd.Series): The high series from the OHLCV Data.
        low (pd.Series): The low series from the OHLCV Data.

    Returns:
        list[float]: The supports and resistances levels.
    """

    def is_fractal_support(low: pd.Series, i: int) -> bool:
        """Determine bullish fractal supports.

        Args:
            low (pd.Series): The low series from the OHLCV Data.
            i (int): The current index.

        Returns:
            bool: Whether it's a support or not according to fractal method.
        """
        return (
            low[i] < low[i - 1]
            and low[i] < low[i + 1]
            and low[i + 1] < low[i + 2]
            and low[i - 1] < low[i - 2]
        )

    def is_fractal_resistance(high: pd.Series, i: int) -> bool:
        """Determine bearish fractal resistances.

        Args:
            low (pd.Series): The high series from the OHLCV Data.
            i (int): The current index.

        Returns:
            bool: Whether it's a resistance or not according to fractal method.
        """
        return (
            high[i] > high[i - 1]
            and high[i] > high[i + 1]
            and high[i + 1] > high[i + 2]
            and high[i - 1] > high[i - 2]
        )

    levels: list[float] = []
    k = 5
    for i in range(k, high.shape[0] - k):
        if is_fractal_support(low, i):
            if is_far_from_level(
                low.iloc[i],
                levels,
                high,
                low,
            ):
                levels.append(float(low.iloc[i]))
        elif is_fractal_resistance(high, i):
            if is_far_from_level(
                high.iloc[i],
                levels,
                high,
                low,
            ):
                levels.append(float(high.iloc[i]))

    return levels


def detect_profiled_support_resistance(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    kde_factor: float = 0.075,
    total_levels: str | int = "all",
) -> pd.DataFrame:
    """Detect the support and resistance level over an historical time period using price and volume.

    Args:
        close (pd.Series[float | int]): The price history usually closing price.
        volume (pd.Series[float | int], optional): The volume history. Defaults to None.
        kde_factor (float, optional):  The coefficient used to calculate the estimator bandwidth. The higher coefficient is the strongest levels will only be detected. Defaults to 0.075.
        total_levels (str | int, optional): The total number of levels to detect. If "all" is provided, all levels will be detected. Defaults to "all".

    Returns:
        pd.DataFrame: The DataFrame containing the levels [min price, max price] and weights [0, 1] associated with each.
    """
    if volume is not None:
        assert len(close) == len(
            volume
        ), "Error, provide same size price and volume Series."

    # Generate a number of sample of the complete price history range in order to apply density estimation.
    xr = np.linspace(start=close.min(), stop=close.max(), num=len(close))

    # Generate the kernel density estimation of the price weighted by volume over a certain number of sample xr.
    # It's possible to interpolate less precisely with decreasing the num parameter above.
    if volume is not None:
        estimated_density = stats.gaussian_kde(
            dataset=close, weights=volume, bw_method=kde_factor
        )(xr)
    else:
        estimated_density = stats.gaussian_kde(dataset=close, bw_method=kde_factor)(xr)

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


# to make sure the new level area does not exist already
def is_far_from_level(
    value: float, levels: list[float], high: pd.Series, low: pd.Series
) -> bool:
    ave = np.mean(high - low)
    return np.sum([abs(value - level) < ave for level in levels]) == 0
