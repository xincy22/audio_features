"""特征聚合工具。"""

import numpy as np
from scipy import stats


class FeatureAggregator:
    """将帧级特征聚合为片段级统计特征。"""

    def aggregate_features(self, features, aggregation_methods):
        """按指定统计方法聚合特征。

        Parameters
        ----------
        features : dict
            特征字典，键为特征名称，值为特征数组。
        aggregation_methods : list or tuple
            聚合方法列表，支持 ``mean``、``std``、``min``、``max``、``median``、
            ``skewness``、``kurtosis``、``range``、``quantile_25``、``quantile_75``。

        Returns
        -------
        dict
            聚合后的特征字典。

        Raises
        ------
        ValueError
            输入非法或聚合方法不支持时抛出。
        """
        if not isinstance(features, dict):
            raise ValueError("features must be a dict")
        if not isinstance(aggregation_methods, (list, tuple)):
            raise ValueError("aggregation_methods must be a list or tuple")

        aggregated = {}
        for name, values in features.items():
            arr = np.asarray(values)
            if arr.ndim == 1:
                axis = 0
            elif arr.ndim == 2:
                axis = 1
            else:
                raise ValueError("feature arrays must be 1D or 2D")

            for method in aggregation_methods:
                key = f"{name}_{method}"
                if method == "mean":
                    aggregated[key] = np.mean(arr, axis=axis)
                elif method == "std":
                    aggregated[key] = np.std(arr, axis=axis)
                elif method == "min":
                    aggregated[key] = np.min(arr, axis=axis)
                elif method == "max":
                    aggregated[key] = np.max(arr, axis=axis)
                elif method == "median":
                    aggregated[key] = np.median(arr, axis=axis)
                elif method == "skewness":
                    aggregated[key] = stats.skew(arr, axis=axis, bias=False)
                elif method == "kurtosis":
                    aggregated[key] = stats.kurtosis(arr, axis=axis, bias=False)
                elif method == "range":
                    aggregated[key] = np.max(arr, axis=axis) - np.min(arr, axis=axis)
                elif method == "quantile_25":
                    aggregated[key] = np.quantile(arr, 0.25, axis=axis)
                elif method == "quantile_75":
                    aggregated[key] = np.quantile(arr, 0.75, axis=axis)
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")

        return aggregated

    def aggregate_statistics(self, features):
        """计算常用统计量并按特征归类。

        Parameters
        ----------
        features : dict
            特征字典，键为特征名称，值为特征数组。

        Returns
        -------
        dict
            嵌套字典：``{feature_name: {stat_name: value}}``。
        """
        stats_methods = ["mean", "std", "min", "max", "median", "range"]
        aggregated = self.aggregate_features(features, stats_methods)

        grouped = {}
        for key, value in aggregated.items():
            feature_name, stat = key.rsplit("_", 1)
            grouped.setdefault(feature_name, {})[stat] = value
        return grouped
