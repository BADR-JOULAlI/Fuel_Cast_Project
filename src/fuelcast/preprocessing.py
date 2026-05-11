from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FuelCastPreprocessor(BaseEstimator, TransformerMixin):
    """Reusable preprocessing for the FuelCast dataset.

    The transformer keeps the deployment path close to the notebook while
    preventing target leakage. Any feature containing MomentaryFuel is removed
    before training because those columns directly encode fuel consumption.
    """

    target = "Consumer_Total_MomentaryFuel"

    redundant_columns = [
        "Propeller_Port_ShaftPower",
        "Propeller_Starboard_ShaftPower",
        "Propeller_Starboard_ShaftTorque",
        "Propeller_Starboard_RotationSpeed",
        "Weather_SwellWaveHeight",
        "Weather_SwellWavePeriod",
        "Weather_WindWaveHeight",
        "Weather_WindWavePeriod",
    ]

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y=None):
        prepared = self._prepare_frame(X, fit=True)

        self.numeric_columns_ = prepared.select_dtypes(include="number").columns.tolist()
        self.categorical_columns_ = prepared.select_dtypes(include="object").columns.tolist()

        self.numeric_medians_ = prepared[self.numeric_columns_].median()
        self.lower_bounds_ = prepared[self.numeric_columns_].quantile(self.lower_q)
        self.upper_bounds_ = prepared[self.numeric_columns_].quantile(self.upper_q)

        if self.categorical_columns_:
            self.categorical_modes_ = {
                col: prepared[col].mode(dropna=True).iloc[0]
                if not prepared[col].mode(dropna=True).empty
                else "Unknown"
                for col in self.categorical_columns_
            }
        else:
            self.categorical_modes_ = {}

        encoded = self._encode(prepared, fit=True)
        self.feature_names_out_ = encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        prepared = self._prepare_frame(X, fit=False)

        for col in self.numeric_columns_:
            if col not in prepared.columns:
                prepared[col] = np.nan

        for col in self.categorical_columns_:
            if col not in prepared.columns:
                prepared[col] = self.categorical_modes_.get(col, "Unknown")

        prepared[self.numeric_columns_] = prepared[self.numeric_columns_].fillna(
            self.numeric_medians_
        )
        prepared[self.numeric_columns_] = prepared[self.numeric_columns_].clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1,
        )

        for col, mode in self.categorical_modes_.items():
            prepared[col] = prepared[col].fillna(mode)

        encoded = self._encode(prepared, fit=False)
        return encoded

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _prepare_frame(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = X.copy()

        if self.target in df.columns:
            df = df.drop(columns=[self.target])

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        self._add_feature_if_possible(
            df,
            "Total_Engine_ShaftPower",
            [col for col in df.columns if "ShaftPower" in col and "GeneratorEngine" in col],
        )
        self._add_feature_if_possible(
            df,
            "Total_Propeller_Power",
            ["Propeller_Port_ShaftPower", "Propeller_Starboard_ShaftPower"],
        )

        if {"Propeller_Port_ShaftPower", "Propeller_Starboard_ShaftPower"}.issubset(df.columns):
            df["Propulsion_Balance_Ratio"] = (
                df["Propeller_Port_ShaftPower"]
                / (df["Propeller_Starboard_ShaftPower"] + 1e-6)
            )

        if {"Weather_WaveHeight", "Weather_WindSpeed10M"}.issubset(df.columns):
            df["Weather_Severity_Index"] = (
                df["Weather_WaveHeight"] * df["Weather_WindSpeed10M"]
            )

        if {"Weather_OceanCurrentVelocity", "Ship_SpeedOverGround"}.issubset(df.columns):
            df["Ocean_Resistance_Index"] = (
                df["Weather_OceanCurrentVelocity"] * df["Ship_SpeedOverGround"]
            )

        if {"Ship_SpeedOverGround", "Total_Propeller_Power"}.issubset(df.columns):
            df["Propulsion_Efficiency"] = (
                df["Ship_SpeedOverGround"] / (df["Total_Propeller_Power"] + 1e-6)
            )

        if "Propulsion_Balance_Ratio" in df.columns:
            df["Propulsion_Balance_Ratio_log"] = np.log1p(
                df["Propulsion_Balance_Ratio"].clip(lower=0)
            )

        leakage_cols = [
            col
            for col in df.columns
            if "MomentaryFuel" in col or col == "Total_Engine_Fuel"
        ]
        generator_cols = [col for col in df.columns if "GeneratorEngine" in col]

        df = df.drop(
            columns=leakage_cols + generator_cols + self.redundant_columns,
            errors="ignore",
        )

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return df

    @staticmethod
    def _add_feature_if_possible(df: pd.DataFrame, name: str, columns: list[str]):
        available = [col for col in columns if col in df.columns]
        if available:
            df[name] = df[available].sum(axis=1)

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        base_cols = self.numeric_columns_ + self.categorical_columns_
        encoded = pd.get_dummies(df[base_cols], columns=self.categorical_columns_, drop_first=True)

        if fit:
            return encoded

        return encoded.reindex(columns=self.feature_names_out_, fill_value=0)
