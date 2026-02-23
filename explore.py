from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from dataclasses import dataclass
from datetime import time


@dataclass
class ExplorationResult:
    year: int | None
    distance: str | None
    n_races: int
    n_results: int
    n_participants: int
    # Normality of raw finish times
    shapiro_pvalues: dict[int, float]   # race_id -> p-value; <0.05 means not normal
    # Distribution shape per race
    skew_stats: pd.DataFrame            # columns: name, event_no, mean_pace, median_pace, mean_median_diff_pct, skew, kurtosis
    # Normalized pace spread (informs β scale)
    norm_pace_std: float
    norm_pace_skew: float
    # Pairwise gaps (calibrate GAP_MULTIPLIER)
    pairwise_median_gap: float
    pairwise_p75_gap: float
    pairwise_p90_gap: float
    suggested_multiplier: float         # 1 / pairwise_median_gap, i.e. β halves at median gap
    # Field composition (informs MIN_RACES)
    repeat_rate_2plus: float            # fraction of participants in 2+ races
    repeat_rate_3plus: float
    repeat_rate_4plus: float
    # Gap structure: do gaps grow toward the back?
    gap_trend_slope: float              # positive = gaps widen toward back of field
    # Participant consistency: is pace a stable signal?
    race_to_race_corr: float            # correlation between pace in race N and race N+1
    median_participant_std: float       # median within-runner pace std across races


def _to_seconds(t) -> float:
    if isinstance(t, time):
        return t.hour * 3600 + t.minute * 60 + t.second
    return float("nan")


class RaceExplorer:
    def __init__(self, races_df: pd.DataFrame, results_df: pd.DataFrame):
        self.races_df = races_df
        self.results_df = results_df

    # ------------------------------------------------------------------ #
    # Pipeline steps                                                       #
    # ------------------------------------------------------------------ #

    def _filter(
        self, year: int | None, distance: str | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = pd.Series(True, index=self.races_df.index)
        if year is not None:
            mask &= self.races_df["year"] == year
        if distance is not None:
            mask &= self.races_df["distance_name"].astype(str) == distance

        race_ids = self.races_df[mask].index
        races = self.races_df.loc[race_ids]
        results = self.results_df[self.results_df["race_id"].isin(race_ids)].copy()
        return races, results

    def _compute_pace(
        self, races: pd.DataFrame, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Add pace_s (seconds) and pace (s/km) columns. For 2025 finals uses track_time."""
        results = results.copy()

        def get_pace_seconds(row):
            race_meta = races.loc[row["race_id"]]
            is_final = race_meta["event_no"] == race_meta["no_events_season"]
            is_2025 = race_meta["year"] == 2025
            if is_final and is_2025 and pd.notna(row.get("track_time")):
                return _to_seconds(row["track_time"])
            return _to_seconds(row["result"])

        results["pace_s"] = results.apply(get_pace_seconds, axis=1)
        results["pace"] = results["pace_s"] / results["race_id"].map(races["distance_km"])
        return results

    def _compute_normalized_pace(self, results: pd.DataFrame) -> pd.DataFrame:
        """Add norm_pace = (pace - race_median) / race_median."""
        results = results.copy()
        median_pace = results.groupby("race_id")["pace"].transform("median")
        results["norm_pace"] = (results["pace"] - median_pace) / median_pace
        return results

    # ------------------------------------------------------------------ #
    # Analysis steps                                                       #
    # ------------------------------------------------------------------ #

    def _test_normality(
        self, races: pd.DataFrame, results: pd.DataFrame
    ) -> dict[int, float]:
        """Shapiro-Wilk p-value per race on raw finish seconds."""
        pvalues: dict[int, float] = {}
        for race_id in races.index:
            subset = results[results["race_id"] == race_id]["pace_s"].dropna()
            if len(subset) >= 8:
                _, p = stats.shapiro(subset)
                pvalues[int(race_id)] = float(p)
        return pvalues

    def _compute_skew_stats(
        self, races: pd.DataFrame, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Per-race mean/median pace and skewness of normalized pace."""
        rows = []
        for race_id in races.index:
            pace = results[results["race_id"] == race_id]["pace"].dropna()
            norm = results[results["race_id"] == race_id]["norm_pace"].dropna()
            mean_p, median_p = pace.mean(), pace.median()
            rows.append(
                {
                    "race_id": race_id,
                    "name": races.loc[race_id, "name"],
                    "event_no": races.loc[race_id, "event_no"],
                    "mean_pace": mean_p,
                    "median_pace": median_p,
                    "mean_median_diff_pct": (mean_p - median_p) / median_p * 100,
                    "skew": float(stats.skew(norm)) if len(norm) > 0 else float("nan"),
                    "kurtosis": float(stats.kurtosis(norm)) if len(norm) > 0 else float("nan"),
                }
            )
        return pd.DataFrame(rows).set_index("race_id")

    def _compute_pairwise_gaps(
        self, races: pd.DataFrame, results: pd.DataFrame
    ) -> tuple[np.ndarray, dict[int, float]]:
        """
        All pairwise |norm_pace| differences within each race.
        Returns the gap array and percentile dict.
        Note: O(n²) per race — fine for typical field sizes (~100-400 runners).
        """
        gaps: list[float] = []
        for race_id in races.index:
            subset = (
                results[results["race_id"] == race_id]["norm_pace"]
                .dropna()
                .sort_values()
                .values
            )
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    gaps.append(float(subset[j] - subset[i]))
        arr = np.array(gaps)
        percentiles = {p: float(np.percentile(arr, p)) for p in (25, 50, 75, 90)}
        return arr, percentiles

    def _compute_field_composition(self, results: pd.DataFrame) -> dict:
        """Fraction of participants who appear in N or more races."""
        race_counts = results.groupby("participant")["race_id"].count()
        n = len(race_counts)
        return {
            "race_counts": race_counts,
            "rate_2plus": float((race_counts >= 2).sum() / n),
            "rate_3plus": float((race_counts >= 3).sum() / n),
            "rate_4plus": float((race_counts >= 4).sum() / n),
        }

    def _compute_gap_structure(
        self, races: pd.DataFrame, results: pd.DataFrame
    ) -> dict:
        """
        Linear trend of consecutive-finisher gap vs finishing position percentile.
        Positive slope means gaps widen toward the back.
        """
        pos_pcts: list[float] = []
        gap_vals: list[float] = []
        for race_id in races.index:
            subset = (
                results[results["race_id"] == race_id]
                .sort_values("pace")["norm_pace"]
                .dropna()
                .values
            )
            n = len(subset)
            for i, gap in enumerate(np.diff(subset)):
                pos_pcts.append(i / n * 100)
                gap_vals.append(float(gap))
        slope = float(np.polyfit(pos_pcts, gap_vals, 1)[0])
        return {
            "trend_slope": slope,
            "median_consecutive_gap": float(np.median(gap_vals)),
        }

    def _compute_participant_consistency(self, results: pd.DataFrame) -> dict:
        """
        For repeat participants:
        - median within-runner norm_pace std across appearances
        - correlation between first and second race pace
        """
        repeat = results.groupby("participant").filter(lambda x: len(x) >= 2)

        consistency = repeat.groupby("participant")["norm_pace"].agg(["std", "count"])
        consistency.columns = ["pace_std", "n_races"]

        ordered = results.sort_values("race_id")
        first_pace = ordered.groupby("participant")["norm_pace"].nth(0).rename("pace_r1")
        second_pace = ordered.groupby("participant")["norm_pace"].nth(1).rename("pace_r2")
        paired = first_pace.to_frame().join(second_pace, how="inner").dropna()

        corr = float(paired["pace_r1"].corr(paired["pace_r2"])) if len(paired) > 1 else float("nan")
        median_std = float(consistency["pace_std"].median()) if len(consistency) > 0 else float("nan")

        return {"race_to_race_corr": corr, "median_participant_std": median_std}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(
        self, year: int | None = None, distance: str | None = None
    ) -> ExplorationResult:
        races, results = self._filter(year, distance)
        if races.empty:
            raise ValueError(f"No races found for year={year!r}, distance={distance!r}")

        results = self._compute_pace(races, results)
        results = self._compute_normalized_pace(results)

        shapiro = self._test_normality(races, results)
        skew_stats = self._compute_skew_stats(races, results)
        gaps, percentiles = self._compute_pairwise_gaps(races, results)
        composition = self._compute_field_composition(results)
        gap_structure = self._compute_gap_structure(races, results)
        consistency = self._compute_participant_consistency(results)

        norm_valid = results["norm_pace"].dropna()

        return ExplorationResult(
            year=year,
            distance=distance,
            n_races=len(races),
            n_results=len(results),
            n_participants=int(results["participant"].nunique()),
            shapiro_pvalues=shapiro,
            skew_stats=skew_stats,
            norm_pace_std=float(norm_valid.std()),
            norm_pace_skew=float(stats.skew(norm_valid)),
            pairwise_median_gap=percentiles[50],
            pairwise_p75_gap=percentiles[75],
            pairwise_p90_gap=percentiles[90],
            suggested_multiplier=round(1.0 / percentiles[50], 1),
            repeat_rate_2plus=composition["rate_2plus"],
            repeat_rate_3plus=composition["rate_3plus"],
            repeat_rate_4plus=composition["rate_4plus"],
            gap_trend_slope=gap_structure["trend_slope"],
            race_to_race_corr=consistency["race_to_race_corr"],
            median_participant_std=consistency["median_participant_std"],
        )

    def combinations(self) -> list[tuple[int, str]]:
        """All (year, distance) pairs present in the loaded data."""
        return list(
            self.races_df[["year", "distance_name"]]
            .drop_duplicates()
            .sort_values(["year", "distance_name"])
            .itertuples(index=False, name=None)
        )

    def run_all(self) -> list[ExplorationResult]:
        """Run analyze() for every (year, distance) combination in the data."""
        return [self.analyze(year, distance) for year, distance in self.combinations()]

    @staticmethod
    def summary_table(results: list[ExplorationResult]) -> pd.DataFrame:
        """Flatten a list of ExplorationResults into a single comparison DataFrame."""
        rows = []
        for r in results:
            rows.append(
                {
                    "year": r.year,
                    "distance": r.distance,
                    "n_races": r.n_races,
                    "n_participants": r.n_participants,
                    "norm_pace_std": round(r.norm_pace_std, 4),
                    "norm_pace_skew": round(r.norm_pace_skew, 3),
                    "median_gap": round(r.pairwise_median_gap, 4),
                    "suggested_multiplier": r.suggested_multiplier,
                    "repeat_2plus": f"{r.repeat_rate_2plus:.1%}",
                    "repeat_3plus": f"{r.repeat_rate_3plus:.1%}",
                    "corr_r1_r2": round(r.race_to_race_corr, 3),
                    "median_participant_std": round(r.median_participant_std, 4),
                    "gap_trend_slope": round(r.gap_trend_slope, 6),
                }
            )
        return pd.DataFrame(rows)
