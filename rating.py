import json
import argparse
import trueskill
import constants
import pandas as pd
import matplotlib.pyplot as plt

from models.stirnu_buks import parse_stirnu_buks_to_dataframes
from collections import defaultdict
from datetime import time


RACES_FILE_PATH = constants.RACES_FILE_PATH
BETA_MIN = constants.BETA_MIN
BETA_BASE = constants.BETA_BASE
GAP_MULTIPLIER = constants.GAP_MULTIPLIER


def time_to_seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second


def get_race_ids(races_df: pd.DataFrame, year: int, distance_name: str) -> list[int]:
    mask = (races_df["year"] == year) & (races_df["distance_name"] == distance_name)
    return races_df[mask].sort_values("event_no").index.tolist()


def calculate_trueskill(
    races_df: pd.DataFrame,
    results_df: pd.DataFrame,
    year: int,
    distance_name: str,
) -> pd.DataFrame:
    env = trueskill.TrueSkill(beta=BETA_BASE, draw_probability=0.0)
    ratings = defaultdict(lambda: env.create_rating())
    race_counts = defaultdict(int)

    race_ids = get_race_ids(races_df, year, distance_name)

    for race_id in race_ids:
        race_meta = races_df.loc[race_id]
        is_final = race_meta["event_no"] == race_meta["no_events_season"]

        race_results = results_df[results_df["race_id"] == race_id].copy()

        # Pick the right time field
        if is_final:
            race_results = race_results.dropna(subset=["track_time"])
            race_results["sort_time"] = race_results["track_time"]
        else:
            race_results["sort_time"] = race_results["result"]

        race_results = race_results.sort_values("sort_time")
        runners = race_results["participant"].tolist()

        rating_groups = [{r: ratings[r]} for r in runners]
        ranks = list(range(len(runners)))

        updated_groups = env.rate(rating_groups, ranks=ranks)

        for group in updated_groups:
            for runner, rating in group.items():
                ratings[runner] = rating
                race_counts[runner] += 1

    # Build leaderboard
    rows = []
    for runner, rating in ratings.items():
        rows.append(
            {
                "participant": runner,
                "mu": round(rating.mu, 2),
                "sigma": round(rating.sigma, 2),
                "score": round(rating.mu - 3 * rating.sigma, 2),
                "races": race_counts[runner],
            }
        )

    leaderboard = (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )
    leaderboard.index += 1
    return leaderboard


def calculate_pace_adjusted_trueskill(
    races_df: pd.DataFrame,
    results_df: pd.DataFrame,
    year: int,
    distance_name: str,
) -> pd.DataFrame:
    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = defaultdict(lambda: env.create_rating())
    race_counts = defaultdict(int)

    race_ids = get_race_ids(races_df, year, distance_name)

    for race_id in race_ids:
        race_meta = races_df.loc[race_id]
        is_final = race_meta["event_no"] == race_meta["no_events_season"]

        race_results = results_df[results_df["race_id"] == race_id].copy()

        if is_final:
            race_results = race_results.dropna(subset=["track_time"])
            race_results["seconds"] = race_results["track_time"].apply(time_to_seconds)
            distance_km = race_meta["distance_km"]
        else:
            race_results["seconds"] = race_results["result"].apply(time_to_seconds)
            distance_km = race_meta["distance_km"]

        race_results["pace"] = race_results["seconds"] / distance_km
        median_pace = race_results["pace"].median()
        race_results["norm_pace"] = (race_results["pace"] - median_pace) / median_pace

        race_results = race_results.sort_values("pace").reset_index(drop=True)
        runners = race_results["participant"].tolist()

        # Pairwise 1v1 updates
        for i in range(len(runners)):
            for j in range(i + 1, len(runners)):
                winner, loser = runners[i], runners[j]
                gap_norm = abs(
                    race_results.loc[i, "norm_pace"] - race_results.loc[j, "norm_pace"]
                )
                # Large normalized gap → low beta → decisive win
                # Small normalized gap → beta near base → uncertain
                dynamic_beta = max(BETA_MIN, BETA_BASE / (1 + gap_norm * GAP_MULTIPLIER))
                env_dynamic = trueskill.TrueSkill(
                    beta=dynamic_beta, draw_probability=0.0
                )

                new_winner, new_loser = env_dynamic.rate_1vs1(
                    ratings[winner], ratings[loser]
                )
                ratings[winner] = new_winner
                ratings[loser] = new_loser

        for runner in runners:
            race_counts[runner] += 1

    rows = []
    for runner, rating in ratings.items():
        rows.append(
            {
                "participant": runner,
                "mu": round(rating.mu, 2),
                "sigma": round(rating.sigma, 2),
                "score": round(rating.mu - 3 * rating.sigma, 2),
                "races": race_counts[runner],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda df: range(1, len(df) + 1))
        .set_index("rank")
    )


def plot_leaderboard(
    leaderboard: pd.DataFrame, top_n: int = 10, sort_by: str = "score"
):
    df = leaderboard.copy()
    df = df.sort_values(sort_by).tail(top_n)

    fig, ax = plt.subplots()

    ax.barh(
        df["participant"],
        df["mu"],
        xerr=df["sigma"] * 3,
        capsize=4,
        color="steelblue",
        alpha=0.7,
        error_kw={"ecolor": "gray"},
    )

    ax.set_xlabel("TrueSkill Score (μ ± 3σ)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate TrueSkill for 1 race season."
    )

    parser.add_argument("--year", type=int, required=True, help="Race season year.")

    parser.add_argument(
        "--distance",
        type=str,
        default="stirnu_buks",
        help="Name of the race distance (vāvere, zaķis, stirnu_buks, lūsis)",
    )

    parser.add_argument(
        "--path", type=str, default="races.json", help="The file path to the results"
    )

    args = parser.parse_args()

    races = json.loads(RACES_FILE_PATH.read_text())
    races_sb_raw: list[dict] = races.get("stirnu_buks")

    # TODO: Eventually add races to a DB and query that
    races_sb, results_sb = parse_stirnu_buks_to_dataframes(races_sb_raw)

    # leaderboard = calculate_trueskill(
    #     races_sb,
    #     results_sb,
    #     args.year,
    #     args.distance,
    # )

    leaderboard = calculate_pace_adjusted_trueskill(
        races_sb,
        results_sb,
        args.year,
        args.distance,
    )

    plot_leaderboard(leaderboard, sort_by="mu")
