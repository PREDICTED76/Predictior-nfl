
from math import erf, sqrt
import pandas as pd


def over_probability(line, projection, std_dev):
    """Estimate the chance a player's result exceeds a prop line."""
    std_dev = max(float(std_dev), 0.1)
    z = (float(line) - float(projection)) / (std_dev * sqrt(2))
    return 0.5 * (1 - erf(z))


def project_player_prop(
    player,
    team,
    opponent,
    prop_type,
    season_average,
    recent_average,
    usage_change_pct,
    opponent_allowed,
    league_average_allowed,
    line,
    std_dev,
):
    """Create a transparent, matchup-adjusted player prop estimate."""

    season_average = float(season_average)
    recent_average = float(recent_average)
    league_average_allowed = max(float(league_average_allowed), 0.1)
    opponent_allowed = max(float(opponent_allowed), 0.0)

    # Blend longer-term performance with recent form.
    base_projection = 0.65 * season_average + 0.35 * recent_average

    # Adjust for the opponent's defense relative to league average.
    defense_factor = opponent_allowed / league_average_allowed
    defense_factor = max(0.75, min(defense_factor, 1.25))

    # Limit the impact of changes in usage to avoid extreme projections.
    usage_factor = 1 + max(-0.20, min(float(usage_change_pct) / 100, 0.20))

    projection = base_projection * defense_factor * usage_factor
    probability_over = over_probability(line, projection, std_dev)
    probability_over = max(0.01, min(probability_over, 0.99))

    return {
        "Player": player,
        "Team": team,
        "Opponent": opponent,
        "Prop": prop_type,
        "Line": round(float(line), 1),
        "Projection": round(projection, 1),
        "Over Probability": round(probability_over * 100, 1),
        "Under Probability": round((1 - probability_over) * 100, 1),
        "Defense Factor": round(defense_factor, 3),
        "Usage Change (%)": round(float(usage_change_pct), 1),
        "Status": "Experimental — not yet calibrated",
    }


def rank_player_props(props):
    """Sort candidate props by distance between projection and line."""
    if not props:
        return pd.DataFrame()

    frame = pd.DataFrame(props)
    frame["Projection Edge"] = frame["Projection"] - frame["Line"]
    return frame.sort_values("Over Probability", ascending=False)
