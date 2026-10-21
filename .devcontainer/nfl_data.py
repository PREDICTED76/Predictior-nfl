
import streamlit as st
import nflreadpy as nfl
import pandas as pd


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_stats(season: int) -> pd.DataFrame:
    """Load player statistics for a season."""
    data = nfl.load_player_stats([season])
    return data.to_pandas()


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_stats(season: int) -> pd.DataFrame:
    """Load team statistics for a season."""
    data = nfl.load_team_stats([season])
    return data.to_pandas()


@st.cache_data(ttl=3600, show_spinner=False)
def load_snap_counts(season: int) -> pd.DataFrame:
    """Load player snap counts for a season."""
    data = nfl.load_snap_counts([season])
    return data.to_pandas()


def get_data_status() -> dict:
    return {
        "player_stats": "Available to load",
        "team_stats": "Available to load",
        "snap_counts": "Available to load",
    }
