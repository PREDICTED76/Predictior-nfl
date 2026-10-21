import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictor | NFL Analytics", page_icon="🏈", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "ridge_model.joblib"
FEATURES_PATH = BASE / "feature_cols.json"
FORM_PATH = BASE / "current_form.csv"
ELO_PATH = BASE / "current_elo.csv"
WEEKLY_PATH = BASE / "week7_predictions.csv"
HIST_PATH = BASE / "historical_data.csv"

NAVY = "#07111f"
PANEL = "#101d30"
BLUE = "#2784ff"
GOLD = "#ffd45a"
MUTED = "#9aacc4"
GREEN = "#3ce0a1"

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at top, #12223a 0, #07111f 60%); color:#f3f7ff}
[data-testid="stSidebar"] {background:#091525;border-right:1px solid #263851}
.block-container {padding-top:1.5rem;max-width:1400px}
h1,h2,h3 {letter-spacing:-.03em}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#132238,#0d192a);border:1px solid #263851;padding:14px;border-radius:14px}
div[data-testid="stMetricLabel"] {color:#9aacc4}
div[data-testid="stMetricValue"] {color:#f3f7ff}
div.stButton>button {border-radius:10px;border:1px solid #315276}
hr {border-color:#263851}
.small-muted {color:#9aacc4;font-size:.85rem}
.warning {padding:12px 15px;border-radius:10px;background:#282312;border:1px solid #5c4b20;color:#f4d77e}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

required = [MODEL_PATH, FEATURES_PATH, FORM_PATH, ELO_PATH]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("Missing model package files: " + ", ".join(missing))
    st.stop()

model = load_model()
features = json.loads(FEATURES_PATH.read_text())
form = load_csv(FORM_PATH)
elo = load_csv(ELO_PATH)
weekly = load_csv(WEEKLY_PATH) if WEEKLY_PATH.exists() else pd.DataFrame()
hist = load_csv(HIST_PATH) if HIST_PATH.exists() else pd.DataFrame()

form["team"] = form["team"].astype(str)
elo["team"] = elo["team"].astype(str)
form_by_team = form.set_index("team").to_dict("index")
elo_by_team = elo.set_index("team")["current_elo"].to_dict()
teams = sorted(set(form_by_team) & set(elo_by_team))

TEAM_NAMES = {
"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
"CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
"DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
"HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
"LA":"Los Angeles Rams","LAC":"Los Angeles Chargers","LV":"Las Vegas Raiders","MIA":"Miami Dolphins",
"MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
"NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SF":"San Francisco 49ers",
"SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"
}

def team_label(code):
    return f"{code} — {TEAM_NAMES.get(code, code)}"

def feature_row(away, home):
    a, h = form_by_team[away], form_by_team[home]
    # Keep feature definitions consistent with the current Colab weekly pipeline.
    values = {
        "home_margin_form": h["margin_last5"],
        "home_offense_form": h["points_for_last5"],
        "home_defense_form": h["points_against_last5"],
        "away_margin_form": a["margin_last5"],
        "away_offense_form": a["points_for_last5"],
        "away_defense_form": a["points_against_last5"],
        "elo_diff": float(elo_by_team[home]) - float(elo_by_team[away]) + 55.0,
    }
    return pd.DataFrame([[values[c] for c in features]], columns=features)

def predict(away, home):
    X = feature_row(away, home)
    margin = float(model.predict(X)[0])
    # Illustrative probability conversion, not calibrated by the model.
    home_prob = 1 / (1 + np.exp(-margin / 6.0))
    return margin, float(home_prob), X.iloc[0].to_dict()

def radar(away, home):
    a, h = form_by_team[away], form_by_team[home]
    # This chart uses recent score-derived measures, normalized for visualization;
    # it does not imply the model was trained on all four axes.
    labels = ["Recent margin", "Points scored", "Defense (lower PA)", "Elo rating"]
    def vals(code):
        f = form_by_team[code]
        return [
            float(np.clip((f["margin_last5"] + 15) / 30 * 100, 0, 100)),
            float(np.clip(f["points_for_last5"] / 40 * 100, 0, 100)),
            float(np.clip((40 - f["points_against_last5"]) / 25 * 100, 0, 100)),
            float(np.clip((elo_by_team[code] - 1250) / 600 * 100, 0, 100)),
        ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals(away), theta=labels, fill="toself", name=away,
                                  line=dict(color=GOLD, width=3), fillcolor="rgba(255,212,90,.15)"))
    fig.add_trace(go.Scatterpolar(r=vals(home), theta=labels, fill="toself", name=home,
                                  line=dict(color=BLUE, width=3), fillcolor="rgba(39,132,255,.17)"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
          radialaxis=dict(visible=True, range=[0,100], gridcolor="#30435c", tickfont=dict(color=MUTED, size=9)),
          angularaxis=dict(gridcolor="#30435c", tickfont=dict(color="#e6efff", size=11))),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f7ff"), legend=dict(orientation="h", y=-.12),
        margin=dict(l=30,r=30,t=20,b=30), height=360
    )
    return fig

with st.sidebar:
    st.markdown("# 🏈 Predictor")
    st.caption("Data. Matchups. Edges.")
    page = st.radio("NAVIGATION", ["Home", "Matchups", "Weekly Picks", "Teams", "Performance"], label_visibility="collapsed")
    st.divider()
    st.caption("MODEL STATUS")
    st.success("Ridge model loaded")
    st.caption("Pregame form + Elo features")
    st.caption("No live sportsbook feed connected")

st.title("Predictor")
st.caption("NFL analytics dashboard · trained Ridge pipeline loaded from your Colab export")
st.markdown('<div class="warning">Win probabilities are approximate conversions from predicted margin, not calibrated probabilities. Current team form/Elo are a snapshot from your exported package and will not update automatically.</div>', unsafe_allow_html=True)

if page == "Home":
    st.subheader("Week 7 overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Teams loaded", len(teams))
    c2.metric("Weekly games", len(weekly) if len(weekly) else 0)
    c3.metric("Historical games", len(hist) if len(hist) else "—")
    st.subheader("Featured matchup")
    if len(weekly):
        choices = [f"{r.away_team} @ {r.home_team}" for r in weekly.itertuples()]
        choice = st.selectbox("Select a Week 7 game", choices)
        idx = choices.index(choice)
        row = weekly.iloc[idx]
        away, home = row["away_team"], row["home_team"]
        margin, p_home, _ = predict(away, home)
        left, mid, right = st.columns([2,1,2])
        left.metric(f"{away} · {TEAM_NAMES.get(away,away)}", f"{(1-p_home)*100:.1f}%", "Approx. win probability")
        mid.metric("Projected margin", f"{home if margin >= 0 else away} +{abs(margin):.1f}")
        right.metric(f"{home} · {TEAM_NAMES.get(home,home)}", f"{p_home*100:.1f}%", "Approx. win probability")
        st.plotly_chart(radar(away, home), use_container_width=True)
        st.caption("The radar visualizes recent form and Elo for context; it is not the Ridge model's direct feature-importance chart.")
    st.subheader("Power index leaders")
    lead = elo.sort_values("current_elo", ascending=False).head(8).copy()
    lead["Team"] = lead["team"].map(team_label)
    lead = lead.rename(columns={"current_elo":"Current Elo"})
    st.dataframe(lead[["Team","Current Elo"]].round(1), use_container_width=True, hide_index=True)

elif page == "Matchups":
    st.subheader("Matchup lab")
    c1,c2 = st.columns(2)
    away = c1.selectbox("Away team", teams, index=teams.index("SF") if "SF" in teams else 0, format_func=team_label)
    home = c2.selectbox("Home team", teams, index=teams.index("KC") if "KC" in teams else min(1,len(teams)-1), format_func=team_label)
    margin, p_home, row = predict(away, home)
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{away} win probability*", f"{(1-p_home)*100:.1f}%")
    c2.metric(f"{home} win probability*", f"{p_home*100:.1f}%")
    c3.metric("Predicted home margin", f"{margin:+.2f} pts")
    st.markdown(f"**Model lean:** {home if margin >= 0 else away} by approximately {abs(margin):.1f} points")
    st.plotly_chart(radar(away, home), use_container_width=True)
    st.subheader("Exact model inputs")
    st.dataframe(pd.DataFrame({"Feature":list(row.keys()),"Value":list(row.values())}), use_container_width=True, hide_index=True)
    st.caption("*Approximate probability derived from predicted margin using a simple logistic conversion; it is not yet calibrated on held-out games.")

elif page == "Weekly Picks":
    st.subheader("Weekly predictions exported from Colab")
    if weekly.empty:
        st.info("No weekly_predictions.csv file was found.")
    else:
        show = weekly.copy()
        preferred = [c for c in ["gameday","away_team","home_team","spread_line","projected_home_margin","model_edge","edge_abs","model_pick","decision","line_status"] if c in show.columns]
        st.dataframe(show[preferred], use_container_width=True, hide_index=True)
        st.download_button("Download weekly predictions CSV", show.to_csv(index=False), "weekly_predictions.csv", "text/csv")
        st.caption("These are the frozen exported predictions. The app does not fetch current odds or update scores.")

elif page == "Teams":
    st.subheader("Team profiles")
    team = st.selectbox("Choose a team", teams, format_func=team_label)
    f = form_by_team[team]
    e = elo_by_team[team]
    a,b,c,d = st.columns(4)
    a.metric("Current Elo", f"{e:.0f}")
    b.metric("Last 5 margin", f"{f['margin_last5']:+.1f}")
    c.metric("Points scored / game", f"{f['points_for_last5']:.1f}")
    d.metric("Points allowed / game", f"{f['points_against_last5']:.1f}")
    st.plotly_chart(radar(team, teams[(teams.index(team)+1) % len(teams)]), use_container_width=True)
    st.caption("Team values reflect the snapshot exported from Colab; recent-form metrics are descriptive, not guarantees of future performance.")

elif page == "Performance":
    st.subheader("Model performance")
    m1,m2,m3 = st.columns(3)
    m1.metric("Walk-forward MAE", "9.99 pts")
    m2.metric("Walk-forward RMSE", "12.90 pts")
    m3.metric("ATS edge proven?", "No")
    st.caption("Recorded from the notebook's 2022–2025 walk-forward evaluation. These summary metrics are fixed labels in this first app version.")
    if len(hist) and "season" in hist.columns:
        st.subheader("Historical data coverage")
        seasons = hist.groupby("season").size().reset_index(name="Games")
        st.dataframe(seasons, use_container_width=True, hide_index=True)
    st.markdown("#### Validation checklist")
    st.write("- Preserve a pregame prediction snapshot before each kickoff.")
    st.write("- Verify line source, line convention, and closing-line timestamp.")
    st.write("- Re-test with strictly chronological holdout seasons.")
    st.write("- Calibrate win probabilities separately before presenting them as reliable probabilities.")

st.divider()
st.caption("Predictor prototype · Model package exported 2026-10-21 · No live odds, injuries, or automatic score updates.")
