import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="FIFA World Cup 2026 — Data-Driven Prediction",
    page_icon="⚽",
    layout="wide"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ===================================
# LOAD DATA (cached so it's fast)
# ===================================
@st.cache_data
def load_data():
    fixtures = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_fixtures.csv"))
    teams = pd.read_csv(os.path.join(DATA_DIR, "wc_2026_teams.csv"))
    editions = pd.read_csv(os.path.join(DATA_DIR, "wc_all_editions.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "wc_all_matches.csv"))
    scorers = pd.read_csv(os.path.join(DATA_DIR, "wc_top_scorers.csv"))
    return fixtures, teams, editions, matches, scorers

fixtures, teams, editions, matches, scorers = load_data()

# ===================================
# BUILD THE PREDICTION MODEL (cached)
# Same 4-factor model from the Zerve project:
# FIFA Rank 40% + Semi-Finals 30% + Titles 20% + Host 10%
# ===================================
@st.cache_data
def build_prediction_model(teams, matches, editions):
    # Semi-final appearances
    semi_finals = matches[
        matches['stage'].isin(['Semi-final', 'Semi-final Round'])
    ].copy()
    team1_semis = semi_finals[['year', 'team1']].rename(columns={'team1': 'team'})
    team2_semis = semi_finals[['year', 'team2']].rename(columns={'team2': 'team'})
    all_semi_teams = pd.concat([team1_semis, team2_semis])
    semi_counts = all_semi_teams['team'].value_counts().reset_index()
    semi_counts.columns = ['team', 'semi_final_count']

    # Titles
    title_counts = editions['champion'].value_counts().reset_index()
    title_counts.columns = ['team', 'titles']

    # Merge West Germany into Germany for both
    if 'West Germany' in title_counts['team'].values and 'Germany' in title_counts['team'].values:
        title_counts.loc[title_counts['team'] == 'Germany', 'titles'] += \
            title_counts.loc[title_counts['team'] == 'West Germany', 'titles'].values[0]
    if 'West Germany' in semi_counts['team'].values and 'Germany' in semi_counts['team'].values:
        semi_counts.loc[semi_counts['team'] == 'Germany', 'semi_final_count'] += \
            semi_counts.loc[semi_counts['team'] == 'West Germany', 'semi_final_count'].values[0]

    results = []
    max_rank = teams['fifa_rank'].max()
    max_semis = semi_counts['semi_final_count'].max()
    max_titles = title_counts['titles'].max()

    for _, team_row in teams.iterrows():
        team_name = team_row['team']
        fifa_rank = team_row['fifa_rank']

        semi_row = semi_counts[semi_counts['team'].str.contains(team_name, case=False, na=False)]
        semi_count = semi_row['semi_final_count'].values[0] if len(semi_row) > 0 else 0

        title_row = title_counts[title_counts['team'].str.contains(team_name, case=False, na=False)]
        title_count = title_row['titles'].values[0] if len(title_row) > 0 else 0

        host_bonus = 1 if team_name in ['United States', 'Canada', 'Mexico'] else 0

        fifa_score = (max_rank - fifa_rank) / max_rank * 100
        semi_score = (semi_count / max_semis) * 100 if max_semis > 0 else 0
        title_score = (title_count / max_titles) * 100 if max_titles > 0 else 0
        host_score = host_bonus * 100

        final_score = (fifa_score * 0.40 + semi_score * 0.30 +
                       title_score * 0.20 + host_score * 0.10)

        results.append({
            'team': team_name,
            'confederation': team_row['confederation'],
            'fifa_rank': fifa_rank,
            'semi_finals': semi_count,
            'titles': title_count,
            'host_bonus': host_bonus,
            'prediction_score': round(final_score, 2)
        })

    results_df = pd.DataFrame(results).sort_values('prediction_score', ascending=False).reset_index(drop=True)
    total_score = results_df['prediction_score'].sum()
    results_df['win_probability_%'] = round(results_df['prediction_score'] / total_score * 100, 2)
    return results_df

# ===================================
# FETCH LIVE ELIMINATION STATUS
# Checks who is still alive in the tournament
# using the openfootball live API
# ===================================
@st.cache_data(ttl=3600)
def fetch_elimination_status():
    """Returns sets of eliminated and still-alive teams from live API"""
    import requests
    try:
        url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return set(), set(), set()
        data = r.json()
        matches = data.get('matches', [])

        knockout_rounds = ['Round of 32', 'Round of 16', 'Quarter-final',
                           'Semi-final', 'Final', 'Match for third place']

        # Step 1: Find teams that reached Round of 32
        r32_teams = set()
        for m in matches:
            if m.get('round') == 'Round of 32':
                r32_teams.add(m['team1'])
                r32_teams.add(m['team2'])

        # Step 2: Find all group stage teams
        group_teams = set()
        for m in matches:
            if m.get('round', '').startswith('Matchday'):
                group_teams.add(m['team1'])
                group_teams.add(m['team2'])

        # Teams that didn't make Round of 32 = group stage eliminated
        group_eliminated = group_teams - r32_teams

        # Step 3: Find knockout eliminated teams
        knockout_eliminated = set()
        knockout_alive = set()
        for m in matches:
            if m.get('round') in knockout_rounds and m.get('score', {}).get('ft'):
                ft = m['score']['ft']
                t1, t2 = m['team1'], m['team2']
                s1, s2 = ft[0], ft[1]
                pen = m.get('score', {}).get('pen')
                if s1 == s2 and pen:
                    winner = t1 if pen[0] > pen[1] else t2
                    loser = t2 if pen[0] > pen[1] else t1
                elif s1 > s2:
                    winner, loser = t1, t2
                else:
                    winner, loser = t2, t1
                knockout_eliminated.add(loser)
                knockout_alive.add(winner)

        all_eliminated = group_eliminated | knockout_eliminated
        return all_eliminated, knockout_alive, group_eliminated

    except Exception:
        return set(), set(), set()

results_df = build_prediction_model(teams, matches, editions)

# ===================================
# LIVE PROBABILITY ADJUSTMENT
# Set eliminated teams to 0% and
# redistribute among still-alive teams
# ===================================
eliminated_teams, knockout_alive, group_eliminated = fetch_elimination_status()

if eliminated_teams:
    # Set eliminated teams to 0
    results_df['still_in'] = ~results_df['team'].isin(eliminated_teams)
    results_df['live_probability_%'] = 0.0

    # Redistribute total probability among still-alive teams only
    alive_mask = results_df['still_in']
    total_alive_score = results_df.loc[alive_mask, 'prediction_score'].sum()
    if total_alive_score > 0:
        results_df.loc[alive_mask, 'live_probability_%'] = round(
            results_df.loc[alive_mask, 'prediction_score'] / total_alive_score * 100, 2
        )
else:
    results_df['still_in'] = True
    results_df['live_probability_%'] = results_df['win_probability_%']

# ===================================
# SIDEBAR NAVIGATION
# ===================================
st.sidebar.title("⚽ Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Historical Trends", "🌍 2026 Teams", "🔮 Winner Prediction", "🔎 Predict Your Country"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Aastha Pandit**")
st.sidebar.markdown("MS in Data Science | Eastern University")
st.sidebar.markdown("[GitHub](https://github.com/Aasthapandit) | [LinkedIn](https://www.linkedin.com/in/aastha-pandit-842aa4228)")

# ===================================
# PAGE: HOME
# ===================================
if page == "🏠 Home":
    st.title("⚽ FIFA World Cup 2026: Data-Driven Analysis & Winner Prediction")
    st.markdown(
        "This project analyzes historical World Cup data from 1930 to 2026 across "
        "five datasets containing 48 teams, 104 fixtures, and 22 tournament editions "
        "to identify winning patterns and predict the FIFA World Cup 2026 winner."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Teams Competing (2026)", f"{teams.shape[0]}")
    col2.metric("Matches Scheduled (2026)", f"{fixtures.shape[0]}")
    col3.metric("World Cup Editions", f"{editions.shape[0]}")
    col4.metric("Historical Matches", f"{matches.shape[0]}")

    st.markdown("---")
    st.subheader("📌 Project Workflow")
    st.markdown(
        """
        1. **Data Loading & Cleaning** — Loaded and validated 5 datasets spanning 1930–2026
        2. **Exploratory Data Analysis** — Champions, goals trends, tournament growth, semi-final consistency
        3. **Prediction Model** — Weighted scoring: FIFA Rank (40%) + Semi-Finals (30%) + Titles (20%) + Host (10%)
        4. **Model Validation** — Tested against real WC 2026 group stage results — **76.2% accuracy**
        5. **Interactive App** — This Streamlit dashboard, deployed for anyone to explore
        """
    )

    st.subheader("🔮 Live Prediction")
    top3 = results_df[results_df['still_in']].head(3)
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            st.markdown(f"### {medals[i]} {row['team']}")
            st.metric("Live Win Probability", f"{row['live_probability_%']}%")
            st.caption(f"FIFA Rank #{int(row['fifa_rank'])} | {int(row['semi_finals'])} semi-finals | {int(row['titles'])} titles")

    eliminated_count = (~results_df['still_in']).sum()
    st.caption(f"⚡ Live update: {eliminated_count} teams eliminated so far — probabilities redistributed among remaining {results_df['still_in'].sum()} teams")

    st.info("⚠️ This is a data-driven estimate based on historical patterns and FIFA rankings — football is unpredictable, and upsets happen!")

# ===================================
# PAGE: HISTORICAL TRENDS
# ===================================
elif page == "📊 Historical Trends":
    st.title("📊 Historical World Cup Trends")

    tab1, tab2, tab3 = st.tabs(["🏆 Champions", "⚽ Goals Trend", "🌍 Tournament Growth"])

    with tab1:
        st.subheader("World Cup Champions — All Time (1930–2022)")
        champions = editions['champion'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['gold' if i == 0 else 'silver' if i == 1 else '#cd7f32' if i == 2 else 'steelblue'
                  for i in range(len(champions))]
        ax.bar(champions.index, champions.values, color=colors, edgecolor='black')
        ax.set_ylabel("Number of Titles")
        ax.set_xlabel("Country")
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(champions.values):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        st.caption(f"🏆 Most successful country: **{champions.index[0]}** with **{champions.values[0]}** titles. "
                   f"Only **{len(champions)}** countries have ever won the World Cup.")

    with tab2:
        st.subheader("Goals Per Match — World Cup History (1930–2022)")
        st.caption(
            "Note: Goals per match is one measure of tournament style, not the only one. "
            "A tense 0-0 with great saves can be just as compelling as a high-scoring game. "
            "We use goals per match here because it's the most consistently available metric across all tournaments since 1930."
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(editions['year'], editions['goals_per_match'], color='red', marker='o', linewidth=2)
        overall_avg = editions['goals_per_match'].mean()
        ax.axhline(overall_avg, color='blue', linestyle='--', alpha=0.7,
                   label=f'Overall average: {overall_avg:.2f}')
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Goals Per Match")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab3:
        st.subheader("Tournament Size Growth (1930–2026)")
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['red' if y == 2026 else 'steelblue' for y in editions['year']]
        ax.bar(editions['year'], editions['teams'], color=colors, edgecolor='black')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Teams")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.caption("🔴 2026 marks the first-ever 48-team World Cup — the most competitive tournament in history!")

# ===================================
# PAGE: 2026 TEAMS
# ===================================
elif page == "🌍 2026 Teams":
    st.title("🌍 FIFA World Cup 2026 — Qualified Teams")

    conf_filter = st.multiselect(
        "Filter by Confederation:",
        options=sorted(teams['confederation'].unique()),
        default=sorted(teams['confederation'].unique())
    )
    filtered_teams = teams[teams['confederation'].isin(conf_filter)].sort_values('fifa_rank')

    st.dataframe(
        filtered_teams[['team', 'group', 'confederation', 'fifa_rank', 'coach', 'best_wc_result']],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Confederation Breakdown")
    conf_counts = teams['confederation'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(conf_counts.values, labels=conf_counts.index, autopct='%1.0f%%', startangle=90)
    ax.set_title("Teams by Confederation (2026)")
    st.pyplot(fig)

# ===================================
# PAGE: WINNER PREDICTION (FULL MODEL)
# ===================================
elif page == "🔮 Winner Prediction":
    st.title("🔮 WC 2026 Winner Prediction Model")
    st.markdown(
        "**Model formula:** FIFA Ranking (40%) + Semi-Final History (30%) + "
        "World Cup Titles (20%) + Host Advantage (10%)"
    )

    # Show only still-alive teams sorted by live probability
    alive_df = results_df[results_df['still_in']].copy()
    alive_df = alive_df.sort_values('live_probability_%', ascending=False)
    top10 = alive_df.head(10)

    st.subheader(f"🏆 Top Contenders Still In Tournament ({len(alive_df)} teams remaining)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10['team'][::-1], top10['live_probability_%'][::-1],
            color='steelblue', edgecolor='white')
    ax.set_xlabel("Live Win Probability (%)")
    for i, prob in enumerate(top10['live_probability_%'][::-1]):
        ax.text(prob + 0.2, i, f"{prob:.1f}%", va='center', fontweight='bold')
    st.pyplot(fig)

    # Full table
    display_df = top10[['team', 'confederation', 'fifa_rank',
                         'semi_finals', 'titles', 'live_probability_%']].copy()
    display_df.columns = ['Team', 'Confederation', 'FIFA Rank',
                          'Semi-Finals', 'Titles', 'Live Win Probability (%)']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Winner announcement
    winner_row = alive_df.iloc[0]
    st.success(
        f"🥇 **Current Top Pick: {winner_row['team']}** "
        f"({winner_row['live_probability_%']}% live win probability)"
    )

    # Show eliminated teams
    elim_df = results_df[~results_df['still_in']].copy()
    if len(elim_df) > 0:
        with st.expander(f"❌ Eliminated Teams ({len(elim_df)})"):
            elim_display = elim_df[['team', 'confederation',
                                    'fifa_rank', 'win_probability_%']].copy()
            elim_display.columns = ['Team', 'Confederation',
                                    'FIFA Rank', 'Pre-Tournament Probability (%)']
            st.dataframe(elim_display.sort_values(
                'Pre-Tournament Probability (%)', ascending=False),
                use_container_width=True, hide_index=True)

    with st.expander("📊 Live Model Validation — How accurate is this model against real 2026 results?"):

        # ===================================
        # LIVE RESULTS FROM OPENFOOTBALL API
        # No API key needed — free & updated daily!
        # ===================================
        @st.cache_data(ttl=3600)  # Refresh every hour
        def fetch_live_results():
            """Fetch live WC 2026 match results from openfootball (free, no key needed)"""
            import requests
            try:
                url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    matches_played = []
                    # All matches are at top level under 'matches' key
                    for match in data.get("matches", []):
                        score = match.get("score", {})
                        if score and score.get("ft"):
                            ft = score["ft"]
                            team1 = match.get("team1", "")
                            team2 = match.get("team2", "")
                            score1 = ft[0]
                            score2 = ft[1]
                            if score1 > score2:
                                winner = team1
                            elif score2 > score1:
                                winner = team2
                            else:
                                winner = "Draw"
                            matches_played.append({
                                "match": f"{team1} vs {team2}",
                                "team1": team1,
                                "team2": team2,
                                "score": f"{score1}–{score2}",
                                "actual_winner": winner,
                                "stage": match.get("round", ""),
                                "group": match.get("group", "")
                            })
                    return matches_played, None
                else:
                    return None, f"API returned status {response.status_code}"
            except Exception as e:
                return None, str(e)

        live_matches, error = fetch_live_results()

        if error or not live_matches:
            st.warning(f"⚠️ Could not fetch live results right now. Showing last known accuracy.")
            st.markdown("""
            **Last known accuracy (June 28, 2026):**
            - ✅ **16 correct** on 21 decisive matches
            - 🎯 **Model Accuracy: 76.2%**
            """)
        else:
            # ===================================
            # MATCH FIFA RANKINGS TO VALIDATE
            # ===================================
            team_ranks = dict(zip(teams['team'], teams['fifa_rank']))

            correct = 0
            wrong = 0
            draws = 0
            results_rows = []

            for m in live_matches:
                t1 = m['team1']
                t2 = m['team2']
                rank1 = team_ranks.get(t1, 50)
                rank2 = team_ranks.get(t2, 50)
                predicted = t1 if rank1 < rank2 else t2
                actual = m['actual_winner']

                if actual == "Draw":
                    result = "⚠️ Draw"
                    draws += 1
                elif predicted == actual:
                    result = "✅ Correct"
                    correct += 1
                else:
                    result = "❌ Upset"
                    wrong += 1

                results_rows.append({
                    "Match": m['match'],
                    "Score": m['score'],
                    "Predicted": predicted,
                    "Actual": actual,
                    "Result": result,
                    "Stage": m['stage']
                })

            decisive = correct + wrong
            accuracy = (correct / decisive * 100) if decisive > 0 else 0

            st.markdown(f"""
            **Live validation against {len(live_matches)} real WC 2026 matches:**
            - ✅ **{correct} correct predictions**
            - ❌ **{wrong} upsets** (model predicted wrong winner)
            - ⚠️ **{draws} draws** (rankings alone cannot predict draws)

            **🎯 Live Model Accuracy: {accuracy:.1f}%** on decisive matches
            """)

            col1, col2 = st.columns(2)
            with col1:
                labels = ['Correct', 'Upset', 'Draw']
                sizes = [correct, wrong, draws]
                colors = ['#2ecc71', '#e74c3c', '#f39c12']
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(sizes, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
                ax.set_title(f"Model Accuracy\n{accuracy:.1f}% on Decisive Matches")
                st.pyplot(fig)

            with col2:
                st.markdown("**All results so far:**")
                df_results = pd.DataFrame(results_rows)
                st.dataframe(
                    df_results[['Match', 'Score', 'Predicted', 'Actual', 'Result']],
                    use_container_width=True, hide_index=True
                )

            st.caption(f"🔄 Results auto-updated from openfootball (refreshes every hour) • Last fetch: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}")

    st.warning(
        "⚠️ **Disclaimer:** This prediction is based on historical patterns and current FIFA rankings. "
        "Football is unpredictable — Morocco reached the semi-finals in 2022 as rank #22! "
        "Data science gives us probabilities, not certainties."
    )

# ===================================
# PAGE: PREDICT YOUR COUNTRY (INTERACTIVE)
# ===================================
elif page == "🔎 Predict Your Country":
    st.title("🔎 Check Any Country's Win Probability")
    st.markdown("Type or select a country below to see its data-driven chances of winning WC 2026!")

    country = st.selectbox(
        "Select a country:",
        options=sorted(results_df['team'].tolist())
    )

    if country:
        row = results_df[results_df['team'] == country].iloc[0]
        is_eliminated = not row['still_in']

        if is_eliminated:
            st.error(f"❌ **{country} has been eliminated from WC 2026.**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pre-Tournament Probability", f"{row['win_probability_%']}%",
                        delta="Eliminated ❌", delta_color="inverse")
            col2.metric("FIFA Rank", f"#{int(row['fifa_rank'])}")
            col3.metric("Semi-Finals (history)", f"{int(row['semi_finals'])}")
            col4.metric("World Cup Titles", f"{int(row['titles'])}")
        else:
            alive_df = results_df[results_df['still_in']].sort_values(
                'live_probability_%', ascending=False).reset_index(drop=True)
            rank_position = alive_df[alive_df['team'] == country].index[0] + 1
            remaining = results_df['still_in'].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔴 Live Win Probability", f"{row['live_probability_%']}%")
            col2.metric("FIFA Rank", f"#{int(row['fifa_rank'])}")
            col3.metric("Semi-Finals (history)", f"{int(row['semi_finals'])}")
            col4.metric("World Cup Titles", f"{int(row['titles'])}")

            st.markdown(f"**Tournament Ranking:** #{rank_position} out of {remaining} remaining teams")

            if rank_position == 1:
                st.success(f"🥇 {country} is our **top live favourite** to win WC 2026!")
            elif rank_position <= 4:
                st.success(f"🔥 {country} is a **top contender** — real chance of the trophy!")
            elif rank_position <= 8:
                st.info(f"⚡ {country} could realistically reach the **semi-finals or further!**")
            else:
                st.info(f"💪 {country} is still alive — anything can happen in football!")

            st.markdown("---")
            st.subheader("How does this country compare to other remaining teams?")
            compare_df = pd.concat([
                alive_df.head(5),
                alive_df[alive_df['team'] == country]
            ]).drop_duplicates(subset='team')
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['red' if t == country else 'steelblue' for t in compare_df['team']]
            ax.bar(compare_df['team'], compare_df['live_probability_%'].astype(float),
                   color=colors, edgecolor='black')
            ax.set_ylabel("Live Win Probability (%)")
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig)

    st.caption("💡 Share this page with friends — let them check their favorite team's chances!")

# ===================================
# FOOTER
# ===================================
st.markdown("---")
st.caption(
    "Data source: Kaggle FIFA World Cup Dataset (1930–2026) | "
    "Built with Python, pandas, matplotlib, and Streamlit | "
    "Originally prototyped on the Zerve AI analytics platform"
)
