"""Cricket Prediction Model
Original file is located at
"""


import pandas as pd
import os

# Load player info
players_df = pd.read_csv('players_info.csv')  # Must contain: player_id, player_name

# Files and merging instructions
files_info = [
    {'filename': 'merged_t20_batting_cleaned.csv', 'keys': ['batsman']},
    {'filename': 'merged_t20_bowling_cleaned.csv', 'keys': ['bowler_id']},
    {'filename': 'merged_t20_partnership_cleaned.csv', 'keys': ['player1', 'player2']},
    {'filename': 'merged_t20_fow_cleaned.csv', 'keys': ['batsman_id']},
    {'filename': 't20_matches_cleaned.csv', 'keys': []},

    {'filename': 'merged_odi_batting_cleaned.csv', 'keys': ['batsman']},
    {'filename': 'merged_odi_bowling_cleaned.csv', 'keys': ['bowler_id']},
    {'filename': 'merged_odi_partnership_cleaned.csv', 'keys': ['player1', 'player2']},
    {'filename': 'merged_odi_fow_cleaned.csv', 'keys': ['batsman_id']},
    {'filename': 'odi_matches_cleaned.csv', 'keys': []}
]

for file in files_info:
    fname = file['filename']
    keys = file['keys']
    df = pd.read_csv(fname)

    print(f"\n🔄 Processing: {fname}")

    if not keys:
        print(f"⚠️  Skipped: No player-related columns.")
        continue

    for key in keys:
        if key not in df.columns:
            print(f"❌ Column '{key}' not found in {fname}, skipping.")
            continue

        temp_df = players_df.rename(columns={
            'player_id': key,
            'player_name': f'{key}_name'
        })

        df = df.merge(temp_df, on=key, how='left')

        # Replace ID with name
        df[key] = df[f'{key}_name']
        df.drop(columns=[f'{key}_name'], inplace=True)
        print(f"✅ Merged and replaced: {key}")

    # Save result
    outname = f'merged_{fname}'
    df.to_csv(outname, index=False)
    print(f"📁 Saved: {outname}")


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("t20_matches_cleaned.csv")
df['team1_name'] = df['team1_name'].str.lower().str.strip()
df['team2_name'] = df['team2_name'].str.lower().str.strip()
df['match_winner'] = df['match_winner'].str.lower().str.strip()

def simulate_t20(team1, team2, n=1000):
    team1 = team1.lower().strip()
    team2 = team2.lower().strip()


    matches = df[((df['team1_name'] == team1) & (df['team2_name'] == team2)) |
                 ((df['team1_name'] == team2) & (df['team2_name'] == team1))]

    if matches.empty:
        print("❌ No match data found.")
        return

    win_counts = {team1: 0, team2: 0, "tie": 0}

    for _ in range(n):
        random_row = matches.sample(1).iloc[0]
        winner = random_row['match_winner']

        if winner == team1:
            win_counts[team1] += 1
        elif winner == team2:
            win_counts[team2] += 1
        else:
            win_counts["tie"] += 1

    plt.bar(win_counts.keys(), win_counts.values(), color=["blue", "green", "gray"])
    plt.title(f"{team1.title()} vs {team2.title()} – Simulated Outcomes ({n} runs)")
    plt.ylabel("Wins")
    plt.show()

    return win_counts

simulate_t20("india", "pakistan")

df = pd.read_csv("odi_matches_cleaned.csv")
df['team1_name'] = df['team1_name'].str.lower().str.strip()
df['team2_name'] = df['team2_name'].str.lower().str.strip()
df['match_winner'] = df['match_winner'].str.lower().str.strip()

def simulate_odi(team1, team2, n=1000):
    team1 = team1.lower().strip()
    team2 = team2.lower().strip()
    matches = df[((df['team1_name'] == team1) & (df['team2_name'] == team2)) |
                 ((df['team1_name'] == team2) & (df['team2_name'] == team1))]
    if matches.empty:
        return "No match data"

    win_counts = {team1: 0, team2: 0, "tie": 0}
    for _ in range(n):
        winner = matches.sample(1)['match_winner'].values[0]
        if winner in win_counts:
            win_counts[winner] += 1
        else:
            win_counts["tie"] += 1

    plt.bar(win_counts.keys(), win_counts.values(), color=["orange", "purple", "gray"])
    plt.title(f"{team1} vs {team2} Win Probabilities (ODI)")
    plt.show()
    return win_counts

simulate_odi("india", "australia")

df = pd.read_csv("merged_t20_batting_cleaned.csv")
top_batters = df.groupby("batsman")["runs"].sum().sort_values(ascending=False).head(10)
top_batters.plot(kind="bar", title="Top T20 Run Scorers", color="green")

df = pd.read_csv("merged_odi_batting_cleaned.csv")
top_batters = df.groupby("batsman")["runs"].sum().sort_values(ascending=False).head(10)
top_batters.plot(kind="bar", title="Top ODI Run Scorers", color="blue")

df = pd.read_csv("merged_t20_bowling_cleaned.csv")
top_bowlers = df.groupby("bowler_id")["wickets"].sum().sort_values(ascending=False).head(10)
top_bowlers.plot(kind="bar", title="Top T20 Wicket Takers", color="red")

df = pd.read_csv("merged_odi_bowling_cleaned.csv")
top_bowlers = df.groupby("bowler_id")["wickets"].sum().sort_values(ascending=False).head(10)
top_bowlers.plot(kind="bar", title="Top ODI Wicket Takers", color="orange")

df = pd.read_csv("merged_t20_partnership_cleaned.csv")
df['pair'] = df['player1'].astype(str).str.strip() + " & " + df['player2'].astype(str).str.strip()
top_pairs = df.groupby("pair")["partnership_runs"].sum().sort_values(ascending=False).head(10)
top_pairs.plot(kind="barh", title="Top T20 Partnerships", color="purple")

df = pd.read_csv("merged_odi_partnership_cleaned.csv")
df['pair'] = df['player1'].astype(str).str.strip() + " & " + df['player2'].astype(str).str.strip()
top_pairs = df.groupby("pair")["partnership_runs"].sum().sort_values(ascending=False).head(10)
top_pairs.plot(kind="barh", title="Top ODI Partnerships", color="blue")

df = pd.read_csv("t20_fow_cleaned.csv")
avg_fow = df.groupby("team")["runs"].mean().sort_values(ascending=False).head(10)
avg_fow.plot(kind="bar", title="T20 Fall of Wicket Average Score", color="brown")

df = pd.read_csv("odi_fow_cleaned.csv")
avg_fow = df.groupby("team")["runs"].mean().sort_values(ascending=False).head(10)
avg_fow.plot(kind="bar", title="ODI Fall of Wicket Average Score", color="teal")

import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = {
    "t20_batting": "merged_t20_batting_cleaned.csv",
    "t20_bowling": "merged_t20_bowling_cleaned.csv",
    "t20_fow": "merged_t20_fow_cleaned.csv",
    "t20_matches": "merged_t20_matches_cleaned.csv",
    "t20_partnership": "merged_t20_partnership_cleaned.csv",
    "odi_batting": "merged_odi_batting_cleaned.csv",
    "odi_bowling": "merged_odi_bowling_cleaned.csv",
    "odi_fow": "merged_odi_fow_cleaned.csv",
    "odi_matches": "merged_odi_matches_cleaned.csv",
    "odi_partnership": "merged_odi_partnership_cleaned.csv"
}

# Load all datasets
data = {key: pd.read_csv(path) for key, path in file_paths.items()}

# Utility Functions
def top_batsmen(df, format_name):
    top = df.groupby('batsman')['runs'].sum().sort_values(ascending=False).head(10)
    print(f"\n🏏 Top 10 Batters in {format_name}")
    print(top)
    return top

def top_bowlers(df, format_name):
    top = df.groupby('bowler_id')['wickets'].sum().sort_values(ascending=False).head(10)
    print(f"\n🎯 Top 10 Bowlers in {format_name}")
    print(top)
    return top

def top_partnerships(df, format_name):
    df['pair'] = df['player1'].astype(str).str.strip() + " & " + df['player2'].astype(str).str.strip()
    top = df.groupby('pair')['partnership_runs'].sum().sort_values(ascending=False).head(10)
    print(f"\n🤝 Top 10 Partnerships in {format_name}")
    print(top)
    return top

def fow_analysis(df, format_name):
    top = df.groupby('wicket')['runs'].mean().sort_index()
    print(f"\n📉 Average Runs at Fall of Each Wicket ({format_name})")
    print(top)
    return top

def team_wins(df, format_name):
    top = df['match_winner'].value_counts().head(10)
    print(f"\n🏆 Most Successful Teams in {format_name}")
    print(top)
    return top

# Run analysis for both formats
for format_type in ['t20', 'odi']:
    print(f"\n{'='*40}\n🔍 Analyzing {format_type.upper()} Cricket Data\n{'='*40}")
    top_batsmen(data[f'{format_type}_batting'], format_type.upper())
    top_bowlers(data[f'{format_type}_bowling'], format_type.upper())
    top_partnerships(data[f'{format_type}_partnership'], format_type.upper())
    fow_analysis(data[f'{format_type}_fow'], format_type.upper())
    team_wins(data[f'{format_type}_matches'], format_type.upper())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load batting data
df = pd.read_csv("merged_t20_batting_cleaned.csv")

# Basic preprocessing
df = df[df['runs'].notnull() & df['batsman'].notnull()]

# Set simulation parameters
num_simulations = 1000
overs = 20
balls_per_over = 6
max_balls = overs * balls_per_over

# Create a pool of historical ball-by-ball outcomes
ball_outcomes = df['runs'].values
# Optional: include 0 more often to reflect dot balls or add mode-wicket probability

# Simulate one innings
def simulate_innings():
    total_runs = 0
    wickets = 0
    for ball in range(max_balls):
        run = np.random.choice(ball_outcomes)
        total_runs += run
        # Simulate wicket with 5% probability per ball
        if random.random() < 0.05:
            wickets += 1
            if wickets >= 10:
                break
    return total_runs

# Run the simulation many times
simulated_scores = [simulate_innings() for _ in range(num_simulations)]

# Plot the result
plt.figure(figsize=(10, 5))
plt.hist(simulated_scores, bins=30, color='dodgerblue', edgecolor='black')
plt.title(f"Monte Carlo Simulation of T20 Innings (n={num_simulations})")
plt.xlabel("Total Runs Scored")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Summary stats
print("🏏 Monte Carlo Simulation Summary:")
print(f"  • Average score: {np.mean(simulated_scores):.2f}")
print(f"  • Median score : {np.median(simulated_scores)}")
print(f"  • Max score    : {np.max(simulated_scores)}")
print(f"  • Min score    : {np.min(simulated_scores)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load batting data
df = pd.read_csv("merged_odi_batting_cleaned.csv")

# Basic preprocessing
df = df[df['runs'].notnull() & df['batsman'].notnull()]

# Set simulation parameters
num_simulations = 1000
overs = 20
balls_per_over = 6
max_balls = overs * balls_per_over

# Create a pool of historical ball-by-ball outcomes
ball_outcomes = df['runs'].values
# Optional: include 0 more often to reflect dot balls or add mode-wicket probability

# Simulate one innings
def simulate_innings():
    total_runs = 0
    wickets = 0
    for ball in range(max_balls):
        run = np.random.choice(ball_outcomes)
        total_runs += run
        # Simulate wicket with 5% probability per ball
        if random.random() < 0.05:
            wickets += 1
            if wickets >= 10:
                break
    return total_runs

# Run the simulation many times
simulated_scores = [simulate_innings() for _ in range(num_simulations)]

# Plot the result
plt.figure(figsize=(10, 5))
plt.hist(simulated_scores, bins=30, color='dodgerblue', edgecolor='black')
plt.title(f"Monte Carlo Simulation of ODI Innings (n={num_simulations})")
plt.xlabel("Total Runs Scored")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Summary stats
print("🏏 Monte Carlo Simulation Summary:")
print(f"  • Average score: {np.mean(simulated_scores):.2f}")
print(f"  • Median score : {np.median(simulated_scores)}")
print(f"  • Max score    : {np.max(simulated_scores)}")
print(f"  • Min score    : {np.min(simulated_scores)}")

import pandas as pd
import numpy as np
import random

# Load data (replace path as needed)
df = pd.read_csv("merged_t20_batting_cleaned.csv")

# Clean
df = df[df['runs'].notnull() & df['batsman'].notnull()]

# SIMULATION FUNCTION
def simulate_match(openers, players, overs=20, simulations=1000):
    all_scores = []

    # Filter data for selected players only
    player_df = df[df['batsman'].isin(players)]

    if player_df.empty:
        print("⚠️ No valid data for selected players.")
        return

    ball_outcomes = player_df['runs'].values

    for _ in range(simulations):
        total_runs = 0
        wickets = 0
        for ball in range(overs * 6):
            run = np.random.choice(ball_outcomes)
            total_runs += run

            if random.random() < 0.05:
                wickets += 1
                if wickets >= 10:
                    break
        all_scores.append(total_runs)

    # Results
    avg_score = np.mean(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)
    print(f"🧪 Simulated {simulations} matches:")
    print(f"➡️ Avg Score: {avg_score:.2f} | Min: {min_score} | Max: {max_score}")

openers = [48405, 49752]
team = [48405, 49752, 61990, 70633,
        7593, 49247, 70640, 61325,
        60530, 95316, 54282]

simulate_match(openers=openers, players=team, overs=20, simulations=1000)

openers = [61325, 54282 ]
team = [48405, 49752, 61990, 70633,
        7593, 49247, 70640, 61325,
        60530, 95316, 54282]

simulate_match(openers=openers, players=team, overs=20, simulations=1000)

openers = ['Jasprit Bumrah', 'Josh Hazelwood' ]
team = ['Rohit Sharma', 'Virat Kohli', 'KL Rahul', 'Hardik Pandya', 'Jasprit Bumrah', 'Sachin Tendulkar','MS Dhoni', 'Virendra Sehwag', 'Bhuveneshwar Kumar', 'Josh Hazelwood', 'Ravindra Jadeja']

simulate_match(openers=openers, players=team, overs=20, simulations=1000)
