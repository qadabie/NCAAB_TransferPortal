from db_connect import get_connection
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import pyarrow
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import ruptures as rpt
from sklearn.ensemble import RandomForestRegressor
from Convert_Team_Names import convert_team_name

conn = get_connection()
df = pd.read_parquet('merged_jaccard_similarity.parquet')
df25 = pd.read_parquet('merged_jaccard_similarity_2025.parquet')
df_regular_season = pd.read_csv('last_regular_season_game.csv')
team_table = pd.read_sql('SELECT * FROM team_table', conn)

# Create a dictionary mapping team_id to team_name from team_table
team_table['team_name'] = convert_team_name(team_table['team'], 'espn', 'kenpom')
team_id_to_name = dict(zip(team_table['team_id'], team_table['team_name']))

# Create a dictionary for last game data
last_game_dict = {}
for _, row in df_regular_season.iterrows():
    key = (row['team'], row['year'])
    last_game_dict[key] = row['last_game_before_march_2nd_week']
# Combine the two DataFrames
df = pd.concat([df, df25], ignore_index=True)
df['team_name'] = df['team_id'].map(team_id_to_name)
df['last_regular_season_game'] = df.apply(lambda row: last_game_dict.get((row['team_name'], row['season'])), axis=1)
df.dropna(subset=['last_regular_season_game'], inplace=True)
pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
print(df['season'].value_counts())

# Flatten all values from the lists in the DataFrame
all_jaccard = [val for sublist in df['jaccard_similarities'] for val in sublist]
all_def_diff = [val for sublist in df['def_rating_diffs'] for val in sublist]
all_def_to_pct_diff = [val for sublist in df['def_to_diffs'] for val in sublist]
all_def_apl_diff = [val for sublist in df['def_apl_diffs'] for val in sublist]
all_def_ft_rate_diff = [val for sublist in df['def_ft_rate_diffs'] for val in sublist]
all_home_diff = [val for sublist in df['home_diffs'] for val in sublist]
# Remove pairs where any value is nan
x_vals = []
y_vals = []
for x1, x2, x3, x4, x5, y in zip(all_def_diff, all_def_to_pct_diff, all_def_apl_diff, all_def_ft_rate_diff, all_home_diff, all_jaccard):
    if not (np.isnan(x1) or np.isnan(x2) or np.isnan(x3) or np.isnan(x4) or np.isnan(y)):
        x_vals.append([x1, x2, x3, x4, x5])
        y_vals.append(y)
x_arr = np.array(x_vals)
y_arr = np.array(y_vals)

# Scale the x values
scaler = QuantileTransformer(output_distribution='normal', random_state=0)
x_arr = scaler.fit_transform(x_arr)

# Fit a linear regression model
reg = RandomForestRegressor(random_state=0, n_estimators=100)
reg.fit(x_arr, y_arr)
y_pred = reg.predict(x_arr)
residuals = np.array(all_jaccard) - y_pred
# Add residuals to the DataFrame
# Split the flat residuals array back into the original row-wise structure
residuals_split = []
idx = 0
for sim_list in df['jaccard_similarities']:
    n = len(sim_list)
    residuals_split.append((residuals[idx:idx+n] * 100))
    idx += n
df['residuals'] = residuals_split
# Take a sample row (first row) and get its residuals
sample_idx = 0
sample_residuals = np.array(df.iloc[sample_idx]['residuals'])
# Fine-tune sigma and pen parameters here
pen_model = "l2"  # Use "l2" for L2 norm, "rbf" for RBF kernel
sigma = 1  # Adjust this value for gaussian_filter1d smoothing
penalty = 3  # Adjust this value for change point detection

df['smoothed'] = df['residuals'].apply(lambda res: gaussian_filter1d(np.array(res), sigma=sigma))

def create_season_plot(team_id, season, df=df):
    # Filter the DataFrame for the selected team and season
    sample_row = df[(df['team_id'] == team_id) & (df['season'] == season)].iloc[0]
    residuals = np.array(sample_row['residuals'])
    smoothed = np.array(sample_row['smoothed'])
    change_points = np.array(sample_row['change_points'])
    # Calculate and plot average smoothed value for each segment between change points

    # Plot residuals, smoothed residuals, and change points
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Residuals', alpha=0.5)
    plt.plot(smoothed, label='Smoothed Residuals', linewidth=2)
    for cp in change_points:
        plt.axvline(cp, color='red', linestyle='--', alpha=0.7, label='Change Point' if cp == change_points[0] else "")
    prev_cp = 0
    for i, cp in enumerate(change_points):
        segment = smoothed[prev_cp:cp]
        if len(segment) > 0:
            avg_val = np.mean(segment)
            plt.hlines(avg_val, prev_cp, cp, colors='green', linestyles='-', linewidth=2, alpha=0.7,
                    label='Segment Avg' if i == 0 else "")
            # Annotate the average value in the middle of the segment
            mid = prev_cp + (cp - prev_cp) // 2
            plt.text(mid, avg_val, f"{avg_val:.2f}", color='green', fontsize=9, va='bottom', ha='center')
        prev_cp = cp
    plt.axvline(x=27, color='black', linestyle='--')
    plt.text(27.2, plt.gca().get_ylim()[1]*0.95, "postseason play", color='black', rotation=90, va='top')
    plt.axvline(x=13, color='blue', linestyle='--')
    plt.text(13.2, plt.gca().get_ylim()[1]*0.95, "conference play", color='blue', rotation=90, va='top')
    plt.title(f"Season Analysis for Team Alabama, Season {season}: Residuals, Smoothed, and Change Points")
    plt.xlabel('Game Index')
    plt.ylabel('Residual')
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_animated_plot(team_id, season):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    row_sample = df[(df['team_id'] == team_id) & (df['season'] == season)].iloc[0]
    raw_signal = np.array(row_sample['jaccard_similarities'])*100
    adjusted_signal = np.array(row_sample['residuals'])
    smooth_1 = np.array(row_sample['smoothed'])
    smooth_half = gaussian_filter1d(np.array(row_sample['residuals']), sigma=0.5)
    smooth_double = gaussian_filter1d(np.array(row_sample['residuals']), sigma=2.0)
    x = np.arange(len(raw_signal))
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())

    raw_line, = ax.plot([], [], label='Raw Signal', color='blue', alpha=0.6)
    adj_line, = ax.plot([], [], label='Adjusted Signal', color='orange', alpha=0.8)
    smooth_line, = ax.plot([], [], label='Smoothed Signal (sigma=1)', color='green', linewidth=2)
    smooth_half_line, = ax.plot([], [], label='Smooth (sigma=0.5)', color='purple', linestyle='--')
    smooth_double_line, = ax.plot([], [], label='Smooth (sigma=2.0)', color='red', linestyle=':')

    # Hide all lines initially
    raw_line.set_visible(False)
    adj_line.set_visible(False)
    smooth_line.set_visible(False)
    smooth_half_line.set_visible(False)
    smooth_double_line.set_visible(False)
    # Add arrows and labels instead of legend
    # Store annotations for later use
    annotations = []

    def add_annotations():
        ann1 = ax.annotate('Raw Signal', xy=(len(raw_signal)*0.8, raw_signal[int(len(raw_signal)*0.8)]), 
                    xytext=(len(raw_signal)*0.75, raw_signal[int(len(raw_signal)*0.8)] - 15),
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                    fontsize=10, color='#2E86AB', weight='bold', visible=False)
        
        ann2 = ax.annotate('Adjusted Signal', xy=(len(adjusted_signal)*0.75, adjusted_signal[int(len(adjusted_signal)*0.75)]), 
                    xytext=(len(adjusted_signal)*0.7, adjusted_signal[int(len(adjusted_signal)*0.75)] + 12),
                    arrowprops=dict(arrowstyle='->', color='#F24236', lw=1.5),
                    fontsize=10, color='#F24236', weight='bold', visible=False)
        
        ann3 = ax.annotate('Smoothed (σ=1)', xy=(len(smooth_1)*0.6, smooth_1[int(len(smooth_1)*0.6)]), 
                    xytext=(len(smooth_1)*0.55, smooth_1[int(len(smooth_1)*0.6)] + 8),
                    arrowprops=dict(arrowstyle='->', color='#F6AE2D', lw=1.5),
                    fontsize=10, color='#F6AE2D', weight='bold', visible=False)
        
        ann4 = ax.annotate('Smoothed (σ=0.5)', xy=(len(smooth_half)*0.4, smooth_half[int(len(smooth_half)*0.4)]), 
                    xytext=(len(smooth_half)*0.35, smooth_half[int(len(smooth_half)*0.4)] + 5),
                    arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                    fontsize=10, color='#A23B72', weight='bold', visible=False)
        
        ann5 = ax.annotate('Smoothed (σ=2.0)', xy=(len(smooth_double)*0.2, smooth_double[int(len(smooth_double)*0.2)]), 
                    xytext=(len(smooth_double)*0.15, smooth_double[int(len(smooth_double)*0.2)] + 2),
                    arrowprops=dict(arrowstyle='->', color='#2F9B69', lw=1.5),
                    fontsize=10, color='#2F9B69', weight='bold', visible=False)
        
        return [ann1, ann2, ann3, ann4, ann5]

    annotations = add_annotations()
    
    def update(frame):
        if frame <= 30:
            raw_line.set_data(x, raw_signal)
            raw_line.set_visible(True)
            annotations[0].set_visible(True)  # Show raw signal annotation
        elif frame <= 40:
            raw_line.set_data(x, raw_signal)
            adj_line.set_data(x, adjusted_signal)
            raw_line.set_visible(True)
            adj_line.set_visible(True)
            annotations[0].set_visible(True)  # Keep raw signal annotation
            annotations[1].set_visible(True)  # Show adjusted signal annotation
        else:
            raw_line.set_data(x, raw_signal)
            adj_line.set_data(x, adjusted_signal)
            smooth_line.set_data(x[:frame-40+1], smooth_1[:frame-40+1])
            smooth_half_line.set_data(x[:frame-40+1], smooth_half[:frame-40+1])
            smooth_double_line.set_data(x[:frame-40+1], smooth_double[:frame-40+1])
            raw_line.set_visible(True)
            adj_line.set_visible(True)
            smooth_line.set_visible(True)
            smooth_half_line.set_visible(True)
            smooth_double_line.set_visible(True)
            # Show all annotations when smoothed lines appear
            for ann in annotations:
                ann.set_visible(True)

        if frame <= 30:
            # Zoom out to show full range
            ax.set_ylim(-10, 100)
        elif 30 <= frame <= 40:
            # Interpolate ymax from 100 to 20 over 30 frames
            progress = (frame - 30) / (40 - 30)
            ymax = 100 - progress * (100 - 20)
            ax.set_ylim(-10, ymax)
        elif frame > 40:
            ax.set_ylim(-10, 20)  # Stay zoomed in

        return raw_line, adj_line, smooth_line, smooth_half_line, smooth_double_line
    ax.set_title("Signal Processing Animation (Alabama 2024)")
    ax.set_xlabel("Season Game Index")
    ax.set_ylabel("Similarity Value")
    ax.set_facecolor('#f5f5f5')  # Light grey background
    ax.grid(True, alpha=0.3, color='white', linewidth=1.5)
    
    # Update colors for better contrast on grey background
    raw_line.set_color('#2E86AB')  # Blue
    adj_line.set_color('#F24236')  # Red-orange
    smooth_line.set_color('#F6AE2D')  # Yellow-orange
    smooth_half_line.set_color('#A23B72')  # Purple
    smooth_double_line.set_color('#2F9B69')  # Green

    ani = animation.FuncAnimation(fig, update, frames=96, interval=200, blit=False, repeat=True)

    # To save as a file (optional)
    ani.save("signal_processing.gif", writer='pillow', dpi=400)

    plt.show()


# Apply change detection to all smoothed series and record adjust time and EoS similarity
adjust_times = []
eos_similarities = []
segments = []
print(df['season'].value_counts())
# Initialize list to store results for different penalty values
all_results = []

for pen_value in range(1, 6):  # penalty values from 1 to 5
    pen_value  =pen_value*.5
    adjust_times = []
    eos_similarities = []
    segments = []
    change_point_list = []
    for idx, row in df.iterrows():
        smoothed = np.array(row['smoothed'])
        if len(smoothed) >= 8:  # min_size * 2
            model = rpt.Pelt(model=pen_model, jump=1, min_size=3).fit(smoothed)
            change_points = model.predict(pen=pen_value)
            last_game = int(row['last_regular_season_game']) 
            eos_similarity = smoothed[last_game - 7] if last_game is not None and last_game - 7 < len(smoothed) else np.nan
            # Find the start of the last change point segment
            # Find the change point segment that includes the last_game
            if len(change_points) > 1 and last_game is not None:
                # Find which segment the last_game falls into
                segment_start = 0
                for cp in change_points[:-1]:  # Exclude the last point (end of series)
                    if last_game-7 < cp:
                        # last_game is in the current segment
                        adjust_time = segment_start / (last_game-7)
                        break
                    segment_start = cp
      
            # Append the results
    
        adjust_times.append(adjust_time)
        eos_similarities.append(eos_similarity)
        segments.append(len(change_points))
        change_point_list.append(change_points)
    
    # Store results for this penalty value
    temp_df = df.copy()
    temp_df['adjust_time'] = adjust_times
    temp_df['EoS_similarity'] = eos_similarities
    temp_df['segments'] = segments
    temp_df['penalty'] = pen_value
    all_results.append(temp_df)
    
    # Save values to main df when penalty is 1.5
    if pen_value == 1.5:
        df['adjust_time'] = adjust_times
        df['EoS_similarity'] = eos_similarities
        df['segments'] = segments
        df['change_points'] = change_point_list
    
    print(f"Penalty {pen_value}:")
    print(f"Number of values = 1: {sum(1 for x in adjust_times if x == 1)}")
    avg_smoothed_length = temp_df['smoothed'].apply(len).mean()
    avg_segments = temp_df['segments'].mean()
    result = avg_smoothed_length / avg_segments
    print(f"Average smoothed length: {avg_smoothed_length:.2f}")
    print(f"Average segments: {avg_segments:.2f}")
    print(f"Average smoothed length / Average segments: {result:.2f}")
    print()

# Concatenate all results
df_all_penalties = pd.concat(all_results, ignore_index=True)

# Create vertically stacked histograms
fig, axes = plt.subplots(5, 1, figsize=(8, 15))
fig.suptitle('Histogram of Adjust Time by Penalty Value', fontsize=16)

for i, pen_value in enumerate(range(1, 6)):
    pen_value = pen_value * 0.5  # Convert to the actual penalty value
    penalty_data = df_all_penalties[df_all_penalties['penalty'] == pen_value]
    penalty_data['adjust_time'].dropna().astype(float).hist(
        bins=12, color='skyblue', edgecolor='black', ax=axes[i]
    )
    axes[i].set_title(f'Penalty = {pen_value}')
    axes[i].set_xlabel('Adjust Time (Game Index)')
    axes[i].set_ylabel('Frequency')

plt.show()

# # Save the summary DataFrame as a CSV
df.to_csv('team_adjust_times_and_eos_similarity.csv', index=False)

# Set team_id and season for the sample plot
# Teams of interest: 333 Alabama, 228 Clemson, 41 UCONN, 2628 TCU, 
# team_id = 248  # Change this to the desired team_id
# season = 2024  # Change this to the desired season
create_season_plot(333, season = 2024)
# create_animated_plot(333, season = 2024)