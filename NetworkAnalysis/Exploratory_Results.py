#Exploratory Results
import pandas as pd

df = pd.read_csv('merged_conf.csv')
yearly_avg = df.groupby('year')[['min_sum_tr', 'min_sum_fr', 'min_sum_new']].mean()
yearly_avg = yearly_avg / 5
yearly_avg.columns = ['Transfer %', 'Freshman %', 'New %']
print(yearly_avg)

postseason_avg = df[df['POSTSEASON'] >= 4].groupby('year')['min_sum_tr'].mean()
postseason_avg = postseason_avg / 5
print("\nAverage min_sum_tr for teams with POSTSEASON >= 3:")
print(postseason_avg)

# Add average row
print(f"\nOverall average: {postseason_avg.mean():.4f}")