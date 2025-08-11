# Install required packages if not already installed
if (!requireNamespace("hoopR", quietly = TRUE)) {
    install.packages("hoopR")
}
if (!requireNamespace("arrow", quietly = TRUE)) {
    install.packages("arrow")
}

# Load libraries
library(hoopR)
library(arrow)
library(dplyr)

# Load mbb schedule data
mbb_player_box <- load_mbb_player_box()

# Save the entire dataframe as a parquet file
write_parquet(mbb_player_box, "mbb_player_box_full.parquet")

# Save the first 10,000 rows as a CSV file
mbb_player_box_subset <- head(mbb_player_box, 10000)
write.csv(mbb_player_box_subset, "mbb_player_box_10000.csv", row.names = FALSE)

# Print confirmation
cat("NBA schedule data has been saved:\n")
cat("- Full dataset as mbb_player_box_full.parquet\n")
cat("- First 10,000 rows as mbb_player_box_10000.csv\n")