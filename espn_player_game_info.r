library(arrow)
library(dplyr)
library(hoopR)  

# Load player box scores for multiple seasons
seasons <- c(2022)
all_box_scores <- list()

for (season in seasons) {
    cat("Loading player box scores for season:", season, "\n")
    
    tryCatch({
        # Load data for the current season
        season_data <- load_mbb_player_box(season)
        
        # Select only the columns we need
        if (!is.null(season_data) && nrow(season_data) > 0) {
            filtered_data <- season_data %>%
                select(game_id, season, game_date, athlete_id, athlete_display_name, 
                             team_id, team_display_name, minutes, starter, active)
            
            all_box_scores[[length(all_box_scores) + 1]] <- filtered_data
        }
    }, error = function(e) {
        cat("Error loading season:", season, "-", conditionMessage(e), "\n")
    })
}

# Combine all the data
combined_box_scores <- bind_rows(all_box_scores)

# Check if the parquet file already exists
if (file.exists("player_box_scores.parquet")) {
    # Read existing data
    existing_data <- read_parquet("player_box_scores.parquet")
    
    # Combine with new data
    combined_box_scores <- bind_rows(existing_data, combined_box_scores)
    
    cat("Appended new data to existing parquet file\n")
}

# Save as parquet file
write_parquet(combined_box_scores, "player_box_scores.parquet")

# Save first 100 rows as CSV
# combined_box_scores %>%
#     head(100) %>%
#     write.csv("player_box_scores_sample.csv", row.names = FALSE)

cat("Processing complete. Data saved to player_box_scores.parquet and player_box_scores_sample.csv\n")