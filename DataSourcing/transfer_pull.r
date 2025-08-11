#' Identify transfers in a team roster
#'
#' @param team Team name as used in KenPom
#' @param year Year to check for transfers
#' @return A data frame with players from the specified year marked as transfers
#' @importFrom hoopR kp_team_players
#' @importFrom dplyr filter mutate left_join
# Load required packages
# if (!requireNamespace('pacman', quietly = TRUE)){
#   install.packages('pacman')
# }
# pacman::p_load_current_gh("sportsdataverse/hoopR", dependencies = TRUE, update = TRUE)
library(hoopR)
library(dplyr)
library(tibble)

# Set KenPom credentials
# login(
#   user_email = Sys.getenv("KP_USER"),
#   user_pw = Sys.getenv("KP_PW")
# )
# Verify login is successful
if (!has_kp_user_and_pw()) {
  stop("KenPom credentials not set or invalid. Please check your credentials.")
}

identify_transfers <- function(team, year) {
  # Load required packages
  if (!requireNamespace("hoopR", quietly = TRUE)) {
    stop("Package 'hoopR' is needed for this function. Please install it.")
  }
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is needed for this function. Please install it.")
  }
  
  # Step 1: Pull rosters for current and previous year
  current_roster <- tibble(kp_team_player_stats(team = team, year = year)$all_games)
  previous_roster <- tibble(kp_team_player_stats(team = team, year = year - 1)$all_games)
  # Print column names of current_roster
  #cat("Current roster columns:", paste(names(current_roster), collapse=", "), "\n")
  # Step 2: Filter out freshmen from current year
  # Convert rosters to data frames if they aren't already
  
  current_roster <- as.data.frame(current_roster)
  previous_roster <- as.data.frame(previous_roster)
  # Calculate stat differences for players that appear in both rosters
  player_comparisons <- current_roster %>%
    left_join(previous_roster, by = "player_id", suffix = c("_current", "_previous")) %>%
    mutate(
      min_pct_diff = min_pct_current - min_pct_previous,
      o_rtg_diff = o_rtg_current - o_rtg_previous,
      poss_pct_diff = poss_pct_current - poss_pct_previous,
      shots_pct_diff = shots_pct_current - shots_pct_previous,
      e_fg_pct_diff = e_fg_pct_current - e_fg_pct_previous,
      ts_pct_diff = ts_pct_current - ts_pct_previous,
      or_pct_diff = or_pct_current - or_pct_previous,
      dr_pct_diff = dr_pct_current - dr_pct_previous,
      a_rate_diff = a_rate_current - a_rate_previous,
      to_rate_diff = to_rate_current - to_rate_previous,
      blk_pct_diff = blk_pct_current - blk_pct_previous,
      stl_pct_diff = stl_pct_current - stl_pct_previous,
      f_cper40_diff = f_cper40_current - f_cper40_previous,
      f_dper40_diff = f_dper40_current - f_dper40_previous,
      ft_rate_diff = ft_rate_current - ft_rate_previous,
      ft_pct_diff = ft_pct_current - ft_pct_previous,
      fg_2_pct_diff = fg_2_pct_current - fg_2_pct_previous,
      fg_3_pct_diff = fg_3_pct_current - fg_3_pct_previous
    ) %>%
    select(
      number = number_current,
      player = player_current,
      ht = ht_current,
      wt = wt_current,
      yr = yr_current,
      g = g_current,
      team = team_current,
      year = year_current,
      player_id,
      min_pct_diff,
      o_rtg_diff,
      poss_pct_diff,
      shots_pct_diff,
      e_fg_pct_diff,
      ts_pct_diff,
      or_pct_diff,
      dr_pct_diff,
      a_rate_diff,
      to_rate_diff,
      blk_pct_diff,
      stl_pct_diff,
      f_cper40_diff,
      f_dper40_diff,
      ft_rate_diff,
      ft_pct_diff,
      fg_2_pct_diff,
      fg_3_pct_diff
    )
  # Now filter out freshmen from current year

  non_freshmen <- current_roster %>%
    dplyr::filter(current_roster$yr != "Fr")
  # Step 3 & 4: Compare with previous year's roster and flag transfers
  # Create a list of players from previous year
  previous_players <- previous_roster$player_id
  returning_players <- non_freshmen$player_id

  # Find potential transfers (players who are not freshmen but weren't on last year's roster)
  potential_transfers <- returning_players[!returning_players %in% previous_players]
  cat("Potential transfers:", paste(potential_transfers, collapse=", "), "\n")
  # Identify players who are in the current roster but not in the previous year's roster

  # Mark players as transfers if they're not in previous year's roster
  transfers <- player_comparisons %>%
    dplyr::mutate(is_transfer = (.data$player_id %in% potential_transfers))
  
  transfers
}

# Example usage:
# Wrap in try-catch to handle potential authentication errors
all_teams <- kp_efficiency(min_year = 2022, max_year = 2022)$team
# Create empty list to store results
all_transfers <- list()

# Loop through each team
for(team_name in all_teams) {
  cat("Processing team:", team_name, "\n")
  
  # Try to identify transfers, continue to next team if an error occurs
  tryCatch({
    team_transfers <- identify_transfers(team = team_name, year = 2022)
    all_transfers[[team_name]] <- team_transfers
    cat("Found", sum(team_transfers$is_transfer), "transfers for", team_name, "\n")
  }, error = function(e) {
    cat("Error processing", team_name, ":", conditionMessage(e), "\n")
  })
  
  # Add small delay to avoid overloading the API
  Sys.sleep(1)
}

# Combine all results into one data frame
combined_transfers <- dplyr::bind_rows(all_transfers)

# Install arrow package if not already installed
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}

# Save the combined results as a parquet file
arrow::write_parquet(combined_transfers, "ncaab_transfers_2022.parquet")
cat("Transfer data saved to ncaab_transfers_2022.parquet\n")