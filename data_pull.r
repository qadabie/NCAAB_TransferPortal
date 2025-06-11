#!/usr/bin/env Rscript
library(hoopR)
library(tidyverse)
library(janitor)
library(zoo)
library(nanoparquet)
output_file <- "24_25_mbb_pbp.parquet"


player_logs <- load_mbb_player_box(season = most_recent_mbb_season()) %>%
  clean_names() %>%
  rename(player_name = athlete_display_name,
         player_id = athlete_id) %>%
  mutate(across(c(player_id, team_id),
                ~ if (!is.numeric(.)) as.numeric(.) else .))
# Function to load play-by-play data for each game
function_pbp <- function(x) {
  load_mbb_pbp(game_id = x) %>%
    dplyr::mutate(game_id = x)
}

games <- player_logs |>
  mutate(game_date = as_date(game_date)) |>
  distinct(game_id) |>
  pull(game_id)

# Take only the first game from the games vector for testing
test_game <- games[1]
cat("Testing with game ID:", test_game, "\n")

# Uncomment the next line when ready to process all games
# pbp_month <- map_df(games, function_pbp)

# For now, process just the first game
pbp_month <- function_pbp(test_game)
# Print column names of pbp_month to inspect the structure
# cat("Columns in pbp_month:\n")
# print(colnames(pbp_month))
#pbp_month <- map_df(games, function_pbp)
mbb_pbp_raw <- pbp_month %>%
  mutate(across(c(athlete_id_1, athlete_id_2),
                ~ if (!is.numeric(.)) as.numeric(.) else .))

# Create more consistent variable naming to match the original code structure
mbb_pbp_raw <- mbb_pbp_raw %>%
  rename(player1_id = athlete_id_1,
         player2_id = athlete_id_2) %>%
  mutate(event_type = type_text,
         event_action_type = type_id,
         description = text)
# player_logs <- player_logs %>%
#   mutate(across(c(player_id, team_id), as.numeric))

# # Add debugging prints to identify which join might be causing issues
# cat("Starting left_join operations...\n")

# # First join
# print("Checking player1_id join data:")
# print(head(player_logs %>% distinct(player1_id = player_id)))

# mbb_pbp <- mbb_pbp_raw %>%
#   left_join(player_logs %>%
#               distinct(player1_id = player_id)) 
# cat("After first join - row count:", nrow(mbb_pbp), "\n")

# # Second join
# print("Checking player2_id join data:")
# print(head(player_logs %>% distinct(player2_id = player_id)))

# mbb_pbp <- mbb_pbp %>%
#   left_join(player_logs %>%
#               distinct(player2_id = player_id))
# cat("After second join - row count:", nrow(mbb_pbp), "\n")

# # Third join
# # print("Checking team_id join data:")
# # print(head(player_logs %>% distinct(team_id = team_uid)))
# # print("Column names in mbb_pbp:")
# # print(names(mbb_pbp))

# mbb_pbp <- mbb_pbp %>%
#   left_join(player_logs %>%
#               distinct(team_id = team_id))
# cat("After third join - row count:", nrow(mbb_pbp), "\n")

# # Continue with select
# mbb_pbp <- mbb_pbp %>%
#   select(game_id, half, time, number_event = id, msg_type = event_type, 
#          locX = coordinate_x, locY = coordinate_y, home_team_abbrev, description)
# cat("After select - row count:", nrow(mbb_pbp), "\n")

# # Fourth join - the complex one
# print("Checking team join data:")
# print(head(player_logs %>%
#               distinct(game_id = as.integer(game_id), home_team_abbrev =team_slug, team_location)))

# mbb_pbp <- mbb_pbp %>%
#   left_join(player_logs %>%
#               distinct(game_id = as.integer(game_id), home_team_abbrev = team_slug, team_location) %>%
#               pivot_wider(names_from = team_location,
#                           values_from = home_team_abbrev,
#                           names_prefix = "team_"))
# cat("After fourth join - row count:", nrow(mbb_pbp), "\n")

# # Continue with the rest of the transformations
# mbb_pbp <- mbb_pbp %>%
#   separate(time, into = c("min_remain", "sec_remain"), sep = ":") %>%
#   mutate(min_remain = as.numeric(min_remain),
#        sec_remain = as.numeric(sec_remain),
#        secs_left_qtr = (min_remain * 60) + sec_remain) %>%
#   mutate(number_event = row_number()) %>%  
#   mutate(shot_pts = case_when(
#     msg_type == "MADE_FG" & str_detect(description, "3PT") ~ 3,
#     msg_type == "MADE_FG" ~ 2,
#     msg_type == "MADE_FT" ~ 1,
#     TRUE ~ 0
#   )) %>%
#   group_by(game_id) %>%
#   ungroup() %>%
#   arrange(game_id, number_event)

# cat("Final data frame row count:", nrow(mbb_pbp), "\n")

# Write the final output
mbb_pbp_raw |> 
  write_parquet(
    str_glue(output_file)
  )
  # Write player_logs to a parquet file
player_logs |> 
  write_parquet("24_25mbb_gamelog.parquet")

cat("Player logs saved to 24_25mbb_gamelog.parquet\n")