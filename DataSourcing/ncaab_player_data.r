# Load required libraries
library(hoopR)
library(tidyverse)
library(janitor)
library(DBI)
library(RPostgres)
library(dotenv)

dotenv::load_dot_env(file = ".env")

pg_host <- Sys.getenv("PG_HOST")
pg_port <- Sys.getenv("PG_PORT")
pg_db <- Sys.getenv("PG_DATABASE")
pg_user <- Sys.getenv("PG_USER")
pg_password <- Sys.getenv("PG_PASSWORD")
pg_sslmode <- Sys.getenv("PG_SSLMODE")

# Connect to the Postgres database
con <- dbConnect(
  Postgres(),
  dbname = pg_db,
  host = pg_host,
  port = as.integer(pg_port),
  user = pg_user,
  password = pg_password,
  sslmode = ifelse(pg_sslmode != "", pg_sslmode, NULL)
)

seasons <- 2020:2023

# Pull and combine player box logs for each season
player_logs <- map_df(seasons, function(season) {
  cat("Loading season:", season, "\n")
  load_mbb_player_box(season = season)
}) %>%
  clean_names() %>%
  rename(
    player_name = athlete_display_name,
    player_id = athlete_id
  ) %>%
  mutate(across(c(player_id, team_id), ~ if (!is.numeric(.)) as.numeric(.) else .))

player_logs <- player_logs %>% as_tibble()

# Write to DB
dbWriteTable(
  con,
  name = "mbb_player_box",
  value = player_logs,
  overwrite = TRUE
)

cat("Data successfully written to database table: mbb_player_box\n")

# Disconnect
dbDisconnect(con)
