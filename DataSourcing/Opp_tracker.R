# Load required libraries
library(hoopR)
library(dplyr)
library(readr)
library(purrr)

# Read team table
teams <- read_csv("team_table.csv")

# Initialize empty data frame for results and failed teams vector
all_data <- data.frame()
failed_teams <- c()
team_dict <- list(
  "Michigan State" = "Michigan St.",
  "Iowa State" = "Iowa St.",
  "Mississippi State" = "Mississippi St.",
  "Ole Miss" = "Mississippi",
  "UConn" = "Connecticut",
  "American University" = "American",
  "San José State" = "San Jose St.",
  "San Diego State" = "San Diego St.",
  "Alabama State" = "Alabama St.",
  "Arkansas State" = "Arkansas St.",
  "Oklahoma State" = "Oklahoma St.",
  "Norfolk State" = "Norfolk St.",
  "Colorado State" = "Colorado St.",
  "Jackson State" = "Jackson St.",
  "Jacksonville State" = "Jacksonville St.",
      "Miami (OH)" = "Miami OH",
      "South Carolina State" = "South Carolina St.",
      "Wichita State" = "Wichita St.",
      "Kent State" = "Kent St.",
      "Kennesaw State" = "Kennesaw St.",
      "Delaware State" = "Delaware St.",
      "Bethune-Cookman" = "Bethune Cookman",
      "California Baptist" = "Cal Baptist",
      "Utah State" = "Utah St.",
      "McNeese" = "McNeese St.",
      "Ohio State" = "Ohio St.",
      "Grambling" = "Grambling St.",
      "Kansas State" = "Kansas St.",
      "Florida State" = "Florida St.",
      "Northwestern State" = "Northwestern St.",
      "Idaho State" = "Idaho St.",
      "Nicholls" = "Nicholls St.",
      "Cleveland State" = "Cleveland St.",
      "Portland State" = "Portland St.",
      "Texas A&M-Corpus Christi" = "Texas A&M Corpus Chris",
      "SE Louisiana" = "Southeastern Louisiana",
      "Omaha" = "Nebraska Omaha",
      "Arizona State" = "Arizona St.",
      "Long Island University" = "Long Island",
      "Miami" = "Miami FL",
      "Mississippi Valley State" = "Mississippi Valley St.",
      "Sam Houston" = "Sam Houston St.",
      "East Tennessee State" = "East Tennessee St.",
      "Arkansas-Pine Bluff" = "Arkansas Pine Bluff",
      "Albany NY" = "Albany",
      "Long Beach State" = "Long Beach St.",
      "Weber State" = "Weber St.",
      "Tarleton State" = "Tarleton St.",
      "Oregon State" = "Oregon St.",
      "Southeast Missouri State" = "Southeast Missouri St.",
      "Cal State Fullerton" = "Cal St. Fullerton",
      "Utah Tech" = "Utah Tech",
      "St. Thomas-Minnesota" = "St. Thomas",
      "Washington State" = "Washington St.",
      "North Dakota State" = "North Dakota St.",
      "App State" = "Appalachian St.",
      "South Dakota State" = "South Dakota St.",
      "Boise State" = "Boise St.",
      "Illinois State" = "Illinois St.",
      "Georgia State" = "Georgia St.",
      "Murray State" = "Murray St.",
      "Youngstown State" = "Youngstown St.",
      "Coppin State" = "Coppin St.",
      "Texas State" = "Texas St.",
      "Tennessee State" = "Tennessee St.",
      "UIC" = "Illinois Chicago",
      "Cal State Bakersfield" = "Cal St. Bakersfield",
      "St. Francis (PA)" = "Saint Francis",
      "NC State" = "N.C. State",
      "Gardner-Webb" = "Gardner Webb",
      "Kansas City" = "Kansas City",
      "Loyola Maryland" = "Loyola MD",
      "Ball State" = "Ball St.",
      "Wright State" = "Wright St.",
      "Fresno State" = "Fresno St.",
      "Alcorn State" = "Alcorn St.",
      "Morgan State" = "Morgan St.",
      "East Texas A&M" = "Texas A&M Commerce",
      "Montana State" = "Montana St.",
      "Indiana State" = "Indiana St.",
      "Penn State" = "Penn St.",
      "South Carolina Upstate" = "South Carolina Upstate",
      "Florida International" = "Florida International",
      "IU Indy" = "IUPUI",
      "Sacramento State" = "Sacramento St.",
      "UT Martin" = "Tennessee Martin",
      "Stonehill" = "Stonehill",
      "New Mexico State" = "New Mexico St.",
      "Seattle U" = "Seattle",
      "Hawai'i" = "Hawaii",
  "UL Monroe" = "Louisiana Monroe",
  "Mercyhurst" = "Mercyhurst University",
  "West Georgia" = "West Georgia",
  "Missouri State" = "Missouri St.",
  "Chicago State" = "Chicago St.",
  "Pennsylvania" = "Penn",
  "Aquinas" = "Aquinas College",
  "Puerto Rico-Rio Piedras" = "Puerto Rico Rio Piedras",
  "Life Pacific" = "Life Pacific University",
  "Morehead State" = "Morehead St.",
  "Queens University" = "Queens",
  "Florida Tech" = "Florida Tech",
  "Puerto Rico-Bayamon" = "Puerto Rico Bayamon"
)
# Process teams data with the dictionary
process_team_names <- function(team_name) {
    # Check if the team name is in the dictionary
    if (team_name %in% names(team_dict)) {
        return(team_dict[[team_name]])
    } else {
        return(team_name)
    }
}

# Apply the mapping to the teams dataframe
teams <- teams %>%
    mutate(team = sapply(team, process_team_names))

# Print confirmation message
cat("Team names updated according to dictionary mapping\n")
# Michigan State: Michigan St.
#
# Function to get opponent tracker data for a team and year
get_opp_data <- function(team_name, year) {
    tryCatch({
        # Get opponent tracker data
        opp_data <- kp_opptracker(team = team_name, year = year)
        
        # Add team and year columns
        opp_data <- opp_data %>%
            mutate(team = team_name, year = year) %>%
            select(team, year, date, adj_de, def_to_pct, def_apl, def_ft_rate)
        
        return(opp_data)
    }, error = function(e) {
        # Return NULL if there's an error
        return(NULL)
    })
}

# Loop through each team and year
for (team_name in teams$team) {
    team_success <- FALSE
    
    for (year in 2022:2024) {
        result <- get_opp_data(team_name, year)
        
        if (!is.null(result)) {
            all_data <- bind_rows(all_data, result)
            team_success <- TRUE
        }
    }
    
    # If all years failed for this team, add to failed teams list
    if (!team_success) {
        failed_teams <- c(failed_teams, team_name)
    }
}

# Print summary
cat("Data collection complete.\n")
cat("Number of successful data points:", nrow(all_data), "\n")
cat("Number of failed teams:", length(failed_teams), "\n")

# Print failed teams if any
if (length(failed_teams) > 0) {
    cat("Failed teams:\n")
    print(failed_teams)
}

# View first few rows of the data
head(all_data)

# Optional: Save the data
# Install arrow package if not already installed
if (!requireNamespace("arrow", quietly = TRUE)) {
    install.packages("arrow")
}

# Load arrow package
library(arrow)

# Save the data as a parquet file
write_parquet(all_data, "opponent_tracker_data.parquet")

cat("Data saved to opponent_tracker_data.parquet\n")