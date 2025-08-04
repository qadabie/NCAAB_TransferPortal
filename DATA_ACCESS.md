### Data Access Statement

The data used in this project originates from the HoopR library, which provides cleaned NCAA basketball data sourced from 
KenPom and other public repositories. For project purposes, our team extracted individual player statistics from the 2021–2025 
seasons using HoopR and stored the data in a secure PostgreSQL database hosted by Aiven.

If you’re a collaborator or reviewer and would like access to the underlying data, please contact the project team directly for
database access. A CSV file containing player_table database information as of 8-4-2025 can be found in the data folder.

#### References
HoopR: Curley, J. P. (2023). hoopR: The SportsDataverse’s R Package for Men’s College Basketball Play-by-Play Data.
Available at: https://hoopr.sportsdataverse.org

KenPom.com: Ken Pomeroy’s advanced NCAA basketball statistics and player ratings.
https://kenpom.com (Note: Used indirectly via HoopR)

Aiven: Managed cloud data infrastructure platform used by the project team to host and secure the PostgreSQL database.
https://aiven.io