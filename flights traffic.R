####### Analysis flights Traffic

# Research Questions
# Which city has the most traffic? 
# Which city has the least traffic?
# What month is the most busiest in the year?
# Which airport route has the busiest one?

# Importing libraries
library(data.table)
library(tidyverse)
library(stringr)
library(plotly)
library(readr)
library(xml2)

# Importing dataset
airport <- read_csv('../input/au-dom-traffic/audomcitypairs-20180406.csv')

airport$City1 <- airport$City1 %>% str_to_lower()
airport$City1 <- airport$City1 %>% str_to_title()
airport$City2 <- airport$City2 %>% str_to_lower()
airport$City2 <- airport$City2 %>% str_to_title()
airport <- airport %>% filter(Year < 2018)
airport <- airport %>% filter(Year >= 2000)


city <- fread("../input/world-cities-database/worldcitiespop.csv", data.table = FALSE)

city.australia <- city %>% filter(Country == "au")
city.australia <- city.australia %>% select(-Country, -Population, -Region, -City)
names(city.australia)[1] <- "City"

# Data Component
airport %>% str()

# Cleaning
port.city <- c("Adelaide", "Albury", "Alice Springs", "Armidale", "Ayers Rock",
               "Ballina", "Brisbane", "Broome", "Bundaberg", "Burnie", "Cairns", "Canberra", 
               "Coffs Harbour", "Darwin", "Devonport", "Dubbo", "Emerald", "Geraldton", 
               "Gladstone", "Gold Coast", "Hamilton Island", "Hervey Bay", "Hobart",
               "Kalgoorlie","Karratha", "Launceston", "Mackay", "Melbourne", "Mildura", 
               "Moranbah", "Mount Isa", "Newcastle", "Newman", "Perth", "Port Hedland", 
               "Port Lincoln", "Port Macquarie", "Proserpine", "Rockhampton", "Sunshine Coast", 
               "Sydney", "Tamworth", "Townsville", "Wagga Wagga")

city.australia <- city.australia %>% filter(City %in% port.city)

# Merge dataset
airport <- merge(airport, city.australia, by.x = "City1", by.y = "City")
names(airport)[13] <- "City1.Latitude"
names(airport)[14] <- "City1.Longitude"

airport <- merge(airport, city.australia, by.x = "City2", by.y = "City")
names(airport)[15] <- "City2.Latitude"
names(airport)[16] <- "City2.Longitude"

# Analysis
# Map Visualization of all routes

airport <- airport %>%  mutate(id = rownames(airport))
airport.1 <- airport %>% 
  select(-contains("Latitude"), -contains("Longitude"))
airport.1 <- airport.1 %>% 
  gather('City1', 'City2', key = "Airport.type", value = "City")

airport.1$Airport.type <- airport.1$Airport.type %>% str_replace(pattern = "City1", replacement = "Departure")
airport.1$Airport.type <- airport.1$Airport.type %>% str_replace(pattern = "City2", replacement = "Arrive")
airport.1 <- merge(airport.1, city.australia, by.x = "City", by.y = "City")

world.map <- map_data ("world")
au.map <- world.map %>% filter(region == "Australia")
au.map <- fortify(au.map)

ggplot() + 
  geom_map(data=au.map, map=au.map,
           aes(x=long, y=lat, group=group, map_id=region),
           fill="white", colour="black") + 
  ylim(-43, -10) +
  xlim(110, 155) +
  geom_point(data = airport.1, aes(x = Longitude, y = Latitude)) +
  geom_line(data = airport.1, aes(x = Longitude, y = Latitude, group = id), colour = "red", alpha = .1) +
  labs(title = "Australian Domestic Aircraft Routes")

# Time Series Analysis
plot.year <- airport.1 %>% 
  ggplot(aes(x = Year, fill = City)) + 
  geom_bar() +
  labs(title = "Airport Traffic Amount by City from 2000 to 2017")
plot.year %>%
  ggplotly()

# Which city has the most traffic?
traffic.transition <- airport.1 %>% 
  group_by(City, Year) %>% 
  summarise(Annual.Aircraft_Trips = sum(Aircraft_Trips)) %>%
  ungroup() %>% 
  ggplot(aes(x = Year, y = Annual.Aircraft_Trips, group = City, colour = City)) +
  geom_line(show.legend = F) + 
  labs(title = "Annual Airport Traffic by each city from 2000 to 2017", y = "Annual Traffic")

traffic.transition %>%
  ggplotly()

# When to travel?
# Travel Season by each state

au.state.list <- fread("../input/city-list-of-australia/AUS_state.csv", data.table = FALSE)
au.state.list <- au.state.list %>% select(contains("SUA"), contains("State"))
names(au.state.list)[1] <- "City"
names(au.state.list)[2] <- "State"

airport.state <- merge(airport.1, au.state.list, by.x = "City", by.y = "City")
airport.state$Month_num <- as.factor(airport.state$Month_num)
airport.state %>% 
  ggplot(aes(x = Month_num, y = Aircraft_Trips, fill = Month_num)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~State, scales = "free") +
  labs(x = "Month", y = "Monthly Aircraft Trips", title = "Monthly Aircraft Trips by each state")


