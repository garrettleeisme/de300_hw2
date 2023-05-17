# Data Engineering 300 - Homework 2
### Name: Garrett Lee
### Date: May 17, 2023

## Commands to Run Docker
docker build -t hw2:0.2 . <br>
docker run -v "$(pwd)/data":/tmp/data hw2:0.2

## Answers to Questions
2 - To add weather conditions for the pickup times, I assumed that the weather does not change from the previous top of the hour (e.g., weather at 12:59pm will follow the weather listed for 12pm).

3a - If I look at the WSSSE visualization to use the elbow method to find the best k, such value is 5. The note here is that even though there is a sharp decrease in the WSSSE for k=4, there is substantial decrease of the error gained from k=4 to k=5, which prompted the choice of k for the latter. k=5 yielded a silhouette score of 0.60. <br>
![Figure 1](FiguresFromPreviousOutput/k_means_graph.png)

3b - The figure representing the clusters is below: <br>
![Figure 2](FiguresFromPreviousOutput/all_clusters.png)

3c-i - (*Note: I had to filter some points out to zoom in to the LaGuardia Airport since some points were too far away from the airport to be considered as originating at the airport*.) By zooming in the trips originating from LaGuardia Airport (approximately 40.75-40.78 north and 73.850-73.885 west), we realize that there is a group of trips originating from the Grand Central Parkway (the road right in front of the entrance to the airport). Outside of it, there is also some calls for taxi on 23rd Avenue and 94th Street, the section that intersects Grand Central Parkway. The LaGuardia Airport cluster has 2122 pick-up locations, the centroid coordinate of (40.77, -73.87), and the variance of 5.01e-06 latitude degrees and  2.06e-05 longitude degrees. (Reference: output.txt) <br>
![Figure 3](FiguresFromPreviousOutput/laguardia_cluster.png)

3c-ii - (*Note: I had to filter some points out to zoom in to the JFK Airport since some points were too far away from the airport to be considered as originating at the airport*.) By zooming in the trips originating from JFK Airport (approximately 40.63-40.70 north and 73.77-73.825 west), we see a distinct curve line leading to the center of the cluster, which goes along the Van Wyck Expressway leading to the terminals. The expressway leading to the airport goes to the middle of the terminals, where the traffic breaks off to different terminals. Therefore, it is not surprising that (unlike LaGuardia) we see a complete oval shape for pick-up coordinates. The cluster has 1611 pick-up locations, the centroid coordinate of (40.65,-73.79), and the variance of 3.43e-06 latitude degrees and 2.06e-05 longitude degrees. (Reference: output.txt) <br>
![Figure 4](FiguresFromPreviousOutput/jfk_cluster.png)

3c-iii - Comparing the two outputs, it is clear that:
1. There are more taxi trips made in LaGuardia than that of JFK during the same time. This makes sense because JFK has more transit options while LaGuardia does not. 
2. The variance of the JFK cluster is slightly larger than that of LaGuardia. We could connect this to the fact that JFK has more terminals, leading the trip requests to be more spread out.
 
3d - The silhouette score for the dropoff coordinates is 0.51. (Reference: output.txt)

4 - Intra-cluster trips: Zone 4 (lower parts of Midtown and upper parts of Lower Manhattan) seem to have many trips happening within the area. The next cluster that follows is lower Manhattan and lower Brooklyn (Zone 2), which makes sense since there is a lot of activity happening within the area, including the financial area. Interestingly, there does not seem to be many intra-cluster trips on the east side. This might support the hypothesis that most taxi users are using the service to get to Manhattan for commuting. Excluding the intra-cluster trips (Zones 0, 2, and 4 are the top three), the trips happening between these three zones seem to take a high proportion. Therefore, we can conclude that trips in Brooklyn and Manhattan dominate. It nmight be interesting to design more clusters and see if this trend can be observed only for the Manhattan island. 

5-i - The distribution of the logarithm of trip durations looks roughly normally distributed, but it seems to be slighly right-tailed. The center seems to be at around 2.5 logarithm units. <br>
![Figure 4](FiguresFromPreviousOutput/jfk_cluster.png)

5-ii - By inspecting the graph, We observe a small gradual decrease in the number of trips throughout the month, but that trend is not too obvious.  <br>
![Figure 5](FiguresFromPreviousOutput/log_trip_durations_hist.png)

6-i - Longer trips are made mid-week in NYC Taxis. One explanation of this could be that businessman who travel during the week tend to use taxi cabs to avoid transferring since they might be on tight schedules. The weekends, on the other hand, people might have more flexibility to transfer to another mode of transportation or take public transit. 
(Reference: data/log_trip_durations_by_day_of_week.png)

6-ii - It is clear that there are longer travels made in the taxis than weekends. This may be caused by the rush hour traffic. (Reference: data/log_trip_durations_weekday_or_weekend.png)

6-iii - It is clear that the morning seems to be the time when there are long trips. Again, we will see if the morning rush hour hypothesis is true. (Reference:data/log_trip_durations_parts_of_day.png)

6-iv - We finally reach the conclusion that during weekday rush periods (weekday mornings) taxi trips are the longest. This matches with our intuition that cars stuck in morning traffic jams will dominate the average length of trips. (Reference: data/log_trip_durations_weekday_rush.png)

7b - The linear asymptote in the graph most likely represents the upper limit as to how fast a taxi can go. Any point that is below that line represents a taxi trip that was not efficient, since the vehicle experienced delays or travelled not as fast as the vehicle can. <br>
![Figure 6](FiguresFromPreviousOutput/log_trip_durations_log_distance.png)

7c - Since the distribution of the log of the ground distance is not skewed (skewed statistics of 0.151), so we will keep the logarithm transformation for future use. (Reference: output.txt)

8a - Here, we need to convert the categorical variables into numerical ones so that the regression can be done. I will use one hot encoding for the day_of_week and part_of_day columns, because there is no ascending or descending trend that is associated with each category in the two variables. Each category has its own characteristics, so the regression will benefit if there are separate attributes representing each. I additionally chose to include temperature, pressure, and precipitation because I suspected that they will influence the trip duration. I chose not to include other columns, since I suspected that they will not impact the trip duration (e.g., for the number of passengers, I think that having more passengers does not make the trip longer or shorter. The distance of travel should still be the same).

8b - The Test RSME is 0.429, MAE 0.330, and the R-squared is 0.578. (Reference: output.txt)

8c - The Test RSME is 0.400, MAE 0.314, and the R-squared is 0.633. Log_distance (importance statistic of 0.919) seems to be significantly important, while all other features are practically unimportant. The next important features after the logarithm of distance is the indicator variable for night trips (0.0176), the indicator variable for afternoon trips (0.0147), temperature (0.0106), and pressure (0.0105). Since the trip distance clearly dominates the trip duration (which is intuitive), the next question should be trying to predict the trip duration without the distance information. (Reference: output.txt)
