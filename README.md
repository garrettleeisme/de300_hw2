# Data Engineering 300 - Homework 2
### Name: Garrett Lee
### Date: May 17, 2023

## Commands to Run Docker
docker build -t hw2:0.2 . <br>
docker run -v "$(pwd)/data":/tmp/data hw2:0.2

## Answers to Questions
2 - To add weather conditions for the pickup times, I assumed that the weather does not change from the previous top of the hour (e.g., weather at 12:59pm will follow the weather listed for 12pm).

3a - If I look at the WSSSE visualization to use the elbow method to find the best k, such value is 5. k=5 yielded a silhouette score of 0.60. 
![Figure 1](FiguresFromPreviousOutput/k_means_graph.png)

3b - The figure representing the clusters is below: 
![Figure 2](FiguresFromPreviousOutput/all_clusters.png)

3c-i - (*Note: I had to filter some points out to zoom in to the LaGuardia Airport since some points were too far away from the airport to be considered as originating at the airport*.) By zooming in the trips originating from LaGuardia Airport (approximately 40.75-40.78 north and 73.850-73.885 west), we realize that there is a group of trips originating from the Grand Central Parkway (the road right in front of the entrance to the airport). Outside of it, there is also some calls for taxi on 23rd Avenue and 94th Street, the section that intersects Grand Central Parkway. (Reference: data/laguardia_cluster.png) The LaGuardia Airport cluster has 2122 pick-up locations, the centroid coordinate of (40.77, -73.87), and the variance of 5.01e-06 latitude degrees and  2.06e-05 longitude degrees. 
![Figure 3](FiguresFromPreviousOutput/data/laguardia_cluster.png)

3c-ii - (*Note: I had to filter some points out to zoom in to the JFK Airport since some points were too far away from the airport to be considered as originating at the airport*.) By zooming in the trips originating from JFK Airport (approximately 40.63-40.70 north and 73.77-73.825 west), we see a distinct curve line leading to the center of the cluster, which goes along the Van Wyck Expressway leading to the terminals. The expressway leading to the airport goes to the middle of the terminals, where the traffic breaks off to different terminals. Therefore, it is not surprising that (unlike LaGuardia) we see a complete oval shape for pick-up coordinates. (Reference: data/jfk_cluster.png) The cluster has 1611 pick-up locations, the centroid coordinate of (40.65,-73.79), and the variance of 3.43e-06 latitude degrees and 2.06e-05 longitude degrees.
![Figure 4](FiguresFromPreviousOutput/data/jfk_cluster.png)

3c-iii - Comparing the two outputs, it is clear that:
1. There are more taxi trips made in LaGuardia than that of JFK during the same time. This makes sense because JFK has more transit options while LaGuardia does not. 
2. The variance of the JFK cluster is slightly larger than that of LaGuardia. We could connect this to the fact that JFK has more terminals, leading the trip requests to be more spread out.
 
3d - The silhouette score for the dropoff coordinates is 0.60. (Reference: output.txt)

4 - There seems to be the most number of taxi trips originating from East Queens to Midtown (Manhattan), closely followed by trips from Uptown (Manhattan) to Cluster 4 (lower parts of Midtown and upper parts of Lower Manhattan). Roughly half of the previous two trip counts are trip counts from East Queens to West Queens and Uptown and those from Uptown to the Financial District and lower Brooklyn. (Reference: output.txt)

5a - The distribution of the logarithm of trip durations looks roughly normally distributed, but it seems to be slighly right-tailed. The center seems to be at around 2.5 logarithm units. 
![Figure 4](FiguresFromPreviousOutput/data/jfk_cluster.png)

5b - By inspecting the graph, We observe a small gradual decrease in the number of trips throughout the month, but that trend is not too obvious.  
![Figure 5](FiguresFromPreviousOutput/data/log_trip_durations_hist.png)

6a - Longer trips are made mid-week in NYC Taxis. One explanation of this could be that businessman who travel during the week tend to use taxi cabs to avoid transferring since they might be on tight schedules. The weekends, on the other hand, people might have more flexibility to transfer to another mode of transportation or take public transit. 
(Reference: data/log_trip_durations_by_day_of_week.png)

6b - It is clear that there are longer travels made in the taxis than weekends. This may be caused by the rush hour traffic. (Reference: data/log_trip_durations_weekday_or_weekend.png)

6c - It is clear that the morning seems to be the time when there are long trips. Again, we will see if the morning rush hour hypothesis is true. (Reference:data/log_trip_durations_parts_of_day.png)

6c - We finally reach the conclusion that during weekday rush periods (weekday mornings) taxi trips are the longest. This matches with our intuition that cars stuck in morning traffic jams will dominate the average length of trips. (Reference: data/log_trip_durations_weekday_rush.png)

7 - Since the distribution of the log of the ground distance is not skewed, so we will keep the logarithm transformation for future use. (Reference: output.txt)

7 - The linear asymptote in the graph most likely represents the upper limit as to how fast a taxi can go. Any point that is below that line represents a taxi trip that was not efficient, since the vehicle experienced delays. 
![Figure 6](FiguresFromPreviousOutput/data/log_trip_durations_log_distance.png)

8a - Here, we need to convert the categorical variables into numerical ones so that the regression can be done. I will use one hot encoding for the day_of_week and part_of_day columns, because there is no ascending or descending trend that is associated with each category in the two variables. Each category has its own characteristics, so the regression will benefit if there are separate attributes representing each.

8b - The Test RSME is 0.429, MAE 0.330, and the R-squared is 0.578. (Reference: output.txt)

8c - The Test RSME is 0.400, MAE 0.314, and the R-squared is 0.633. The most important feature is only log_distance (importance statistic of 0.919), while all other features are practically unimportant. (Reference: output.txt)
