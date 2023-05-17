0### DE 300 Homework 2
# Name: Garrett Lee
# Due: May 17, 2023

## ---------------------------------------------
# 0a. Importing Packages and data
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.mllib.stat import Statistics

import matplotlib.pyplot as plt
import math
import pandas 

# 0b. Parameters
TAXI_DATA = "./nyc_taxi_june.csv"
WEATHER_DATA = "./weather_hourly_june.csv"
OUTPUT_DIR = "results" # name of the folder
seconds_to_minute = 60
seconds_to_hour = 3600
low_lim_long = -74.03
high_lim_long = -73.75
low_lim_lat = 40.63
high_lim_lat = 40.85
pickup_long = 'pickup_longitude'
dropoff_long = 'dropoff_longitude'
pickup_lat = 'pickup_latitude'
dropoff_lat = 'dropoff_latitude'
part_of_day_list = ["Morning", "Afternoon", "Evening", "Night"]
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
columns_to_select = ["day", "Temp", "Pressure", "Precip", "log_trip_duration", "day_of_week", "part_of_day", "log_distance"]
feature_cols = ['Temp', 'Pressure', 'Precip', 'log_distance', 'day_of_week_encoded', 'part_of_day_encoded']
input_feature_cols = ["Temp", "Pressure", "Precip", "log_distance", "on_monday", "on_tuesday",
                      "on_wednesday", "on_thursday", "on_friday", "on_saturday", "on_sunday",
                      "at_morning", "at_afternoon", "at_evening", "at_night"]
stats_columns_to_print = "Category, Average, Maximum, Minimum, Median, Variance"


# 0c. Create a SparkSession with eager evaluation disabled
spark = SparkSession.builder \
    .appName("ReadCSV") \
    .config("spark.sql.repl.eagerEval.enabled", False) \
    .getOrCreate()
sc = spark.sparkContext

# 0d. read the CSV file into a DataFrame
taxi = spark.read.csv(TAXI_DATA, header=True, inferSchema=True)
weather = spark.read.csv(WEATHER_DATA, header=True, inferSchema=True)

# 0e. Functions
def print_shape(spark_df):
    '''
    Print the shape of the Spark DataFrame
    '''
    num_rows = spark_df.count()
    num_cols = len(spark_df.columns)
    print("Dataset Taxi has ", num_rows, " rows and ", num_cols, " columns.")

def log_haversine(lat1, lon1, lat2, lon2):
    '''
    Calculate the Haversine distance between two points
    '''
    R = 6371.0 # Earth radius in km

    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))
    # print(lat1_rad)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    # print(distance)
    return float(np.log(distance))

def categorize_time_of_day(entry):
    num_hour = int(entry)
    if (num_hour >= 0) & (num_hour < 6):
        return part_of_day_list[3]
    elif (num_hour >= 6) & (num_hour < 12):
        return part_of_day_list[0]
    elif (num_hour >= 12) & (num_hour < 17):
        return part_of_day_list[1]
    elif (num_hour >= 17) & (num_hour < 21):
        return part_of_day_list[2]
    elif (num_hour >= 21) & (num_hour < 24):
        return part_of_day_list[3]

def categorize_weekday_or_weekend(entry):
    if (entry in weekdays):
        return "Weekday"
    else:
        return "Weekend"

def weekday_rush_or_not(part_of_day, weekday_or_weekend):
    if (part_of_day == part_of_day_list[0]) & (weekday_or_weekend == "Weekday"):
        return "Yes"
    else:
        return "No"
    

def compute_statistics(datafr, x_col):
    datafr = datafr.withColumn("log_trip_duration", datafr["log_trip_duration"].cast("float"))

    formatted_stats = datafr.groupBy(x_col).agg(
        format_number(avg("log_trip_duration"), 3).alias("avg_duration"),
        format_number(max("log_trip_duration"), 3).alias("max_duration"),
        format_number(min("log_trip_duration"), 3).alias("min_duration"),
        format_number(expr("percentile(log_trip_duration, 0.5)"), 3).alias("median_duration"),
        format_number(expr("var_samp(log_trip_duration)"), 3).alias("variance")
    )

    return formatted_stats


## ---------------------------------------------
## 1. Filtering the Taxi Trip Data

# 1a. Remove all observations where the trip duration is less than 3 minutes those where the trip duration is longer than 6 hours.
taxi = taxi.filter((taxi['trip_duration'] >= 3 * seconds_to_minute) & (taxi['trip_duration'] <= 6 * seconds_to_hour))

# 1b. Remove all observations where the pickup and dropoff coordinates are the same.
taxi = taxi.filter((taxi['pickup_longitude']!=taxi['dropoff_longitude'])&(taxi['pickup_latitude']!=taxi['dropoff_latitude']))

# 1c. Remove all observations where either the pickup or the dropoff points lie outside NYC.
taxi = taxi.filter((taxi[pickup_long] >= low_lim_long) & (taxi[pickup_long] <= high_lim_long) & \
                   (taxi[dropoff_long] >= low_lim_long) & (taxi[dropoff_long] <= high_lim_long) & \
                   (taxi[pickup_lat] >= low_lim_lat) & (taxi[pickup_lat] <= high_lim_lat) & \
                   (taxi[dropoff_lat] >= low_lim_lat) & (taxi[dropoff_lat] <= high_lim_lat))

## ---------------------------------------------
## 2. Weather Attributes

# Extract day and hour from timestamp column in taxi DataFrame
taxi = taxi.withColumn("day", dayofyear(taxi["pickup_datetime"]))
taxi = taxi.withColumn("hour", hour(taxi["pickup_datetime"]))

# Extract day and hour from timestamp column in weather DataFrame
weather = weather.withColumn("day", dayofyear(weather["Time"]))
weather = weather.withColumn("hour", hour(weather["Time"]))

# Select columns from weather DataFrame
weather_selected = weather.select(col("day"), col("hour"), col("Temp"), col("Pressure"), col("Precip"))

# Perform left join on selected columns
joined_taxi = taxi.join(weather_selected, on=["day", "hour"], how="left_outer")

## ---------------------------------------------
## 3a. K-Means Clustering - Finding the right k using Elbow Method

print("QUESTION 3a OUTPUTS:")

k_values = list(range(2,30,1))

# create a VectorAssembler to assemble the feature columns into a single vector column
assembler = VectorAssembler(inputCols=["pickup_latitude", "pickup_longitude"], outputCol="features")

# transform the dataframe using the VectorAssembler
temp_taxi = assembler.transform(joined_taxi)

wssse_values = []
silhouette_scores = []
for k_trial in k_values:
    kmeans_trial = KMeans().setK(k_trial).setSeed(1)
    model_trial = kmeans_trial.fit(temp_taxi)

    # Append WSSSE values
    wssse_values.append(model_trial.summary.trainingCost)

    # Append silhouette scores
    predictions_trial = model_trial.transform(temp_taxi)
    predictions_trial = predictions_trial.withColumnRenamed('prediction', 'pickup_cluster')
    evaluator = ClusteringEvaluator().setPredictionCol("pickup_cluster").setMetricName("silhouette").setDistanceMeasure("squaredEuclidean")
    silhouette_score = evaluator.evaluate(predictions_trial)
    silhouette_scores.append(silhouette_score)
    if k_trial == 5:
        print(f"The silhouette score when k={k_trial} is {silhouette_score:.2f}\n")

plt.plot(k_values, wssse_values, 'bx-')
plt.xlabel('k')
plt.ylabel('WSSSE')
plt.title('WSSSE based on K in K-Means Clustering')
plt.savefig(fname = 'data/k_means_graph.png')

## ---------------------------------------------
## 3b. Graphing the Clusters

# create a VectorAssembler to assemble the feature columns into a single vector column
assembler = VectorAssembler(inputCols=["pickup_latitude", "pickup_longitude"], outputCol="features")

# transform the dataframe using the VectorAssembler
KMeans_taxi = assembler.transform(joined_taxi)

# create a KMeans model with k=5
kmeans_5 = KMeans().setK(5).setSeed(1)

# fit the KMeans model to the data
model_5 = kmeans_5.fit(KMeans_taxi)

# use the model to make predictions on the data
predictions_pickup = model_5.transform(KMeans_taxi)

# rename the columns
predictions_pickup = predictions_pickup.withColumnRenamed('prediction', 'pickup_cluster')

# extract the predicted cluster labels from the predictions dataframe
labels = predictions_pickup.select("pickup_cluster").rdd.flatMap(lambda x: x).collect()

# extract the feature vectors from the predictions dataframe
features = predictions_pickup.select("features").rdd.map(lambda x: x[0]).collect()

# convert the feature vectors to a list of (x, y) tuples
coords = [(f[0], f[1]) for f in features]

# get unique labels
unique_labels = set(labels)

plt.clf()

# create scatter plot for each label with appropriate label
for label in unique_labels:
    plt.scatter([coord[1] for i, coord in enumerate(coords) if labels[i] == label],
                [coord[0] for i, coord in enumerate(coords) if labels[i] == label],
                color=plt.cm.jet(label / len(unique_labels)),
                label='Cluster {}'.format(label))

# add legend to plot
plt.legend()

# Appropriate axes
plt.title("K-Means Cluster of Pickup Coordinates")
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")

# Stretch the scatterplot vertically by 4/3
fig = plt.gcf()
fig.set_size_inches(6, 8)

# show the plot
plt.savefig(fname = 'data/all_clusters.png')
plt.clf()

## ---------------------------------------------
## 3c-i. Zooming into LaGuardia Airport origins

print("\nQUESTION 3c OUTPUTS:")

# create a scatter plot of the data with different colors for each cluster
plt.scatter([c[1] for c in coords], [c[0] for c in coords], c=labels)

# Appropriate axes
plt.title("K-Means Cluster of Pickup Coordinates - LaGuardia")
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")
plt.xlim(-73.885, -73.85)
plt.ylim(40.75, 40.79)

# Stretch the scatterplot vertically by 4/3
fig = plt.gcf()
fig.set_size_inches(6, 8)

# show the plot
plt.savefig(fname = 'data/laguardia_cluster.png')
plt.clf()

# Isolating the cluster in LaGuardia
LaGuardia_temp1 = KMeans_taxi.filter((KMeans_taxi['pickup_latitude'] >= 40.760) & (KMeans_taxi['pickup_latitude'] <=40.775) & \
                                       (KMeans_taxi['pickup_longitude'] >= -73.88) & (KMeans_taxi['pickup_longitude'] <= -73.86))
LaGuardia_temp2 = LaGuardia_temp1.select("pickup_latitude", "pickup_longitude").collect()

# Convert dataset to list of dictionaries
LaGuardia_temp3 = [{'pickup_latitude': row.pickup_latitude, 'pickup_longitude': row.pickup_longitude} for row in LaGuardia_temp2]

# Create PySpark DataFrame from list of dictionaries
LaGuardia_origins = spark.createDataFrame(LaGuardia_temp3)

# Calculate centroid of cluster
centroid = LaGuardia_origins.groupBy().agg(mean('pickup_latitude').alias('centroid_latitude'), \
                                           mean('pickup_longitude').alias('centroid_longitude')).collect()[0]
print(f"The centroid of the LaGuardia cluster is ({centroid['centroid_latitude']: .2f},{centroid['centroid_longitude']: .2f}).")

# Calculate the number of points
num_points = LaGuardia_origins.count()
print("The LaGuardia cluster has " + str(num_points) + " points.")

# Calculate shape of cluster
shape = LaGuardia_origins.agg(variance('pickup_latitude').alias('latitude_variance'),\
                              variance('pickup_longitude').alias('longitude_variance')).collect()[0]
print(f"The LaGuardia cluster has a variance of {shape['latitude_variance']: .2e} latitude degrees and {shape['longitude_variance']: .2e} longitude degrees.")

## ---------------------------------------------
## 3c-ii. Zooming into JFK Airport origins
# create a scatter plot of the data with different colors for each cluster
plt.scatter([c[1] for c in coords], [c[0] for c in coords], c=labels)

# Appropriate axes
plt.title("K-Means Cluster of Pickup Coordinates - JFK")
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")
plt.xlim(-73.825, -73.77)
plt.ylim(40.63, 40.70)

# Stretch the scatterplot vertically by 4/3
fig = plt.gcf()
# fig.set_size_inches(6, 8)

# show the plot
plt.savefig(fname = 'data/jfk_cluster.png')
plt.clf()

# Isolating the cluster in JFK
JFK_temp1 = KMeans_taxi.filter((KMeans_taxi['pickup_latitude'] >= 40.64) & (KMeans_taxi['pickup_latitude'] <=40.65) & \
                               (KMeans_taxi['pickup_longitude'] >= -73.792) & (KMeans_taxi['pickup_longitude'] <= -73.775))
JFK_temp2 = JFK_temp1.select("pickup_latitude", "pickup_longitude").collect()

# Convert dataset to list of dictionaries
JFK_temp3 = [{'pickup_latitude': row.pickup_latitude, 'pickup_longitude': row.pickup_longitude} for row in JFK_temp2]

# Create PySpark DataFrame from list of dictionaries
JFK_origins = spark.createDataFrame(JFK_temp3)

# Calculate centroid of cluster
centroid = JFK_origins.groupBy().agg(mean('pickup_latitude').alias('centroid_latitude'), \
                                           mean('pickup_longitude').alias('centroid_longitude')).collect()[0]
print(f"The centroid of the JFK cluster is ({centroid['centroid_latitude']: .2f},{centroid['centroid_longitude']: .2f}).")

# Calculate the number of points
num_points = JFK_origins.count()
print("The JFK cluster has " + str(num_points) + " points.")

# Calculate shape of cluster
shape1 = JFK_origins.agg(variance('pickup_latitude').alias('latitude_variance'), \
                        variance('pickup_longitude').alias('longitude_variance')).collect()[0]
print(f"The JFK cluster has a variance of {shape1['latitude_variance']: .2e} latitude degrees and {shape1['longitude_variance']: .2e} longitude degrees.")

## ---------------------------------------------
## 3d. Assigning the dropoff coordinates to the different clusters

print("\nQUESTION 3d OUTPUTS:")

# Create a new VectorAssembler to assemble the feature columns 'dropoff_latitude' and 'dropoff_longitude' into a single vector column
assembler_dropoff = VectorAssembler(inputCols=["dropoff_latitude", "dropoff_longitude"], outputCol="features")

# Transform the predictions dataframe using the new VectorAssembler to add a new column with the features for the 'dropoff_latitude' and 'dropoff_longitude' columns
KMeans_taxi_dropoff = assembler_dropoff.transform(joined_taxi)

# Use the model.transform() method to predict the cluster labels for the 'dropoff_latitude' and 'dropoff_longitude' features in the transformed joined_taxi dataframe
predictions_dropoff = model_5.transform(KMeans_taxi_dropoff)
predictions_dropoff = predictions_dropoff.withColumnRenamed('prediction', 'dropoff_cluster')

# Select the necessary columns from predictions_pickup DataFrame
pickup_cluster = predictions_pickup.select("id", "pickup_cluster")

# Perform an inner join based on the identifier column
predictions_dropoff = predictions_dropoff.join(pickup_cluster, "id", "inner")

# predictions_dropoff will be used frequently for the remainder of the file. Therefore, I will persist this DataFrame
predictions_dropoff.persist()

evaluator = ClusteringEvaluator().setPredictionCol("dropoff_cluster").setMetricName("silhouette").setDistanceMeasure("squaredEuclidean")
silhouette_score = evaluator.evaluate(predictions_dropoff)
print(f"The silhouette score for the dropoff coordinates is: {silhouette_score:.2f}.")

## ---------------------------------------------
## 4. Further Cluster Analysis

print("\nQUESTION 4 OUTPUTS:")

# Convert predictions dataframe to an RDD
rdd = predictions_dropoff.rdd

# Map each row to a tuple containing pickup and dropoff cluster labels as a string and a count of the number of trips
pair_count = rdd.map(lambda row: ("{}-{}".format(row["pickup_cluster"], row["dropoff_cluster"]), 1)).reduceByKey(lambda a, b: a + b)

# Sort the pairs by count in decreasing order
sorted_pairs = pair_count.sortBy(lambda pair: -pair[1])

# Print all pickup-dropoff trips
print("All pickup-dropoff trips: ")
for pair in sorted_pairs.collect():
    print("{}\t{}".format(pair[0], pair[1]))

# Filter rows where pickup cluster is the same as drop-off cluster
filtered_rdd = rdd.filter(lambda row: row["pickup_cluster"] == row["dropoff_cluster"])

# Map each row to a tuple containing pickup and dropoff cluster labels as a string and a count of the number of trips
intra_cluster_pair_count = filtered_rdd.map(lambda row: ("{}-{}".format(row["pickup_cluster"], row["dropoff_cluster"]), 1)).reduceByKey(lambda a, b: a + b)

# Sort the intra-cluster pairs by count in decreasing order
sorted_intra_cluster_pairs = intra_cluster_pair_count.sortBy(lambda pair: -pair[1])

# Print intra-cluster pickup-dropoff trips
print("Intra-cluster pickup-dropoff trips: ")
for pair in sorted_intra_cluster_pairs.collect():
    print("{}\t{}".format(pair[0], pair[1]))

## ---------------------------------------------
## 5. Visualizing Trip Durations
predictions_dropoff = predictions_dropoff.withColumn("trip_duration_min", col("trip_duration") / seconds_to_minute)

# create a histogram of the "trip_duration" column
rdd = predictions_dropoff.select("trip_duration_min").rdd.map(lambda row: math.log(row[0]))
histogram = rdd.map(lambda x: (x,)).toDF(["log_trip_duration"])
pandas_df = histogram.toPandas()

# plot the histogram using matplotlib
plt.hist(pandas_df)
plt.title("Histogram of Log of Trip Durations")
plt.xlabel("Log of Trip Duration (minutes)")
plt.ylabel("Frequency")
plt.savefig(fname="data/log_trip_durations_hist.png")
plt.clf()

# Add a log of trip duration column to predictions_dropoff
predictions_dropoff = predictions_dropoff.withColumn("log_trip_duration", log("trip_duration_min"))

temp_rdd = predictions_dropoff.groupBy("day").agg(avg("log_trip_duration").alias("average_log_trip_duration")).orderBy("day")
temp_rdd2 = temp_rdd.rdd.map(lambda line: (line[0]-151,line[1]))
temp_rdd3 = temp_rdd2.filter(lambda x: x[0] < 31)

histogram1 = temp_rdd3.map(lambda x: (x[0],x[1])).toDF(["day", "average_trip_duration"])
pandas_df1 = histogram1.toPandas()

# Independent and dependent dataset
x_axis = pandas_df1["day"]
y_axis = pandas_df1["average_trip_duration"]

# plot the histogram using matplotlib
plt.bar(x_axis, y_axis)
plt.title("Log of Trip Durations by Day")
plt.xlabel("Day in June")
plt.ylabel("Number of Trips")
plt.ylim(y_axis.min() - 0.1, y_axis.max() + 0.1)
plt.savefig(fname="data/log_trip_durations_hist_by_day.png")
plt.clf()

## ---------------------------------------------
## 6. Time-based Features

print("\nQUESTION 6 OUTPUTS:")

# Add the day of the week to each row
predictions_dropoff = predictions_dropoff.withColumn("day_of_week", date_format("pickup_datetime", "EEEE"))

# Register the user-defined function as a PySpark UDF
string_length_udf = udf(categorize_time_of_day, StringType())

# Add the part of the day for each row
predictions_dropoff = predictions_dropoff.withColumn("part_of_day", string_length_udf(predictions_dropoff["hour"]))

## 6.1 Day of the week
x_col = "day_of_week"
y_col = "log_dur_by_day_of_week"

grouped_stats = compute_statistics(predictions_dropoff, x_col)
result = grouped_stats.collect()

print("Statistics for Each Day of the Week Trip Durations:")
print(stats_columns_to_print)
for row in result:
    print(row)

# 6.2 Weekday or Weekend?

# Register the user-defined function as a PySpark UDF
string_length_udf = udf(categorize_weekday_or_weekend, StringType())

# Add the part of the day for each row
predictions_dropoff = predictions_dropoff.withColumn("weekday_or_weekend", string_length_udf(predictions_dropoff["day_of_week"]))

x_col = "weekday_or_weekend"
y_col = "log_dur_by_weekday_or_weekend"

grouped_stats = compute_statistics(predictions_dropoff, x_col)
result = grouped_stats.collect()

print("\nStatistics for Weekday or Weekend Trip Durations:")
print(stats_columns_to_print)
for row in result:
    print(row)

# 6.3 Parts of the Day
x_col = "part_of_day"
y_col = "log_dur_part_of_day"

grouped_stats = compute_statistics(predictions_dropoff, x_col)
result = grouped_stats.collect()

print("\nStatistics for Part of Day Trip Durations:")
print(stats_columns_to_print)
for row in result:
    print(row)

# 6.4 Weekday Rush or Not

# Register the user-defined function as a PySpark UDF
string_length_udf = udf(weekday_rush_or_not, StringType())

# Add the part of the day for each row
predictions_dropoff = predictions_dropoff.withColumn("weekday_rush_or_not", \
                                                     string_length_udf(predictions_dropoff["part_of_day"], \
                                                                       predictions_dropoff["weekday_or_weekend"]))

x_col = "weekday_rush_or_not"
y_col = "log_dur_weekday_rush_or_not"

grouped_stats = compute_statistics(predictions_dropoff, x_col)
result = grouped_stats.collect()

print("\nStatistics for Weekday Rush or Not Trip Durations:")
print(stats_columns_to_print)
for row in result:
    print(row)

## ---------------------------------------------
## 7. Ground Distance

print("\nQUESTION 7 OUTPUTS:")

# Define user-defined function
haversine_udf = udf(log_haversine, DoubleType())

# Add a new column to the DataFrame with the calculated distance
df_with_distance = predictions_dropoff.withColumn(
    "log_distance",
    haversine_udf(col("pickup_latitude"), col("pickup_longitude"), \
                  col("dropoff_latitude"), col("dropoff_longitude"))
)

# Release the dataframes that are no longer needed 
predictions_pickup.unpersist()
predictions_dropoff.unpersist()

# Histogram of Log of Distance
column1 = "log_trip_duration"
column2 = "log_distance"

# calculate the skewness of col1
skewness_value = df_with_distance.select(skewness(column2)).collect()[0][0]

# print the skewness value
print("Skewness of " + column2 +  ":", skewness_value)

# Select two columns from the DataFrame
df_select = df_with_distance.select(column1, column2)

# Convert the DataFrame to a Pandas DataFrame
pandas_df = df_select.toPandas()

# Plot a scatter plot using matplotlib
plt.scatter(pandas_df[column1], pandas_df[column2])
plt.title("Log of Trip Duration by Log of Ground Distance")
plt.xlabel("Log of Trip Duration")
plt.ylabel("Log of Trip Distance")
plt.savefig(fname="data/log_trip_durations_log_distance.png")
plt.clf()

## ---------------------------------------------
## 8. Predicting Trip Duration
print("\nQUESTION 8 OUTPUTS:")

# Finally subtracting 151 to get the actual day
df_with_distance = df_with_distance.withColumn("day", col("day") - 152)

# Deleting data that is unnecessary for the predictions
prediction_df = df_with_distance.select(columns_to_select)

## ---------------------------------------------
# 8.1 Converting Categorical Variables
prediction_df = prediction_df.drop("day_of_week_index", "part_of_day_index", "day_of_week_encoded", "part_of_day_encoded", "encoded_features")

# StringIndexer for day_of_week
day_of_week_indexer = StringIndexer(inputCol='day_of_week', outputCol='day_of_week_index')
prediction_df = day_of_week_indexer.fit(prediction_df).transform(prediction_df)

# OneHotEncoder for day_of_week
day_of_week_encoder = OneHotEncoder(inputCols=['day_of_week_index'], outputCols=['day_of_week_encoded'])
prediction_df = day_of_week_encoder.fit(prediction_df).transform(prediction_df)

# Create separate columns for each day of the week
for day in days_of_week:
    column_name = f"on_{day.lower()}"
    prediction_df = prediction_df.withColumn(column_name, when(prediction_df.day_of_week == day, 1).otherwise(0))

# StringIndexer for part_of_day
part_of_day_indexer = StringIndexer(inputCol='part_of_day', outputCol='part_of_day_index')
prediction_df = part_of_day_indexer.fit(prediction_df).transform(prediction_df)

# OneHotEncoder for part_of_day
part_of_day_encoder = OneHotEncoder(inputCols=['part_of_day_index'], outputCols=['part_of_day_encoded'])
prediction_df = part_of_day_encoder.fit(prediction_df).transform(prediction_df)

# Create separate columns for each part of the day
for part in part_of_day_list:
    column_name = f"at_{part.lower()}"
    prediction_df = prediction_df.withColumn(column_name, when(prediction_df.part_of_day == part, 1).otherwise(0))

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
prediction_df = assembler.transform(prediction_df)

prediction_df = prediction_df.drop('day_of_week_index', 'part_of_day_index', 'day_of_week_encoded', 'part_of_day_encoded', 'features', 'day_of_week', 'part_of_day')

## ---------------------------------------------
## 8.2 Training Linear Regression Model
# Split into train and test
test = prediction_df.filter(prediction_df["day"] >= 24)
train = prediction_df.filter(prediction_df["day"] < 24)

# Create the vector assembler and apply it to the datasets
vectorAssembler = VectorAssembler(inputCols= ["Temp", "Pressure", "Precip", "log_distance", "on_monday", "on_tuesday",
               "on_wednesday", "on_thursday", "on_friday", "on_saturday", "on_sunday",
               "at_morning", "at_afternoon", "at_evening", "at_night"], outputCol="features")
train = vectorAssembler.transform(train)
test = vectorAssembler.transform(test)

reg = LinearRegression(featuresCol='features', labelCol="log_trip_duration", regParam=0.1)
model = reg.fit(train)

# Generate predictions on the test dataset
test_predictions = model.transform(test)

# Calculate RMSE, MAE, and R2
evaluator = RegressionEvaluator(labelCol="log_trip_duration", predictionCol="prediction")

# Calculate RSME, MAE, and R-squared
rmse = evaluator.evaluate(test_predictions, {evaluator.metricName: "rmse"})
print("Linear Regression Model RMSE: {:.3f}".format(rmse))
mae = evaluator.evaluate(test_predictions, {evaluator.metricName: "mae"})
print("Linear Regression Model MAE: {:.3f}".format(mae))
r2 = evaluator.evaluate(test_predictions, {evaluator.metricName: "r2"})
print("Linear Regression Model R2: {:.3f}".format(r2))

## ---------------------------------------------
## 8.3 Training Random Forest Model
# Create a vector assembler to combine the input features into a single vector column
# Define the input feature columns

train = train.withColumnRenamed("features", "features_old")
test = test.withColumnRenamed("features", "features_old")


# Create a VectorAssembler
vectorAssembler = VectorAssembler(inputCols=input_feature_cols, outputCol="features")

# Assemble the features for training data
train_data = vectorAssembler.transform(train).select("features", "log_trip_duration")

# Assemble the features for test data
test_data = vectorAssembler.transform(test).select("features", "log_trip_duration")

# Create a RandomForestRegressor
rf = RandomForestRegressor(numTrees=200, maxDepth=10, seed=42, labelCol="log_trip_duration")

# Fit the RandomForestRegressor on the training data
model = rf.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model's performance
evaluator = RegressionEvaluator(labelCol="log_trip_duration", predictionCol="prediction")

# Calculate RSME, MAE, and R-squared
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print("Random Forest Model RMSE: {:.3f}".format(rmse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("Random Forest Model MAE: {:.3f}".format(mae))
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
print("Random Forest Model R2: {:.3f}".format(r2))

# Get the feature importances
feature_importances = model.featureImportances.toArray()

# Map feature importances to their corresponding feature names
feature_importance_map = {feature_name: feature_importance for feature_name, feature_importance in zip(input_feature_cols, feature_importances)}

# Sort the feature importances by their values in descending order
sorted_feature_importances = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importances
print("Feature Importances for the Random Forest Model:")
for feature, importance in sorted_feature_importances:
    print("{}: {:.3f}".format(feature, importance))