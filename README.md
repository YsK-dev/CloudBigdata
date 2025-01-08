# Intro to Amazon EMR

# Overview

### What is EMR?


- _Elastic MapReduce_ is a managed clustered platform to run big data frameworks
<img width="1494" alt="Screenshot 2025-01-04 at 09 47 59" src="https://github.com/user-attachments/assets/95d53f1e-cbf9-40c0-8148-1004397d2ef3" />


    
    - Over 21 different frameworks available!
    - They are “out of box”, pre-installed
    - But really **Spark** is the one 
- It’s _on-demand_
    
    - Companies can save money by spinning up clusters up/down without having to setup their own Hadoop clusters
    - Cost savings using EMR managed auto-scaling
- Effective for OLAP (Online analytical processing) and Batch Processing jobs
    
    - At a minimum, you’ll need terabytes worth of data processing
    - Suitable for “embarrassingly parallel” tasks
- By default, EMR uses YARN (Yet Another Resource Negotiator) for Cluster resource management
    
    - Consists of **Resource Manager** and **Node Managers** to coordinate job scheduling
   <img width="518" alt="Screenshot 2023-08-27 at 2 06 43 PM" src="https://github.com/user-attachments/assets/7467629d-ef18-49d8-b039-3b951107f268" />

    
    YARN was introduced in Apache Hadoop 2.0 to help with cluster management
    

# Integration with AWS Services

There exists many AWS Services that integrate with EMR. The biggest ones to know are

![Simple Storage Service](https://github.com/user-attachments/assets/10417a8e-d757-4c7c-a2b8-d1e00cad19c4)


- **Amazon S3**
	- Acts as the persistent data store.
    - Decouple storage and compute. Avoids “cold data”
    -  EMRFS (EMR File System) enables seamless writing to S3.
![EC2](https://github.com/user-attachments/assets/feb629d4-03ea-4a0f-9760-cbde6433431e)

- **Amazon EC2**
    - Provision virtual machines for our worker nodes
    - Choose from various instance types (e.g., m5 for general processing, c5 for ML tasks, x1 for memory-intensive operations).
    - Fleet instances allow you to mix and match on-demand, spot, reserved instances

![Virtual Private Cloud](https://github.com/user-attachments/assets/22bbfbbc-7ab6-4903-b442-e1fb1006b467)


- **Amazon VPC**
    - Secure your networking environment
    - Limit your IP addresses, subnets, and traffic
    - Don’t put all your services in a single, default VPC
    - Shortly secures networking environments with subnets and traffic controls.
Other notable services:

- **AWS Step Functions** for workflow orchestration.
- **AWS Glue** for serverless ETL operations.
Initially, I work with AWS Glue for my project but later switched between various tools before settling on my current choice.

# Architecture for Amazon EMR

An EMR cluster comprises three node types:

1. **Primary Node**:
    
    - Coordinates data and task distribution.
    - Tracks task statuses and monitors cluster health.
2. **Core Node**:
    
    - Runs tasks and stores data in the ephemeral Hadoop Distributed File System (HDFS).
3. **Task Node** (optional):
    
    - Executes tasks but does not store data.
<img width="303" alt="Screenshot 2023-08-26 at 1 24 11 PM" src="https://github.com/user-attachments/assets/8d926907-2ac6-43d4-a80a-78383663cfe7" />

An EMR cluster consists of a primary node, core nodes and task nodes (optional).

EMR clusters are _**Elastic**_

- Horizontal* scaling using 1) manual or 2) auto-scaling
- EMR managed scaling allows you to set minimum/maximum cluster size
    - 💡 Pro tip: set number of on-demand instances for critical tasks, then provision the rest as spot instances

<img width="748" alt="Screenshot 2023-08-26 at 2 27 16 PM" src="https://github.com/user-attachments/assets/887d86d0-3e38-4c73-9832-958201cc458b" />


example of EMR managed auto-scaling interface

When you run an EMR job, you do this by triggering **Steps**. There are 3 ways to trigger EMR steps

- Management Console
- SSH into the primary node and manually run a task
- AWS Command Line Interface (recommended way)

EMR clusters run on 3 different compute platforms

- **Amazon EC2**: Default option. High performance and expensive
- **Amazon EKS**: Run light-weight applications
- **Serverless**: Small, low code applications


## PySpark Movie Recommendation Project

This project demonstrates a movie recommendation system implemented on Amazon EMR using PySpark.

### Code Explanation

#### Libraries and Setup

- **Flask**: Provides a lightweight web framework to create API endpoints for recommendations and feedback.
- **PySpark**: Leverages Spark’s MLlib for collaborative filtering.
- **boto3**: Enables interaction with AWS S3 for data loading.

```python
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import boto3

app = Flask(__name__)
spark = SparkSession.builder \
    .appName("MovieRecommendationApp") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()
```

#### Data Loading

- Loads movie and ratings datasets from an S3 bucket using `spark.read.csv`.
- Replace `BUCKET_NAME`, `MOVIES_FILE`, and `RATINGS_FILE` with your S3 paths.

```python
BUCKET_NAME = 'your-s3-bucket-name'
MOVIES_FILE = 'path/to/movies.csv'
RATINGS_FILE = 'path/to/ratings.csv'

movies = spark.read.csv(f"s3a://{BUCKET_NAME}/{MOVIES_FILE}", header=True, inferSchema=True)
ratings = spark.read.csv(f"s3a://{BUCKET_NAME}/{RATINGS_FILE}", header=True, inferSchema=True)
```

#### Training the Model

- Uses the **ALS (Alternating Least Squares)** algorithm from PySpark MLlib to train the recommendation model.
- Key parameters:
    - `maxIter`: Maximum number of iterations.
    - `regParam`: Regularization parameter.
    - `coldStartStrategy`: Handles missing data by dropping rows.

```python
def train_model():
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(ratings)
    return model

model = train_model()
```

#### Recommendation Endpoint

- Accepts a user ID and generates a list of recommended movies.
- Steps:
    1. Creates a DataFrame for the user.
    2. Uses `recommendForUserSubset` to get recommendations.
    3. Filters movie metadata from the `movies` dataset for output.

```python
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recommendations = model.recommendForUserSubset(user_df, 10)
    recommendations = recommendations.select("recommendations.movieId", "recommendations.rating").collect()

    movie_ids = [row.movieId for row in recommendations[0]["movieId"]]
    recommended_movies = movies.filter(col("movieId").isin(movie_ids)).collect()

    return jsonify({
        "user_id": user_id,
        "recommendations": [
            {"movieId": row.movieId, "title": row.title} for row in recommended_movies
        ]
    })
```

#### Feedback Endpoint

- Accepts user feedback (movie ID and rating) via a POST request.
- Updates the `ratings` dataset in memory.
- In production, feedback should be saved to a database.

```python
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_id = data['user_id']
    movie_id = data['movie_id']
    rating = data['rating']

    new_rating = [(user_id, movie_id, rating)]
    new_df = spark.createDataFrame(new_rating, ["userId", "movieId", "rating"])
    global ratings
    ratings = ratings.union(new_df)

    return jsonify({"message": "Feedback received successfully"})
```

#### Running the Application

- Runs the Flask app on port 5000.

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Workflow on EMR

1. **Cluster Setup**:
    - Provision an EMR cluster with Spark installed.
2. **Data Loading**:
    - Ensure datasets are uploaded to S3.
3. **Code Deployment**:
    - Deploy the application to the EMR cluster.
4. **API Interaction**:
    - Use the `/recommend` and `/feedback` endpoints to interact with the model.

This project demonstrates how to combine PySpark and AWS services to build scalable, distributed machine learning applications.
