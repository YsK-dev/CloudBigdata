from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Flask app
app = Flask(__name__)

# Initialize Spark Session
spark = SparkSession.builder.appName("MovieRecommendations").getOrCreate()

# Global variables to hold model and results
model = None


def train_model(data_source: str):
    global model

    # Load data
    df = spark.read.option("header", "true").csv(data_source)
    df = df.select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("float")
    )

    # Split data into train and test
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    # Train ALS model with predefined parameters
    als = ALS(
        maxIter=10, regParam=0.1, rank=50,
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop", seed=42
    )
    model = als.fit(train)


@app.route('/train', methods=['POST'])
def train():
    data = request.json
    data_source = data.get("data_source")

    if not data_source:
        return jsonify({"error": "data_source is required"}), 400

    try:
        train_model(data_source)
        return jsonify({"message": "Model training completed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend', methods=['GET'])
def recommend():
    if model is None:
        return jsonify({"error": "Model has not been trained yet"}), 400

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        user_id = int(user_id)
        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        recommendations = model.recommendForUserSubset(user_df, 10)
        recommendations = recommendations.select("recommendations.movieId", "recommendations.rating").collect()

        movie_ids = [row.movieId for row in recommendations[0]["movieId"]]
        return jsonify({"user_id": user_id, "recommended_movie_ids": movie_ids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)
