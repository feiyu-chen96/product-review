from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, rand
from pyspark.sql.types import StringType

spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .config("spark.master", "local") \
    .getOrCreate()

def category_review(votes):
    score = votes[0]/votes[1]
    if score >= 0.8:
        return "good"
    elif score <= 0.2:
        return "bad"
    else:
        return "soso"

category_review_udf = udf(category_review, StringType())

review_df = spark \
    .read.json("data/reviews_Musical_Instruments_5.json") \
    .where(col("helpful")[1] >= 5) \
    .withColumn("category", category_review_udf("helpful")) \
    .select("reviewText", "category") \
    .cache()

review_df_good = review_df.where(col("category") == "good").cache()
review_df_bad = review_df.where(col("category") == "bad").cache()
review_df_soso = review_df.where(col("category") == "soso").cache()

n_good = review_df_good.count()
n_bad = review_df_bad.count()
n_soso = review_df_soso.count()

review_df_good_sample = \
    review_df_good.sample(withReplacement=False, fraction=n_bad/n_good)
review_df_soso_sample = \
    review_df_soso.sample(withReplacement=False, fraction=n_bad/n_soso*3)

review_df_preprocessed = review_df_bad \
    .unionAll(review_df_good_sample) \
    .unionAll(review_df_soso_sample)

review_df_preprocessed.write.parquet("output/reviews_Musical_Instruments_5_preprocessed.parquet")

review_preprocessed_df = spark \
    .read.parquet("output/reviews_Musical_Instruments_5_preprocessed.parquet")

review_preprocessed_df.show()
review_preprocessed_df.groupBy("category").count().show()
