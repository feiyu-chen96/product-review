from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .config("spark.master", "local") \
    .getOrCreate()

data_df = spark \
    .read.parquet("output/reviews_Musical_Instruments_5_preprocessed.parquet")

tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
hashing_term_freq = \
    HashingTF(inputCol="words", outputCol="featuresRaw", numFeatures=5000)
inv_doc_freq = IDF(inputCol="featuresRaw", outputCol="features", minDocFreq=5)
indexer = StringIndexer(inputCol = "category", outputCol = "label")

pipeline = Pipeline(stages=[
    tokenizer, 
    remover,
    hashing_term_freq,
    inv_doc_freq, 
    indexer]
)

pipeline_fitted = pipeline.fit(data_df)
data_prepared_df = pipeline_fitted.transform(data_df)

train_df, test_df = data_prepared_df.randomSplit([0.8, 0.2], seed=42)
log_reg = LogisticRegression(
    featuresCol="features", labelCol="label", predictionCol="prediction",
    maxIter=20, regParam=0.3, elasticNetParam=0
)
log_reg_fitted = log_reg.fit(train_df)

test_pred_df = log_reg_fitted.transform(test_df)
test_pred_df.show()
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="label", metricName="f1"
)
f1 = evaluator.evaluate(test_pred_df)
print("\n\nF1 score: " + str(f1) + "\n\n")
