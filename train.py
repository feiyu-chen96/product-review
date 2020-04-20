from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.stem.lancaster import LancasterStemmer

spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .config("spark.master", "local") \
    .getOrCreate()

data_df = spark \
    .read.parquet("output/reviews_Musical_Instruments_5_preprocessed.parquet")

# tokenize words
tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
data_df_tokenized = tokenizer.transform(data_df)

# remove stop words
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
data_df_filtered = remover.transform(data_df_tokenized)

# stemming
stemmer = LancasterStemmer()
stemmer_udf = udf(
    lambda tokens: [stemmer.stem(token) for token in tokens], 
    ArrayType(StringType())
)
data_f_stemmed = data_df_filtered.withColumn("wordsStemmed", stemmer_udf("words"))

# hashing term frequency 
hashing_term_freq = \
    HashingTF(inputCol="wordsStemmed", outputCol="featuresRaw", numFeatures=5000)
data_df_tf = hashing_term_freq.transform(data_f_stemmed)

# inverse document frequency
inv_doc_freq = IDF(inputCol="featuresRaw", outputCol="features", minDocFreq=5)
inv_doc_freq_fitted = inv_doc_freq.fit(data_df_tf)
data_df_tfidf = inv_doc_freq_fitted.transform(data_df_tf)

# encode classes
indexer = StringIndexer(inputCol="category", outputCol="label")
indexer_fitted = indexer.fit(data_df_tfidf)
data_prepared_df = indexer_fitted.transform(data_df_tfidf)

# train-test split
train_df, test_df = data_prepared_df.randomSplit([0.8, 0.2], seed=42)

# train
log_reg = LogisticRegression(
    featuresCol="features", labelCol="label", predictionCol="prediction",
    maxIter=20, regParam=0.3, elasticNetParam=0
)
log_reg_fitted = log_reg.fit(train_df)

# predict
test_pred_df = log_reg_fitted.transform(test_df)
test_pred_df.show()
evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction", labelCol="label", metricName="f1"
)
f1 = evaluator.evaluate(test_pred_df)
print("\n\nF1 score: " + str(f1) + "\n\n")
