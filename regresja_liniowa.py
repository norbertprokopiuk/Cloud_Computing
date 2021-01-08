from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
import boto3

'''
I use tutorial on page:
https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning
'''


def convertColums(df,names,newType):
    for name in names:
        df=df.withColumn(name,df[name].cast(newType))
    return df


# access to my s3 bucket and download data
s3 = boto3.resource(
    service_name='s3',
    region_name='eu-central-1',
    aws_access_key_id='your acces key ',
    aws_secret_access_key='your secret access key'
)
s3.Bucket('s3-projekt').download_file(Key='insurance.csv', Filename='insurance.csv')


spark=SparkSession.builder\
    .master("spark://ip-172-31-45-172.eu-central-1.compute.internal:7077")\
    .appName("linear regression model")\
    .getOrCreate()

sc=spark.sparkContext
rdd=sc.textFile('insurance.csv')
rdd=rdd.map(lambda line: line.split(","))
rdd.take(2)

#full dataset have numeric and string data. I choose only numeric
df=rdd.map(lambda line: Row(age=line[0],
                            #sex=line[1],
                            bmi=line[2],
                            children=line[3],
                            #smoker=line[4],
                            #region=line[5],
                            charges=line[6])).toDF()
#for now I have data frame but data are a string
#i have to convert it into float

colums=['age','bmi','children','charges']

df=convertColums(df,colums,FloatType())
# after conversion data are a float

# at this moment i have to prepare data to linear regression

input_data=df.rdd.map(lambda x:(x[0],DenseVector(x[1:])))

df=spark.createDataFrame(input_data,["label","features"])

standardScaler=StandardScaler(inputCol="features",outputCol="features_scaled")
scaler=standardScaler.fit(df)

scaled_df=scaler.transform(df)

#machine learning model
#split data
train_data, test_data=scaled_df.randomSplit([0.8,0.2],seed=1234)

#linear regression
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
linearModel=lr.fit(train_data)

predicted=linearModel.transform(test_data)

predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

predictionAndLabel = predictions.zip(labels).collect()
print("results")
print("first 5 elements of prediction")
print(predictionAndLabel[:5])
print("coefficients:" +str(linearModel.coefficients))
print("intercept: "+str(linearModel.intercept))
print("RMSE: "+str(linearModel.summary.rootMeanSquaredError))
print("R2: "+str(linearModel.summary.r2))
spark.stop()