// Written by: Nicholas Cockcroft
// Date: August 1, 2018
// Assignment: Assignment #7

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.rdd._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}


// Open file and store the data after the second column into an rdd vector
val file = sc.textFile("/home/nick/input/wdbc.data")
val rawData = file.map(x=> x.split(","))
val parsedData = rawData.map(x=>Vectors.dense(x.drop(2).map(s=>s.toDouble))) // Putting all the data into the vector EXCEPT the first two column

// Training the cluster with the vector data
val kmeans = new KMeans()
kmeans.setK(2)
val model = kmeans.run(parsedData)

val cluster = rawData.map(x=> (x(0).toString,x(1).toString,model.predict(Vectors.dense(x.drop(2).map(s=>s.toDouble))))) // Creating a new map with the patient id as the key, diagnosis as a value, and also the vector of other attributes as a value

val malignantisone = cluster.map(x=>(x._1,if(x._2 == "M"){1} else {0},x._3)) // Making a new map with the patient id as the key, converting the diagnosis to either 1 or 0 depending if it was M or B, and then keeping the vector of data

val falseprediction = malignantisone.filter(x=>x._2 != x._3)

falseprediction.saveAsTextFile("/home/nick/out/Assignment7")

val malignantiszero = cluster.map(x=>(x._1,if(x._2 == "M"){0} else {1},x._3)) // Making a new map with the patient id as the key, converting the diagnosis to either 1 or 0 depending if it was M or B, and then keeping the vector of data

val falseprediction = malignantiszero.filter(x=>x._2 != x._3)

falseprediction.saveAsTextFile("/home/nick/out/Assignment7")
