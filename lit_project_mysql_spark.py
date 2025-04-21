# STEP 1: INSTALL DEPENDENCIES
!apt-get install openjdk-11-jdk-headless -qq > /dev/null
!wget -q https://downloads.mysql.com/archives/get/p/3/file/mysql-connector-java-8.0.32.tar.gz
!tar -xzf mysql-connector-java-8.0.32.tar.gz
!pip install pyspark -q

import os
import pandas as pd
from pyspark.sql import SparkSession

# STEP 2: SET ENV VARIABLES FOR JAVA AND MYSQL CONNECTOR
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
jdbc_driver_path = "/content/mysql-connector-java-8.0.32/mysql-connector-java-8.0.32.jar"

# STEP 3: CREATE SPARK SESSION
spark = SparkSession.builder \
    .appName("LIT Project - MySQL") \
    .config("spark.jars", jdbc_driver_path) \
    .getOrCreate()

# STEP 4: MYSQL CONNECTION DETAILS (change as per your DB)
mysql_host = "localhost"        # or use your IP or domain
mysql_port = "3306"
mysql_db = "your_db_name"
mysql_user = "your_username"
mysql_password = "your_password"
mysql_table = "your_table_name"

# STEP 5: READ DATA FROM MYSQL TO PYSPARK
jdbc_url = f"jdbc:mysql://{mysql_host}:{mysql_port}/{mysql_db}"

df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .option("dbtable", mysql_table) \
    .option("user", mysql_user) \
    .option("password", mysql_password) \
    .load()

df.show()
