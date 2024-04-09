# pip install sqlalchemy==1.3.9
# pip install -q pandas==1.1.5
# pip install python
# pip install ipython-sql

%load_ext sql

import csv, sqlite3
import pandas as pd

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

%sql sqlite:///my_data1.db

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")

chunksize = 1000  # Número de filas por lote
for chunk in pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv", chunksize=chunksize):
    chunk.to_sql("SPACEXTBL", con, if_exists='append', index=False)

%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null

%sql PRAGMA table_info(SPACEXTBL); # Para ver las columnas

uniques_sites = %sql SELECT DISTINCT Launch_Site FROM SPACEXTBL;
df_uniq_sites = pd.DataFrame(uniques_sites, columns=['Launch_Site'])
df_uniq_sites

%sql SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' LIMIT 5;
