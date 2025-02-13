LOAD DATA INFILE '/tmp/customer_info.csv'
INTO TABLE `mlops`.`customer_info`
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/tmp/credit_score.csv'
INTO TABLE `mlops`.`credit_score`
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/tmp/credit_score_features.csv'
INTO TABLE `mlops`.`credit_score_features`
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;