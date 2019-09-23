# Fishing The Phish: Data science and Machine learning driven phishing link detection model

 Winner of "Akamai Security Challenge" and "MITRE Cyber Phishing Challenge" at ShellHacks 2019. 

 The world has increasingly become dependent on the internet for a majority of its tasks and has become an indispensable part of our daily lives. Therefore it comes as no surprise that many institutions implement various security measures to mitigate and also prevent the cyber-security attacks. Phishing is one of the many security attacks of this spectrum. In the above attack, a user wants to gain customer-sensitive information, by masquerading as a legitimate website. 

 We completed this project as a part of MITRE cyber phishing challenge at ShellHacks 2019. We were given a small dataset of less than 5000 samples which had URLs, and lifetime details (3 features) of certain websites. Our goal was to perform exploratory data analysis to explain feature relevance, extract more features through feature engineering and train machine learning models to detect phishing links in an unseen test dataset.

 We used an ensemble of machine learning models such as Random Forests, K-nearest neighbors, Decision trees, Logistic regression and Adaptive Boosting with decision trees to build our model and developed an algorithm for dynamic optimal model selection for two different set of features and combined the results from various models to build final phishing detection model. Our approach resulted in an F1-score of more than 92% on an unseen test dataset that MITRE used to judge a number of submissions in their track. 
