import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import numpy as np
import re
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def main():
	
	train = pd.read_csv("data/train.csv")
	
	# add missing age values
	train["Age"] = train["Age"].fillna(train["Age"].median())
	
	# convert sex to numeric
	train.loc[train["Sex"] == "male", "Sex"] = 0
	train.loc[train["Sex"] == "female", "Sex"] = 1
	
	# covert emabraked column to numeric
	train["Embarked"] = train["Embarked"].fillna("S")
	train.loc[train["Embarked"] == "S", "Embarked"] = 0
	train.loc[train["Embarked"] == "C", "Embarked"] = 1
	train.loc[train["Embarked"] == "Q", "Embarked"] = 2
	
	#add new features 
	train["FamilySize"] = train["SibSp"] + train["Parch"]
	train["NameLength"] = train["Name"].apply(lambda x: len(x))
	titles = train["Name"].apply(get_title)
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
	for k,v in title_mapping.items():
		titles[titles == k] = v
	
	train["Title"] = titles
	lr = LinearRegression()
	
	# The columns we'll use to predict the target
	predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"]
	
	#choose best features
	selector = SelectKBest(f_classif, k=5)
	selector.fit(train[predictors], train["Survived"])
	scores = -np.log10(selector.pvalues_)

	plt.bar(range(len(predictors)), scores)
	plt.xticks(range(len(predictors)), predictors, rotation='vertical')
	plt.show()

	# Pick only the four best features.
	predictors = ["Pclass", "Sex", "Fare", "Title"]
	
	kf = kf = KFold(train.shape[0], n_folds=3, random_state=1)

	preds =[]
	for train_indices, test_indices in kf:
		train_predictors = train[predictors].iloc[train_indices]
		train_target = train["Survived"].iloc[train_indices]
		lr.fit(train_predictors, train_target)
		test_preds = lr.predict(train[predictors].iloc[test_indices])
		preds.append(test_preds)
	
	preds = np.concatenate(preds, axis=0)
	preds[preds > 0.5] = 1
	preds [preds <= 0.5] =0 
	
	accuracy = [1 if preds[i]==train["Survived"][i] else 0 for i in range(0, len(preds))]
	print "linear regression: " + str(sum(accuracy)/float(len(preds)))
	
	log_r = LogisticRegression(random_state=1)
	scores = cross_validation.cross_val_score(log_r, train[predictors], train["Survived"], cv=3)
	print "logistic regression: " + str(scores.mean())
	
	rf = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
	scores = cross_validation.cross_val_score(rf, train[predictors], train["Survived"], cv=kf)
	print "random forests: " + str(scores.mean())
	
	test = pd.read_csv("data/test.csv")
	# fill in missing values in test set
	test["Age"] = test["Age"].fillna(test["Age"].median())
	test["Fare"] = test["Fare"].fillna(test["Fare"].median())
	
	# convert sex to numeric
	test.loc[test["Sex"] == "male", "Sex"] = 0
	test.loc[test["Sex"] == "female", "Sex"] = 1
	
	# covert emabraked column to numeric
	test["Embarked"] = test["Embarked"].fillna("S")
	test.loc[test["Embarked"] == "S", "Embarked"] = 0
	test.loc[test["Embarked"] == "C", "Embarked"] = 1
	test.loc[test["Embarked"] == "Q", "Embarked"] = 2
	
	# add new features to test set
	test["FamilySize"] = test["SibSp"] + test["Parch"]
	test["NameLength"] = test["Name"].apply(lambda x: len(x))
	titles = test["Name"].apply(get_title)
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
	for k,v in title_mapping.items():
		titles[titles == k] = v	
	test["Title"] = titles
	
	rf.fit(train[predictors], train["Survived"])
	preds = rf.predict(test[predictors])
	
	submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
	submission = submission.set_index("PassengerId")
	submission.to_csv("my_submission.csv")

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

if __name__ == "__main__":
	main()