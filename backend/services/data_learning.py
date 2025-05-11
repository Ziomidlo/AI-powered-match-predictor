import pandas as pd
import seaborn as seaborn
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy.stats as stats



cleaned_data_folder = "../cleaned_data/"
merged_matches = pd.read_csv(cleaned_data_folder + "LearningMatchesData.csv")
upcoming_matches = pd.read_csv(cleaned_data_folder + "LearningUpcomingMatches.csv")

#Correlation Analysis
corr, p_value = spearmanr(merged_matches["h2h_home_wins"], merged_matches["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(merged_matches["h2h_home_wins"], merged_matches["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

#Training models

X = merged_matches[['home_xG_avg', 'away_xG_avg','goal_diff_delta', 'home_team_strength', 'away_team_strength', 
                   'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals', 
                   'venue_home_wins','venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals', 'venue_impact_diff',
                   'home_defense_strength', 'away_defense_strength', 'home_offensive_strength', 'away_offensive_strength', 'home_passing_and_control_strength', 
                   'away_passing_and_control_strength', 'home_team_power_index', 'away_team_power_index', 'home_avg_points_per_game', 'away_avg_points_per_game',
                   'home_avg_position_ovr', 'away_avg_position_ovr', 'home_weight_points', 'away_weight_points']]

XUP = upcoming_matches[X.columns]

y = merged_matches["result_numeric"]

tscv_tuning = TimeSeriesSplit(n_splits= 5)

#Logistic Regression tuning
def logisticRegressionTuning():
   print("--- Tuning Logistic Regression ---")

   lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter= 5000))

   param_distributions_lr = {
      'logisticregression__C': stats.loguniform(0.001, 100),
      'logisticregression__penalty' : ['l2', 'l1'],
      'logisticregression__solver' : ['lbfgs', 'saga']
   }

   random_search_lr = RandomizedSearchCV(lr_pipeline, param_distributions_lr, n_iter=100, cv=tscv_tuning, scoring='accuracy', random_state=42,  n_jobs=-1)

   print("Starting Tuning Logistic Regression...")
   random_search_lr.fit(X, y)

   print("\n--- Results of Tuning  Logistic Regression ---")
   print("Best parameters:", random_search_lr.best_params_)
   print("Best average Result CV (Accuracy): ", random_search_lr.best_score_)

   best_lr_model = random_search_lr.best_estimator_
   print("Best Model:", best_lr_model)

#Logistic Regression 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model =  make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=5000, random_state=42, C=0.008))
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Random Forest Classifier Tuning
def randomForestClassifierTuning():
   print("\n--- Tuning Random Forest Classifier ---")

   rf_clf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced'))

   param_distributions_rf_clf = {
      'randomforestclassifier__n_estimators' : stats.randint(100, 1000),
      'randomforestclassifier__max_depth' : [None] + list(stats.randint(5,30).rvs(10)),
      'randomforestclassifier__min_samples_split': stats.randint(2, 50),
      'randomforestclassifier__min_samples_leaf': stats.randint(1, 30),
      'randomforestclassifier__max_features': ['sqrt', 'log2', None], 
      'randomforestclassifier__criterion': ['gini', 'entropy'],
   }

   print("Startning tuning Random Forest Classifier...")
   random_search_rf_clf = RandomizedSearchCV(rf_clf_pipeline, param_distributions_rf_clf, n_iter=100, scoring="accuracy", random_state=42, n_jobs=-1)
   random_search_rf_clf.fit(X, y)

   print("\n--- Results of Tuning Random Forest Classifier ---")
   print("Best parameters:", random_search_rf_clf.best_params_)
   print("Best average Result CV (Accuracy):", random_search_rf_clf.best_score_)

   best_rf_clf_model = random_search_rf_clf.best_estimator_
   print("Best model:", best_rf_clf_model)

#Random Forest Classifier scores
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', n_estimators=445, random_state=42, class_weight='balanced', max_depth=22, min_samples_leaf=4, min_samples_split=20))
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier - Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#Linear Regression

#Target for linear reggresion
y_home_goals = merged_matches["Home Goals"]
y_away_goals = merged_matches["Away Goals"]

# Split for Home Goals
X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)

# Split for Away Goals
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)


home_goals_model = make_pipeline(StandardScaler(), LinearRegression())
away_goals_model = make_pipeline(StandardScaler(), LinearRegression())
home_goals_model.fit(X_train_home, y_train_home)
away_goals_model.fit(X_train_away, y_train_away)

y_pred_home = home_goals_model.predict(X_test_home)
y_pred_away = away_goals_model.predict(X_test_away)

print("Linear Regression - Home Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_home, y_pred_home))
print("MSE:", mean_squared_error(y_test_home, y_pred_home))

print("\nLinear Regression - Away Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_away, y_pred_away))
print("MSE:", mean_squared_error(y_test_away, y_pred_away))


#Random Forest Regressor Tuning


def randomForestRegressorTuning():
   print("--- Tuning for Random Forest Regressor ---")
   rf_reg_pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

   param_distributions_rf_reg = {
      'randomforestregressor__n_estimators': stats.randint(100, 1000),
      'randomforestregressor__max_depth': [None] + list(stats.randint(5, 30).rvs(10)),
      'randomforestregressor__min_samples_split': stats.randint(2, 50),
      'randomforestregressor__min_samples_leaf': stats.randint(1, 30),
      'randomforestregressor__max_features': ['sqrt', 'log2', None],
      'randomforestregressor__criterion': ['squared_error', 'absolute_error'] 
   }

   mae_scorer = make_scorer(mean_absolute_error)

   print("Starting Tuning for Random Forest Regressor...")
   random_search_rf_reg = RandomizedSearchCV(rf_reg_pipeline, param_distributions_rf_reg, n_iter=100, cv=tscv_tuning, scoring=mae_scorer, random_state=42, n_jobs=-1 )

   print("Tuning for Home Goals...")
   random_search_rf_reg_home = random_search_rf_reg.fit(X, y_home_goals)

   print("\n--- Results of Tuning Random Forest Regressor (Home Goals) ---")
   print("Best parameters:", random_search_rf_reg_home.best_params_)
   print("Best average result CV (MAE):", random_search_rf_reg_home.best_score_)

   best_rf_reg_home_model = random_search_rf_reg_home.best_estimator_
   print("Best model:", best_rf_reg_home_model)

   print("Tuning for Away Goals...")
   random_search_rf_reg_away = random_search_rf_reg.fit(X, y_away_goals)

   print("\n--- Results of Tuning Random Forest Regressor (Away Goals) ---")
   print("Best parameters:", random_search_rf_reg_away.best_params_)
   print("Best average result CV (MAE):", random_search_rf_reg_away.best_score_)

   best_rf_reg_away_model = random_search_rf_reg_away.best_estimator_
   print("Best model:", best_rf_reg_away_model)


#Random Forest Regressor Scores

home_goals_rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=838, random_state=42, max_depth=23, max_features='log2', min_samples_leaf=27, min_samples_split=36))
away_goals_rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=956, random_state=42, max_depth=11, max_features='log2', min_samples_leaf=29, min_samples_split=45))

home_goals_rf.fit(X_train_home, y_train_home)
away_goals_rf.fit(X_train_away, y_train_away)

y_pred_home_rf = home_goals_rf.predict(X_test_home)
y_pred_away_rf = away_goals_rf.predict(X_test_away)

print("Random Forest Regressor - Home Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_home, y_pred_home_rf))
print("MSE:", mean_squared_error(y_test_home, y_pred_home_rf))

print("\nRandom Forest Regressor - Away Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_away, y_pred_away_rf))
print("MSE:", mean_squared_error(y_test_away, y_pred_away_rf))


importances = model.named_steps['logisticregression'].coef_[0]
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))


rf_model_step = rf_model.named_steps['randomforestclassifier']
importancesRF = rf_model_step.feature_importances_
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance RF': importancesRF})
print(feature_importance_rf.sort_values(by='Importance RF', ascending=False))

#logisticRegressionTuning()
#randomForestRegressorTuning()
#randomForestClassifierTuning()

#Upcoming matches prediction

probs = model.predict_proba(XUP)
predicted_classes = model.predict(XUP)

home_xg = home_goals_model.predict(XUP)
away_xg = away_goals_model.predict(XUP)

predicted_matches_df = pd.DataFrame({
   'Home Id' : upcoming_matches['Home Id'],
   "Home": upcoming_matches['Home'],
   "Away Id": upcoming_matches['Away Id'],
   "Away": upcoming_matches['Away'],
   "Predicted Result" : predicted_classes,
   "Home Win %" : (probs[:, 2] * 100).round(2),
   "Draw %" : (probs[:, 1] * 100).round(2),
   "Away Win %" : (probs[:, 0] * 100).round(2),
   "Home xG" : home_xg.round(2),
   "Away xG" : away_xg.round(2)
})

predicted_matches_df.to_csv(cleaned_data_folder + "PredictedUpcomingMatches.csv", index=False)