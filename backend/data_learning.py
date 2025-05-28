from random import randint, uniform
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy.stats as stats
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

trained_ml_models = {}

TRAINING_FEATURE_COLUMNS= ['home_xG_avg', 'away_xG_avg','goal_diff_delta', 'home_team_strength', 'away_team_strength', 
                     'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals', 
                     'venue_home_wins','venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals',
                     'home_defense_strength', 'away_defense_strength', 'home_offensive_strength', 'away_offensive_strength', 'home_passing_and_control_strength', 
                     'away_passing_and_control_strength', 'home_team_power_index', 'away_team_power_index', 'home_avg_points_per_game', 'away_avg_points_per_game',
                     'home_avg_position_ovr', 'away_avg_position_ovr']

def train_models():
   
   global trained_ml_models

   cleaned_data_folder = "cleaned_data/"
   visualization_folder = "visualizations/"
   merged_matches = pd.read_csv(cleaned_data_folder + "LearningMatchesData.csv")
   upcoming_matches = pd.read_csv(cleaned_data_folder + "LearningUpcomingMatches.csv")
   season_stats = pd.read_csv(cleaned_data_folder + "LearningSeasonStatsData.csv")

   #Correlation Analysis
   corr, p_value = spearmanr(merged_matches["h2h_home_wins"], merged_matches["result_numeric"])
   print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
   pearson_corr, p_value = pearsonr(merged_matches["h2h_home_wins"], merged_matches["result_numeric"])
   print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

   #Training models

   X = merged_matches[['home_xG_avg', 'away_xG_avg','goal_diff_delta', 'home_team_strength', 'away_team_strength', 
                     'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals', 
                     'venue_home_wins','venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals',
                     'home_defense_strength', 'away_defense_strength', 'home_offensive_strength', 'away_offensive_strength', 'home_passing_and_control_strength', 
                     'away_passing_and_control_strength', 'home_team_power_index', 'away_team_power_index', 'home_avg_points_per_game', 'away_avg_points_per_game',
                     'home_avg_position_ovr', 'away_avg_position_ovr']]

   XUP = upcoming_matches[X.columns]

   y = merged_matches["result_numeric"]

   tscv_tuning = TimeSeriesSplit(n_splits= 5)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
      random_search_lr.fit(X_train, y_train)

      print("\n--- Results of Tuning  Logistic Regression ---")
      print("Best parameters:", random_search_lr.best_params_)
      print("Best average Result CV (Accuracy): ", random_search_lr.best_score_)

      best_lr_model = random_search_lr.best_estimator_
      print("Best Model:", best_lr_model)


   model = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', LogisticRegression(
         penalty='l1', 
         class_weight='balanced', 
         solver='saga', 
         max_iter=5000, 
         random_state=42, 
         C=1.146
      ))
   ])
   model.fit(X_train, y_train)


   y_pred = model.predict(X_test)
   print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))
   print("Classification Report: ", classification_report(y_test, y_pred))
   print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

   trained_ml_models['logistic_regression_classifier'] = model

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
      random_search_rf_clf.fit(X_train, y_train)

      print("\n--- Results of Tuning Random Forest Classifier ---")
      print("Best parameters:", random_search_rf_clf.best_params_)
      print("Best average Result CV (Accuracy):", random_search_rf_clf.best_score_)

      best_rf_clf_model = random_search_rf_clf.best_estimator_
      print("Best model:", best_rf_clf_model)

   #Random Forest Classifier scores

   rf_model = Pipeline([
      ('scaler', StandardScaler()), 
      ('model',RandomForestClassifier(
         criterion='gini', 
         n_estimators=180, 
         max_features=None,
         random_state=42, 
         class_weight='balanced', 
         max_depth=9, 
         min_samples_leaf=3, 
         min_samples_split=15))
   ])
   
   rf_model.fit(X_train, y_train)

   y_pred_rf = rf_model.predict(X_test)

   print("Random Forest Classifier - Accuracy: ", accuracy_score(y_test, y_pred_rf))
   print("Classification Report:\n", classification_report(y_test, y_pred_rf))
   print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

   trained_ml_models['random_forest_classifier'] = rf_model

   #XGBoost Classifier

   def xgboost_classifier_tuning():
      pipeline_xgb_classifier = make_pipeline(StandardScaler(), XGBClassifier(objective='multi:softmax', eval_metric='mlogloss',  random_state=42))

      params_xgb_classifier = {
         'xgbclassifier__n_estimators': stats.randint(50, 300),
         'xgbclassifier__learning_rate': stats.uniform(0.01, 0.2),
         'xgbclassifier__max_depth' : stats.randint(3, 10),
         'xgbclassifier__subsample' : stats.uniform(0.6, 0.4),
         'xgbclassifier__colsample_bytree' : stats.uniform(0.6, 0.4),
         'xgbclassifier__gamma' : stats.uniform(0, 0.5),
         'xgbclassifier__lambda' : stats.uniform(0.5, 1.5),
         'xgbclassifier__alpha' : stats.uniform(0, 0.5)
      }

      xgb_search = RandomizedSearchCV(pipeline_xgb_classifier, params_xgb_classifier, n_iter=20, cv=tscv_tuning, scoring='accuracy', random_state=42,  n_jobs=-1, verbose=1)
      y_xgb = y_train + 1
      y_xgb_test = y_test + 1 
      xgb_search.fit(X_train, y_xgb)  
      print("\nXGBoost Classifier - Best Parameters: ", xgb_search.best_params_)
      print("XGBoost Classifier - The best accurency CV: ", xgb_search.best_score_)
      y_pred_xgb = xgb_search.predict(X_test)
      print("XGBoost Classifier - Classification raport in test values:\n", classification_report(y_xgb_test, y_pred_xgb))
   
   pipeline_xgb_classifier = Pipeline([
      ('scaler',StandardScaler()), 
      ('model', XGBClassifier(
         alpha=0.176, 
         colsample_bytree = 0.722, 
         gamma = 0.082, 
         learning_rate=0.107, 
         max_depth = 3, 
         n_estimators =100, 
         subsample = 0.708, 
         objective='multi:softmax', 
         eval_metric='mlogloss', 
         random_state=42))
   ])
   
   y_xgb = y_train + 1
   y_xgb_test = y_test + 1 
   pipeline_xgb_classifier.fit(X_train, y_xgb) 
   y_pred_xgb = pipeline_xgb_classifier.predict(X_test) 
   print("XGBoost Classifier - Accuracy :", accuracy_score(y_xgb_test, y_pred_xgb))
   print("Classificaion Report:\n", classification_report(y_xgb_test, y_pred_xgb))
   print("Confusion Matrix: ", confusion_matrix(y_xgb_test, y_pred_xgb))

   trained_ml_models['xgboost_classifier'] = pipeline_xgb_classifier
   
   #Support Vector Machine (SVC)
   def svc_classifier_tuning():
      pipeline_svc_classifier = make_pipeline(StandardScaler(), SVC(random_state=42, probability=True))

      params_svc_classifier = {
         'svc__C' : stats.uniform(0.1, 10),
         'svc__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
         'svc__gamma' : stats.uniform(0.001, 1),
         'svc__degree' : stats.randint(2,4)
      }

      svc_search = RandomizedSearchCV(
         pipeline_svc_classifier, 
         params_svc_classifier, 
         n_iter=20, 
         cv=tscv_tuning,
         scoring='accuracy',
         random_state=42, 
         n_jobs=-1,
         verbose=1)
      
      svc_search.fit(X_train, y_train)
      print("\nSVC Classifier - Best Parameters: ", svc_search.best_params_)
      print("SVC CLassifier - The best average accuracy CV:", svc_search.best_score_)

   pipeline_svc_classifier = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', SVC(
         random_state= 42,
         probability=True,
         C=2.06,
         degree=2,
         gamma=0.962,
         kernel='linear'))
   ])
   pipeline_svc_classifier.fit(X_train, y_train)
   y_pred_svc = pipeline_svc_classifier.predict(X_test)
   print("\nSupport Vector Classifier - Accuracy: ", accuracy_score(y_test, y_pred_svc))
   print("Classificaion Report:\n", classification_report(y_test, y_pred_svc))
   print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_svc))

   trained_ml_models['support_vector_classifier'] = pipeline_svc_classifier

   #Linear Regression (Ridge)

   #Target for linear reggresion
   y_home_goals = merged_matches["Home Goals"]
   y_away_goals = merged_matches["Away Goals"]

   # Split for Home Goals
   X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)

   # Split for Away Goals
   X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)


   home_goals_model = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', Ridge(alpha=1.0))
      ])
   away_goals_model = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', Ridge(alpha=1.0))
      ])
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

   trained_ml_models['linear_regressor_home'] = home_goals_model
   trained_ml_models['linear_regressor_away'] = away_goals_model


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
      random_search_rf_reg_home = random_search_rf_reg.fit(X_train_home, y_train_home)

      print("\n--- Results of Tuning Random Forest Regressor (Home Goals) ---")
      print("Best parameters:", random_search_rf_reg_home.best_params_)
      print("Best average result CV (MAE):", random_search_rf_reg_home.best_score_)

      best_rf_reg_home_model = random_search_rf_reg_home.best_estimator_
      print("Best model:", best_rf_reg_home_model)

      print("Tuning for Away Goals...")
      random_search_rf_reg_away = random_search_rf_reg.fit(X_train_away, y_train_away)

      print("\n--- Results of Tuning Random Forest Regressor (Away Goals) ---")
      print("Best parameters:", random_search_rf_reg_away.best_params_)
      print("Best average result CV (MAE):", random_search_rf_reg_away.best_score_)

      best_rf_reg_away_model = random_search_rf_reg_away.best_estimator_
      print("Best model:", best_rf_reg_away_model)


   #Random Forest Regressor Scores

   home_goals_rf = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', RandomForestRegressor(
      n_estimators=260,
      criterion='absolute_error',
      random_state=42, 
      max_depth=None, 
      max_features=None, 
      min_samples_leaf=29, 
      min_samples_split=22))
   ])
   
   away_goals_rf = Pipeline([
      ('scaler', StandardScaler()), 
      ('model', RandomForestRegressor(
         n_estimators=838, 
         criterion='absolute_error',
         random_state=42, 
         max_depth=21, 
         max_features='log2', 
         min_samples_leaf=27, 
         min_samples_split=36))
   ])

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

   trained_ml_models['random_forest_regressor_home'] = home_goals_rf
   trained_ml_models['random_forest_regressor_away'] = away_goals_rf

   #XGBoost Regressor
   def xgboost_regressor_tuning():
      pipeline_xgb_regressor = make_pipeline(StandardScaler(), XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42))

      params_xgb_regressor = {
         'xgbregressor__n_estimators': stats.randint(50, 300),
         'xgbregressor__learning_rate': stats.uniform(0.01, 0.2),
         'xgbregressor__max_depth': stats.randint(3, 10),
         'xgbregressor__subsample': stats.uniform(0.7, 0.9)
      }

      xgb_regressor_search_home = RandomizedSearchCV(
         pipeline_xgb_regressor,
           params_xgb_regressor, 
           cv=tscv_tuning, 
           scoring='neg_mean_absolute_error', 
           random_state=42, 
           n_jobs=-1, 
           verbose=1)
      xgb_regressor_search_home.fit(X_train_home, y_train_away)
      print("\nXGBoost Regressor (Home Goals) - Best Parameters", xgb_regressor_search_home.best_params_)
      print("XGBoost Regressor (Home Goals) - The Best average MAE CV: ", -xgb_regressor_search_home.best_score_)

      xgb_regressor_search_away = RandomizedSearchCV(
         pipeline_xgb_regressor, 
         params_xgb_regressor, 
         cv=tscv_tuning, 
         scoring='neg_mean_absolute_error', 
         random_state=42, 
         n_jobs=-1, 
         verbose=1)
      
      xgb_regressor_search_away.fit(X_train_home, y_train_home)
      print("\nXGBoost Regressor (Away Goals) - Best Parameters", xgb_regressor_search_away.best_params_)
      print("XGBoost Regressor (Away Goals) - The Best average MAE CV: ", -xgb_regressor_search_away.best_score_)

   home_goals_xgb = Pipeline([
      ('scaler', StandardScaler()),
      ('model', XGBRegressor(
         learning_rate = 0.216,
         max_depth = 7,
         n_estimators = 149,
         subsample = 0.8286
      ))
   ])

   away_goals_xgb = Pipeline([
      ('scaler', StandardScaler()),
      ('model', XGBRegressor(
         learning_rate = 0.216,
         max_depth = 7,
         n_estimators = 149,
         subsample = 0.8286
      ))
   ])
   home_goals_xgb.fit(X_train_home, y_train_home)
   away_goals_xgb.fit(X_train_away, y_train_away)

   y_pred_home_xgb = home_goals_rf.predict(X_test_home)
   y_pred_away_xgb = away_goals_rf.predict(X_test_away)

   print("XGBoost Regressor - Home Goals Prediction:")
   print("MAE:", mean_absolute_error(y_test_home, y_pred_home_xgb))
   print("MSE:", mean_squared_error(y_test_home, y_pred_home_xgb))

   print("\nXGBoost Regressor - Away Goals Prediction:")
   print("MAE:", mean_absolute_error(y_test_away, y_pred_away_xgb))
   print("MSE:", mean_squared_error(y_test_away, y_pred_away_xgb))

   trained_ml_models['xgboost_regressor_home'] = home_goals_xgb
   trained_ml_models['xgboost_regressor_away'] = away_goals_xgb

   #Support Vector Regressor (SVR)

   def svr_regressor_tuning():

      pipeline_svr_regressor = make_pipeline(
         StandardScaler(), 
         SVR() 
      )
      params_svr_regressor = {
         'svr__C': stats.uniform(0.1, 10),
         'svr__epsilon' : stats.uniform(0.01, 0.5),
         'svr__kernel' : ['linear', 'rbf'],
         'svr__gamma' : ['scale', 'auto']
      }

      mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

      svr_search_home = RandomizedSearchCV(
         pipeline_svr_regressor,
         params_svr_regressor,
         n_iter=20, 
         cv=tscv_tuning,
         scoring=mae_scorer, 
         random_state=42,
         n_jobs=-1,
         verbose=1
      )
      svr_search_home.fit(X_train_home, y_train_home)
      print("\nSVR Regressor (Home Goals) - Best Parameters: ", svr_search_home.best_params_)
      print("SVR Regressor (Home Goals) - Best average MAE CV: ", -svr_search_home.best_score_)

      svr_search_away = RandomizedSearchCV(
         pipeline_svr_regressor,
         params_svr_regressor,
         n_iter=20, 
         cv=tscv_tuning,
         scoring=mae_scorer, 
         random_state=42,
         n_jobs=-1,
         verbose=1
      )
      svr_search_away.fit(X_train_away, y_train_away)
      print("\nSVR Regressor (Away Goals) - Best Parameters: ", svr_search_away.best_params_)
      print("SVR Regressor (Away Goals) - Best average MAE CV: ", -svr_search_away.best_score_)

   home_goals_svr = Pipeline([
      ('scaler', StandardScaler()),
      ('model', SVR(
         C=1.495,
         epsilon=0.156,
         gamma='auto',
         kernel='linear'
      ))
   ])
   
   away_goals_svr = Pipeline([
      ('scaler',StandardScaler()),
      ('model', SVR(
         C=5.243,
         epsilon=0.306,
         gamma='scale',
         kernel='linear'
      ))
   ])
   
   home_goals_svr.fit(X_train_home, y_train_home)
   away_goals_svr.fit(X_train_away, y_train_away)

   y_pred_home_svr = home_goals_svr.predict(X_test_home)
   y_pred_away_svr = away_goals_svr.predict(X_test_away)

   print("Support Vector Regressor - Home Goals Prediction:")
   print("MAE:", mean_absolute_error(y_test_home, y_pred_home_svr))
   print("MSE:", mean_squared_error(y_test_home, y_pred_home_svr))

   print("\nSupport Vector Regressor - Away Goals Prediction:")
   print("MAE:", mean_absolute_error(y_test_away, y_pred_away_svr))
   print("MSE:", mean_squared_error(y_test_away, y_pred_away_svr))

   trained_ml_models['support_vector_regressor_home'] = home_goals_svr
   trained_ml_models['support_vector_regressor_away'] = away_goals_svr


   #logisticRegressionTuning()
   #randomForestRegressorTuning()
   #randomForestClassifierTuning()
   #xgboost_classifier_tuning()
   #xgboost_regressor_tuning()
   #svc_classifier_tuning()
   #svr_regressor_tuning()

   #Upcoming matches prediction

   probs_lr = model.predict_proba(XUP)
   probs_rfc = rf_model.predict_proba(XUP)
   probs_xgb = pipeline_xgb_classifier.predict_proba(XUP)
   probs_svc = pipeline_svc_classifier.predict_proba(XUP)

   home_xg = home_goals_model.predict(XUP)
   away_xg = away_goals_model.predict(XUP)

   home_xg_rf = home_goals_rf.predict(XUP)
   away_xg_rf = away_goals_rf.predict(XUP)

   home_xg_xgb = home_goals_xgb.predict(XUP)
   away_xg_xgb = away_goals_xgb.predict(XUP)

   home_xg_svr = home_goals_svr.predict(XUP)
   away_xg_svr = away_goals_svr.predict(XUP)

   predicted_matches_df = pd.DataFrame({
      'Home Id' : upcoming_matches['Home Id'],
      "Home": upcoming_matches['Home'],
      "Away Id": upcoming_matches['Away Id'],
      "Away": upcoming_matches['Away'],
      "home_win_lr" : (probs_lr[:, 2] * 100).round(2),
      "draw_lr" : (probs_lr[:, 1] * 100).round(2),
      "away_win_lr" : (probs_lr[:, 0] * 100).round(2),
      "home_win_rfc" : (probs_rfc[:, 2] * 100).round(2),
      "draw_rfc" : (probs_rfc[:, 1] * 100).round(2),
      "away_win_rfc" : (probs_rfc[:, 0] * 100).round(2),
      "home_win_xgb" : (probs_xgb[:, 2] * 100).round(2),
      "draw_xgb" : (probs_xgb[:, 1] * 100).round(2),
      "away_win_xgb" : (probs_xgb[:, 0] * 100).round(2),
      "home_win_svc" : (probs_svc[:, 2] * 100).round(2),
      "draw_svc" : (probs_svc[:, 1] * 100).round(2),
      "away_win_svc" : (probs_svc[:, 0] * 100).round(2),
      "home_xG_lr" : home_xg.round(2),
      "away_xG_lr" : away_xg.round(2),
      "home_xG_rfr": home_xg_rf.round(2),
      "away_xG_rfr" : away_xg_rf.round(2),
      "home_xG_xgb" : home_xg_xgb.round(2),
      "away_xG_xgb" : away_xg_xgb.round(2),
      "home_xG_svr" : home_xg_svr.round(2),
      "away_xG_svr" : away_xg_svr.round(2)
   })

   predicted_matches_df.to_csv(cleaned_data_folder + "PredictedUpcomingMatches.csv", index=False)


   #Visualization

   h2hHomeWins = merged_matches['h2h_home_wins'].sum()
   h2hAwayWins = merged_matches['h2h_away_wins'].sum()
   h2hDraws = merged_matches['h2h_draws'].sum()
   h2hValues = [h2hHomeWins, h2hAwayWins, h2hDraws]
   labels = ['Home Wins', 'Away Wins', 'Draws']
   resultCounts = merged_matches['result_numeric'].value_counts()

   values = [resultCounts[1], resultCounts[0], resultCounts[-1]]

   # Bar Chart
   plt.figure(figsize=(8,5))
   plt.bar(labels, values, color=['blue', 'red', 'gray'])
   plt.xlabel("Match Result")
   plt.ylabel("Count")
   plt.title("Distribution of Match Results")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Bar chart of Match Result Distribution")


   plt.figure(figsize=(12, 6))
   plt.hist(merged_matches["goal_diff_delta"], bins=20, color='purple', alpha=0.7, edgecolor='black')
   plt.xlabel("Goal Difference (Home - Away)")
   plt.ylabel("Frequency")
   plt.title("Distribution of Goal Difference Delta")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Histogram of Goal Difference")

   plt.figure(figsize=(12, 6))
   sns.boxplot(x=merged_matches["h2h_home_wins"], y=merged_matches['result_numeric'])
   plt.title("H2H home wins vs. match result")
   plt.xlabel("H2H home win matches")
   plt.ylabel("Match Result")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Boxplot of H2H Home Wins and Match Result")

   plt.figure(figsize=(12, 6))
   sns.boxplot(x=merged_matches["h2h_away_wins"], y=merged_matches['result_numeric'])
   plt.title("H2H aways wins vs. match result")
   plt.xlabel("H2H away win matches")
   plt.ylabel("Match Result")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Boxplot of H2H Away Wins and Match Result")

   plt.figure(figsize=(12, 6))
   sns.boxplot(x=merged_matches["h2h_draws"], y=merged_matches['result_numeric'])
   plt.title("H2H draws win vs. match result")
   plt.xlabel("H2H draw matches")
   plt.ylabel("Match Result")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Boxplot of H2H draws and Match Result")

   plt.figure(figsize=(12, 6))
   sns.boxplot(x=merged_matches["h2h_home_goals"], y=merged_matches['result_numeric'])
   plt.title("H2H home goals  vs. match result")
   plt.xlabel("H2H home goals")
   plt.ylabel("Match Result")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Boxplot of H2H Home Goals and Match Result")

   plt.figure(figsize=(12, 6))
   sns.boxplot(x=merged_matches["h2h_away_goals"], y=merged_matches['result_numeric'])
   plt.title("H2H away goals vs. match result")
   plt.xlabel("H2H away goals")
   plt.ylabel("Match Result")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Boxplot of H2H Away Goals and Match Result")


   plt.figure(figsize=(6,6))
   plt.pie(h2hValues, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red', 'gray'])
   plt.title('Head-to-Head Results')
   plt.tight_layout()
   plt.savefig(visualization_folder + "Pie chart of H2H Results")


   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="goals_per_match", hue="Season")
   plt.title("Goals Per Match per Team by Season")
   plt.ylabel("Goals Per Match")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Bar plot of Goals Scored per Match by Season")

   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="shots_on_target_per_goal", hue="Season")
   plt.title("Shots On Target Per Goal Across Seasons")
   plt.ylabel("Shots On Target Per goal")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Bar plot of Shots on Target per Goal Across Seasons")


   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="goals_conceded_per_match", hue="Season")
   plt.title("Goals Conceded Per Match by Team")
   plt.ylabel("Goals Conceded")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Bar plot of Goals Conceded per Match by Season")


   plt.figure(figsize=(10, 8))
   corr_features = season_stats[[
      "shots_on_target_per_goal", "shots_per_goal", "sot_ratio", "goals_per_match",
      "goals_conceded_per_match", "fouls_per_match", "cards_ratio",
      "own_goals_ratio", 'offensive_strength', 'defense_strength', 'corners_per_match', 'free_kicks_per_match',
      'penalties_converted_rate','set_piece_strength', 'passing_and_control_strength', 'avg_position_ovr', 'avg_points_per_game', 'weight_points', 'team_power_index'
   ]]
   sns.heatmap(corr_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Between Season stats")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Heatmap of correlation between Season stats")

   plt.figure(figsize=(10, 8))
   offensive_features = season_stats[["shots_on_target_per_goal", "shots_per_goal", "sot_ratio", "goals_per_match", 'offensive_strength', 'team_power_index']]
   sns.heatmap(offensive_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Between Offensive Metrics")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Heatmap of correlation between offensive metrics And Team Power Index")

   plt.figure(figsize=(10, 8))
   defensive_feature = season_stats[["goals_conceded_per_match", "fouls_per_match", "cards_ratio", "own_goals_ratio", "defense_strength", 'team_power_index']]
   sns.heatmap(defensive_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Between Defensive Metrics")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Heatmap of correlation between Defensive metrics And Team Power Index")

   plt.figure(figsize=(10, 8))
   passing_control_feature = season_stats[['pass_accuracy', 'passes_per_game', 'effective_possesion', 'passing_and_control_strength', 'team_power_index']]
   sns.heatmap(passing_control_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Between Passing & Control Metrics")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Heatmap of correlation between Passing & Control metrics And Team Power Index")

   plt.figure(figsize=(10, 8))
   set_piece_feature = season_stats[['corners_per_match', 'free_kicks_per_match', 'penalties_converted_rate', 'set_piece_strength', 'team_power_index']]
   sns.heatmap(set_piece_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("Correlation Between Set Piece Metrics")
   plt.tight_layout()
   plt.savefig(visualization_folder + "Heatmap of correlation between Set Piece metrics And Team Power Index")

   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="weight_points")
   plt.title("Weight points By Team")
   plt.ylabel("Weight Points")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Barplot of Weight Points by Team")


   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="team_power_index")
   plt.title("Team Power Index Barplot")
   plt.ylabel("Team Power Index (TPI)")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Barplot of Team Power Index")

   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="avg_points_per_game")
   plt.title("Average Points Per Game By Team")
   plt.ylabel("Average Points Per Game")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Barplot of Average Points Per Game by Team")

   plt.figure(figsize=(12, 6))
   sns.barplot(data=season_stats, x="Team", y="avg_position_ovr")
   plt.title("Average Position Overall By Team")
   plt.ylabel("Average Position Overall")
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.savefig(visualization_folder + "Barplot of Average Position Overall by Team")

train_models()