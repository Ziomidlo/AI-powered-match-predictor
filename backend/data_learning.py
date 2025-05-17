from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy.stats as stats


def train_models():
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

   probs_lr = model.predict_proba(XUP)
   probs_rfc = rf_model.predict_proba(XUP)

   home_xg = home_goals_model.predict(XUP)
   away_xg = away_goals_model.predict(XUP)

   home_xg_rf = home_goals_rf.predict(XUP)
   away_xg_rf = away_goals_rf.predict(XUP)

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
      "home_xG_lr" : home_xg.round(2),
      "away_xG_lr" : away_xg.round(2),
      "home_xG_rfr": home_xg_rf.round(2),
      "away_xG_rfr" : away_xg_rf.round(2)

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

   plt.figure(figsize=(13, 8))
   sns.barplot(data=feature_importance_df.sort_values(by='Importance', ascending=False).head(20),
               x='Importance', y='Feature', palette='viridis')
   plt.title("Top 20 Feature Importances (Logistic Regression)")
   plt.savefig(visualization_folder + "Barplot of top 20 importance features")

train_models()