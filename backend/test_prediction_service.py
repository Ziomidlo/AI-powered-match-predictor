import pytest
from unittest.mock import patch, MagicMock

# Importujemy funkcję, którą chcemy testować
from ..prediction_service import generate_and_store_match_prediction
from .. import schemas

def test_generate_and_store_prediction_with_mocked_models():
    """
    Testuje logikę serwisu predykcyjnego, mockując (udając) modele ML.
    """
    # Arrange (przygotowanie)
    
    # 1. Stwórz fałszywy obiekt sesji bazy danych
    mock_db_session = MagicMock()
    
    # 2. Stwórz fałszywy obiekt predykcji, który ma być zaktualizowany
    mock_predicted_match = schemas.PredictedMatch(
        id=1, 
        home_team_id=10, 
        away_team_id=20, 
        is_predicted=False
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_predicted_match
    
    # 3. Stwórz fałszywe modele, które będą "zwracane" przez joblib.load
    # Klasyfikator, który zawsze przewiduje remis (prawdopodobieństwa)
    mock_classifier = MagicMock()
    mock_classifier.predict_proba.return_value = [[0.2, 0.6, 0.2]] # [P(A), P(D), P(H)]
    
    # Regresor, który zawsze przewiduje 1.5 gola dla gospodarzy i 0.5 dla gości
    mock_regressor = MagicMock()
    mock_regressor.predict.return_value = [[1.5, 0.5]]
    
    # Używamy patch, aby 'joblib.load' zwracał nasze fałszywe modele
    # 'side_effect' pozwala zwracać różne wartości przy kolejnych wywołaniach
    with patch('joblib.load', side_effect=[mock_classifier] * 4 + [mock_regressor] * 4) as mock_joblib_load:
        
        # Act (działanie)
        result = generate_and_store_match_prediction(match_id=1, db=mock_db_session)

        # Assert (sprawdzenie)
        # 1. Sprawdź, czy funkcja zwróciła zaktualizowany obiekt
        assert result is not None
        assert result.is_predicted == True
        
        # 2. Sprawdź, czy wyniki w obiekcie zgadzają się z tym, co "przewidział" fałszywy model
        assert result.home_win_probability_lr == 0.2
        assert result.draw_probability_lr == 0.6
        assert result.away_win_probability_lr == 0.2
        assert result.home_expected_goals_lr == 1.5
        assert result.away_expected_goals_lr == 0.5
        
        # 3. Sprawdź, czy sesja została zatwierdzona (commit)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(mock_predicted_match)