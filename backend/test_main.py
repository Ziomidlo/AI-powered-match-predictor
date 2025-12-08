import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importujemy naszą aplikację i modele
from .main import app, get_db
from .models import Base
from . import schemas

# 1. Konfiguracja testowej bazy danych (w pamięci)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tworzymy tabele w testowej bazie danych
Base.metadata.create_all(bind=engine)

# 2. Nadpisanie zależności `get_db`
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Mówimy naszej aplikacji, aby podczas testów używała testowej bazy danych
app.dependency_overrides[get_db] = override_get_db

# 3. Tworzymy klienta testowego
client = TestClient(app)

# 4. Piszemy testy!

def test_read_seasons():
    # Arrange (przygotowanie) - możemy dodać przykładowy sezon do testowej bazy
    db = next(override_get_db())
    db.add(schemas.Season(id=1, season="2023/2024"))
    db.commit()
    
    # Act (działanie) - wysyłamy żądanie do API
    response = client.get("/seasons/")
    
    # Assert (sprawdzenie) - sprawdzamy, czy wszystko się zgadza
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["season"] == "2023/2024"

def test_read_teams_empty():
    response = client.get("/teams/")
    assert response.status_code == 200
    assert response.json() == []

def test_predict_match_not_found():
    response = client.post("/predicted_matches/predict/999") # ID, które nie istnieje
    assert response.status_code == 404
    assert response.json()["detail"] == "Predicted match not found"