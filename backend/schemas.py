from pydantic import BaseModel, Field


class TeamBase(BaseModel):
    team_name: str = Field(examples=["Manchester United FC", "Liverpool FC"], description="Name of the Team")
    venue: str = Field(examples=["Old Trafford", "Anfield"], description="Name of the stadium")

class Team(TeamBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the team")

    model_config = {
        "from_attributes" : True
    }