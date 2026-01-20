from pydantic import BaseModel, Field
from typing import Literal, Optional
import datetime

# Model to route user queries to the appropriate datasource
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource.
    - : are used to define type annotations for the model fields (datasource is an object variable, __init__ is hiddenly called).
    - Literal allows to restrict the possible values for 'datasource' to a predefined set.
    - Field is used to add metadata (like description) to the model fields."""
    
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question, choose the most relevant datasource to answer it."
    )
    
    

# Model to search over tutorial videos about a software framework  
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software framework."""
    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts."
    )
    title_search: str = Field(
        ...,
        description="Alternate search query focused on video titles. Should be succinct and only contain keywords."
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter for videos, inclusive. Only used if specified."
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter for videos, inclusive. Only used if specified."
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter for videos, inclusive. Only used if specified."
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter for videos, inclusive. Only used if specified."
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length filter in seconds, inclusive. Only used if specified."
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length filter in seconds, inclusive. Only used if specified."
    )
    
    def pretty_print(self) -> None:
        """Print non-null fields in a readable format."""
        for field, value in self.model_dump().items():
            if value is not None:
                print(f"{field}: {value}")