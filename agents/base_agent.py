from abc import abstractmethod

class Agent():
    """Base class for all agents."""
    
    @abstractmethod 
    def run_agent(self, query: str) -> str:
        """
        Run the agent.
        """
        pass