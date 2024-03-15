from abc import abstractmethod

class Agent():
    """Base class for all agents."""
    
    @abstractmethod 
    def run_agent(self, query: str):
        """
        Run the agent.
        """
        pass

    @abstractmethod
    def format_agent_response(self, output):
        """
        Format the agent's output.
        """
        pass