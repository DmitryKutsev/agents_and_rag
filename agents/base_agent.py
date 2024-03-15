from abc import abstractmethod

class Agent():
    """Base class for all agents."""
    
    @abstractmethod 
    def run_agent(self, query: str): 
        """
        Run the agent and return the result and the time it took to run the agent.
        """
        pass

    @abstractmethod
    def format_agent_response(self, output):
        """
        Format the agent's output.
        """
        pass