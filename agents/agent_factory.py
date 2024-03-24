from agents.multi_agent_systems import get_mas
from agents.single_agent_systems import get_agent

def agent_system_factory(agent_type: str = "react"):
    if agent_type == "react":
        agent_system = get_agent(agent_type="react")
    elif agent_type == "openai":
        agent_system = get_agent(agent_type="openai")
    elif agent_type == "multi":
        agent_system = get_mas(mas_type="default")
    else:
        raise ValueError(f"Invalid agent type: {agent_type}. Expected 'react' or 'openai' or 'multi'.")
    
    return agent_system