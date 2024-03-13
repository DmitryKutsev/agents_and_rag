import streamlit as st
from dotenv import load_dotenv
from agents.agents import get_agent

def extract_tool_info(process_data):
    # Initialize the result dictionary
    result = {
        'steps': []
    }

    # Extract intermediate steps
    intermediate_steps = process_data.get('intermediate_steps', [])

    # Iterate through each step
    for step_number, (action, observation) in enumerate(intermediate_steps, start=1):
        
        # Depending on the agent type, the log may be formatted differently
        try:
            split_log = action.log.split('\n')
            log = split_log[0] if split_log[0] != "" else split_log[1]
        except:
            log = ""

        step_info = {
            'step': step_number,
            'tool': action.tool,
            'tool_input': action.tool_input,
            'log': log,
            'observation': observation
        }
        result['steps'].append(step_info)

    return result

load_dotenv(override=True)

# Define all available agents
agents_mapping = {
    "ReAct": get_agent(agent_type="react"),
    "OpenAITools": get_agent(agent_type="openai"),
    # Add more agents here as needed
}

# Define the tra
trajectory_evaluators_mapping = {
    "Helpfulness": "HelpfulnessEvaluator",
    "Step Necessity": "StepNecessityEvaluator",
    "Tool Selection": "ToolSelectionEvaluator",
    # Add more evaluators here as needed
}

# Page title
title = "Compare agents"
st.set_page_config(page_title=title)
st.title(title)

with st.form('query_form', clear_on_submit=False):
    query_text = st.text_input(label='Enter your question:', placeholder='What is the average authorized capital of the companies in our database?', key='query_text')
    agent_selection = st.multiselect('Select agents to compare:', ['ReAct', 'OpenAITools'])
    
    submitted = st.form_submit_button('Submit')

if submitted and query_text:
    with st.spinner('Processing your query...'):
        # Initialize a dictionary to store the responses from each agent
        responses = {}

        # Run the selected agents
        for agent in agent_selection:
            responses[agent] = agents_mapping[agent].run_agent(query_text)
        
        # Give the user the option to run evaluation metrics
        with st.form("evaluation_form"):
            eval_selection = st.multiselect('Select trajectory evaluations to run:', ['Helpfulness', 'Step Necessity', 'Tool Selection'])
            eval_submitted = st.form_submit_button('Run Evaluations')
        
        # Setup columns for side by side display
        cols = st.columns(len(agent_selection))
        
        # Display the response output for each agent
        for i, agent in enumerate(agent_selection):
            with cols[i]:
                st.subheader(f"{agent} Agent Response:")
                st.write(responses[agent]['output'])
                
                # Extract and display intermediate steps for each agent
                steps = extract_tool_info(responses[agent])
                for step in steps['steps']:
                    with st.expander(f"Step {step['step']}: Tool - {step['tool']}, Input - {step['tool_input']}", expanded=False):
                        st.text_area("Log:", step['log'], key=f"{agent}_log_{step['step']}")
                        if 'observation' in step:
                            st.text_area("Observation:", step['observation'], key=f"{agent}_observation_{step['step']}")
                
    # If the user has submitted the evaluation form, run the selected evaluations
    if eval_submitted:
        raise NotImplementedError("Evaluation metrics are not yet implemented")



        
                