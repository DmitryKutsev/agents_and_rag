import streamlit as st
from dotenv import load_dotenv
from agents.agent_factory import agent_system_factory
from evaluation.trajectory_evaluation.trajectory_evaluators import get_trajectory_evaluator

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
    "ReAct": agent_system_factory(agent_type="react"),
    "OpenAITools": agent_system_factory(agent_type="openai"),
    #"MultiAgentSystem": agent_system_factory(agent_type="multi"),
    # Add more agents here as needed
}

# Define the tra
trajectory_evaluators_mapping = {
    "Helpfulness": get_trajectory_evaluator("helpfulness"),
    "Step necessity": get_trajectory_evaluator("step_necessity"),
    "Tool selection": get_trajectory_evaluator("tool_selection"),
    # Add more evaluators here as needed
}

# Page title
title = "Compare agents"
st.set_page_config(page_title=title)
st.title(title)

with st.form('query_form', clear_on_submit=False):
    query_text = st.text_input(label='Enter your question:', placeholder='What is the average authorized capital of the companies in our database?', key='query_text')
    agent_selection = st.multiselect('Select agents to compare:', ['ReAct', 'OpenAITools'])
    eval_selection = st.multiselect('Select trajectory evaluations to run:', ['Helpfulness', 'Step necessity', 'Tool selection'])
    query_submitted = st.form_submit_button('Submit')

if query_submitted and query_text and agent_selection:
    with st.spinner('Processing your query...'):
        # Initialize a dictionary to store the responses from each agent
        responses = {}

        # Run the selected agents
        for agent in agent_selection:
            responses[agent] = agents_mapping[agent].run_agent(query_text)
        
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
        if eval_selection:
            evaluations = {}
            for j, agent in enumerate(agent_selection):  
                evaluations[agent] = {}
                for evaluation in eval_selection:
                    evaluations[agent][evaluation] = trajectory_evaluators_mapping[evaluation].evaluate_agent_trajectory(
                        prediction=responses[agent]["output"],
                        input=responses[agent]["input"],
                        agent_trajectory=responses[agent]["intermediate_steps"]
                    )
                with cols[j]:  
                    st.subheader(f"{agent} Agent Trajectory Evaluation:")
                    for evaluation in evaluations[agent]:
                        st.markdown(f"#### {evaluation}")
                        score = evaluations[agent][evaluation]["score"]
                        if score == 1:
                            st.write("Test passed ✅")
                        else:
                            st.write("Test failed ❌")
                        with st.expander("Reasoning", expanded=False):
                            st.write(evaluations[agent][evaluation]["reasoning"])
                            
  




        
                