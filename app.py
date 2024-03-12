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
react_agent = get_agent(agent_type="react")
openai_agent = get_agent(agent_type="openai")

# Page title
title = "Compare agents"
st.set_page_config(page_title=title)
st.title(title)

with st.form('query_form', clear_on_submit=False):
    query_text = st.text_input(label='Enter your question:', placeholder='What is the average authorized capital of the companies in our database?', key='query_text')
    submitted = st.form_submit_button('Submit')
    
if submitted and query_text:
    with st.spinner('Processing your query...'):
        react_response = react_agent.run_agent(query_text)
        openai_response = openai_agent.run_agent(query_text)
        
        # Setup columns for side by side display
        col1, col2 = st.columns(2)
        
        # Display the response output for react_agent
        with col1:
            st.subheader("React Agent Response:")
            st.write(react_response['output'])
            
            # Extract and display intermediate steps for react_agent
            react_steps = extract_tool_info(react_response)
            for step in react_steps['steps']:
                with st.expander(f"Step {step['step']}: Tool - {step['tool']}, Input - {step['tool_input']}", expanded=False):
                    st.text_area("Log:", step['log'], key=f"react_log_{step['step']}")
                    if 'observation' in step:
                        st.text_area("Observation:", step['observation'], key=f"react_observation_{step['step']}")
        
        # Display the response output for openai_agent
        with col2:
            st.subheader("OpenAI Agent Response:")
            st.write(openai_response['output'])
            
            # Extract and display intermediate steps for openai_agent
            openai_steps = extract_tool_info(openai_response)
            for step in openai_steps['steps']:
                with st.expander(f"Step {step['step']}: Tool - {step['tool']}, Input - {step['tool_input']}", expanded=False):
                    st.text_area("Log:", step['log'], key=f"openai_log_{step['step']}")
                    if 'observation' in step:
                        st.text_area("Observation:", step['observation'], key=f"openai_observation_{step['step']}")