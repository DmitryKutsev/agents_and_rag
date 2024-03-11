import streamlit as st
from dotenv import load_dotenv
from react.react_agent import get_react_agent

def extract_tool_info(process_data):
    # Initialize the result dictionary
    result = {
        'steps': []
    }

    # Extract intermediate steps
    intermediate_steps = process_data.get('intermediate_steps', [])

    # Iterate through each step
    for step_number, (action, _) in enumerate(intermediate_steps, start=1):
        step_info = {
            'step': step_number,
            'tool': action.tool,
            'tool_input': action.tool_input,
            'log': action.log.split('\n')[0]
        }
        result['steps'].append(step_info)

        print(action)
        print()

    return result

load_dotenv(override=True)
agent = get_react_agent()

# Page title
title = "Compare agents"
st.set_page_config(page_title=title)
st.title(title)

with st.form('query_form', clear_on_submit=False):
    query_text = st.text_input('Enter your question:', placeholder='Type your query here...', key='query_text')
    submitted = st.form_submit_button('Submit')

if submitted and query_text:
    with st.spinner('Processing your query...'):
        response = agent.run_agent(query_text)
        
        # Display the response output
        st.write(response['output'])
        
        # Extract and display intermediate steps
        steps = extract_tool_info(response)
        st.subheader("Process Steps:")
        for step in steps['steps']:
            with st.expander(f"Step {step['step']}: Tool - {step['tool']}, Input - {step['tool_input']}", expanded=False):
                st.text("Log:")
                st.text_area("", step['log'], height=150, key=f"log_{step['step']}")