import streamlit as st
from rag_pipeline import init_rag_pipeline, get_rag_response

def main():
    #st.title("RAG Chat App with Google AI and Pinecone")

    st.image("legalbanner.svg", use_container_width=True)

    vector_store, llm = init_rag_pipeline()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            response = get_rag_response(vector_store, llm, prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()