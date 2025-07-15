# LegalBot

An experimental legal question answering chatbot, for the Ethiopia jurisdiction (Disclaimer: This bot aint legal, at least for now):)

## Components ---> Chat App + RAG + VDB + LLM

### Chat app

This is the frontend of our chatbot. The user uses it to ask questions and to getback response/answer

### RAG (Retrieval augmented generation)

is the core of our chatbot. RAG fetches (retrieves) context from VDB based on the user question. Then transfers this the LLM to getback answer to the user

### VDB (Vector database)

This is where our data is located. For our case it is the ETHIOPIAN CONSTITUTION. For now we use a VDB called Pinecone

### LLM (Large Language Model)

This is what gives the final answer to the user. For that we use an AI model called Gemini (this is from google)

![components and operation diagram](operation_diagram.png)

1. **Ask question**: get question from the user 

2. **Get context** from the VDB

3. **Combine context with question** and give to the LLM

4. **Get response** and give it back to the chat app (to the user)


## Next Steps
1. Test with some questions (for the sake of correcting issues related with article and chapter number questions)

2. Adding evaluation pipeline

3. AMHARIC feature (Asking questions in amharic + translator component)

4. Adding other law books and legal documents (civil, criminal, family, ... and other Ethiopia Legislative Codes and proclamations)