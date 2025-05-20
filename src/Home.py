import streamlit as st
import os
from dotenv import load_dotenv

def main():
    load_dotenv("/datadrive/CuongHV/project/ownllm/.env")
    st.markdown(
        """
        # Welcome to the Poseidon Platform! 🌊👋
        Step into a world of cutting-edge LLM-powered applications designed to ignite your creativity, 
        elevate your projects, and seamlessly integrate into your products. 
        Whether you're looking for inspiration or practical tools, the Poseidon Team has you covered. Ready to dive in? 🚀

        ### *Tailored Tools for Archers 🎯* **(Highly Recommended)**:
        - [x] 💬 Chat with LLM: Dive into dynamic conversations with your choice of LLM models 🤖. 
        From brainstorming ideas to exploring possibilities, the sky’s the limit 🌟.
        - [x] 💬 Chat with Images: Unleash your creativity by combining text and visuals. 
        Upload images to spark dynamic conversations with LLM models, enabling you to analyze, describe, or generate ideas inspired by your visuals. 
        Perfect for brainstorming, design feedback, or exploring visual concepts in real-time. 🎨✨
        
        - [x] 📕 Chat with PDF: Take PDF interaction to the next level. Engage in detailed conversations backed by 
        document IDs for persistent sessions, and leverage powerful tools for deeper insights.
        Plus, break language barriers effortlessly—chat in your language, regardless of the language inside the PDF. 🌍💬

        - [x] 📚 Chat with PDFs: Seamlessly interact with multiple PDFs, allowing for cross-document insights and enhanced data retrieval.
        Perfect for comprehensive research and analysis.

        ### *Future Works*:
        * [ ] Push the performance for the summarise questions.
        * [ ] You can config the number of relevance documents, and track them.
        
        
        ##### What started as experimental solutions to real-world challenges has evolved into a growing arsenal of \
        game-changing LLM tools 🛠️. And we’re just getting started—keep your eyes peeled for what’s next! 🔥

        ##### Ready to make waves? 🌊
        """
        )
    
if __name__ == "__main__":
    # Setup environment
    main()