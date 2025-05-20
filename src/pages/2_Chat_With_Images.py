import streamlit as st
import base64
import io
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import tracing_v2_enabled
from utils.sidebar_config import setup_llm
from utils.llm_models import get_llm_model

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌", unsafe_allow_html=True)
    def on_llm_end(self, response=None, **kwargs) -> None:
        self.container.markdown(self.text, unsafe_allow_html=True)
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        st.error(f"LLM Error: {error}")
        self.text = ""
    def on_llm_start(self, *args, **kwargs) -> None: self.text = ""
    def on_chat_model_start(self, *args, **kwargs) -> None: self.text = ""
    # on_text might be redundant if on_llm_new_token handles all streaming
    # def on_text(self, text: str, **kwargs) -> None: self.on_llm_new_token(text)

# --- Initialize Page Function (No changes needed) ---
def initialize_page():
    st.set_page_config(page_title="Chat With Images", page_icon="💬", layout="wide")
    st.title("💬 Chat With Images")
    st.subheader("Pick 1 model and chat")

# --- Persistent File Management Function (Corrected) ---
def manage_persistent_files_state(form_key="image_upload_form", uploader_key="persistent_image_uploader"):
    """Manages adding (via form), displaying, and removing files stored in session state."""
    if "persistent_uploaded_files" not in st.session_state:
        st.session_state.persistent_uploaded_files = []

    st.subheader("Attach Images")

    # --- Form for Uploading New Images ---
    # The 'with' block should ONLY contain the form input elements and the submit button
    with st.form(form_key, clear_on_submit=True):
        uploaded_files_in_form = st.file_uploader( # Use a distinct variable name
            "Select images to attach:",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'webp'],
            label_visibility="collapsed",
            key=uploader_key # Use the provided key
        )
        # The only button inside the form should be the submit button
        submitted = st.form_submit_button("📎 Attach Selected Images") # More descriptive label

    # --- Process Form Submission (AFTER the 'with' block) ---
    # This logic runs *only if* the form was submitted in the previous script run
    if submitted and uploaded_files_in_form: # Check if the list is not empty
        files_were_added = False
        num_added = 0
        # Process the files that were uploaded *within that specific form submission*
        for file in uploaded_files_in_form:
            is_duplicate = any(
                f['name'] == file.name and f['size'] == file.size
                for f in st.session_state.persistent_uploaded_files
            )
            if not is_duplicate:
                try:
                    file_bytes = file.getvalue()
                    st.session_state.persistent_uploaded_files.append({
                        # Use getattr for file_id as it might not always exist
                        "id": f"{file.name}_{file.size}_{getattr(file, 'file_id', 'default_id')}",
                        "name": file.name,
                        "type": file.type,
                        "size": file.size,
                        "data": file_bytes
                    })
                    files_were_added = True
                    num_added += 1
                except Exception as e:
                     st.error(f"Error processing file {file.name}: {e}") # Show error during processing

        if files_were_added:
            st.toast(f"Attached {num_added} new image(s).", icon="✅")
            # Rerun is necessary here to update the "Currently attached" display below
            # immediately after files are added to session state.
            st.rerun()
        elif submitted and uploaded_files_in_form: # Files were submitted, but none were new
             st.toast("Selected image(s) were already attached.", icon="ℹ️")


    # --- Display and Manage Currently Attached Images (Outside and After the Form) ---
    # This section displays the current state of st.session_state.persistent_uploaded_files
    files_to_remove_indices = []
    if not st.session_state.persistent_uploaded_files:
        st.caption("No images currently attached. Use the form above to select and attach.")
    else:
        st.write(f"Currently attached ({len(st.session_state.persistent_uploaded_files)}):")
        num_cols = 3 # Adjust layout as needed
        cols = st.columns(num_cols)
        for i, file_info in enumerate(st.session_state.persistent_uploaded_files):
            col_index = i % num_cols
            with cols[col_index]:
                try:
                    st.image(file_info["data"], caption=f"{file_info['name']} ({file_info['size'] / 1024:.1f} KB)", width=150)
                    # This button is correctly placed OUTSIDE the form.
                    # Its click triggers a standard rerun, handled below.
                    if st.button("❌ Remove", key=f"remove_{file_info['id']}", help="Remove this image"):
                        files_to_remove_indices.append(i)
                except Exception as e:
                     # Handle potential errors if file_info["data"] is corrupted or invalid
                     st.error(f"Error displaying image {file_info.get('name', 'unknown')}: {e}")


    # Handle removal requests (triggered by the "Remove" buttons above)
    if files_to_remove_indices:
        for index in sorted(files_to_remove_indices, reverse=True):
            try:
                del st.session_state.persistent_uploaded_files[index]
            except IndexError:
                 st.warning(f"Could not remove file at index {index}.") # Add robustness
        # Rerun is needed to update the display after removal
        st.rerun()


# --- Main Function (Using the corrected manage_persistent_files_state) ---
# Adapting the chat input logic from the previous good example which handles
# the state updates correctly with reruns.
def main():
    initialize_page()

    config = setup_llm(mode='image')
    llm_choice = config['llm_choice']
    model_choice = config['model_choice']
    chat_permission = config['chat_permission']
    param_config = config['parameters']

    # Initialize session state for messages
    if "messages_ig" not in st.session_state:
        st.session_state["messages_ig"] = [AIMessage(content="I'm a chatbot from Kyanon Digital—ask me anything! You can also attach images.")]
    # persistent_uploaded_files initialized within manage_persistent_files_state if needed

    # New Chat button in sidebar
    if st.sidebar.button("✨ New Chat"):
        st.session_state.messages_ig = [AIMessage(content="New chat started! How can I help?")]
        st.session_state.persistent_uploaded_files = [] # Clear files too
        st.rerun()

    col_chat, col_uploader = st.columns([3, 2]) # Adjust column ratio if desired

    with col_uploader:
        if chat_permission:
            # Call the corrected function
            manage_persistent_files_state(form_key="image_upload_form", uploader_key="persistent_uploader_in_form")
        else:
            st.warning("Configure LLM from the sidebar to enable chat and uploads.")

    with col_chat:
        st.subheader("Chat History")
        chat_container = st.container() # Use container for potential scrolling
        with chat_container:
            # Display existing messages (same logic as before)
            for msg in st.session_state.messages_ig:
                 if isinstance(msg, HumanMessage):
                     with st.chat_message("user"):
                         if isinstance(msg.content, list):
                             text_parts = [item["text"] for item in msg.content if item.get("type") == "text"]
                             image_parts = [item["image_url"]["url"] for item in msg.content if item.get("type") == "image_url"]
                             if text_parts: st.markdown(" ".join(text_parts))
                             if image_parts:
                                 num_img_cols = min(len(image_parts), 3)
                                 img_cols = st.columns(num_img_cols)
                                 for idx, img_data_uri in enumerate(image_parts):
                                     try: img_cols[idx % num_img_cols].image(img_data_uri, width=150)
                                     except Exception as e: img_cols[idx % num_img_cols].warning(f"Couldn't display hist img")
                         elif isinstance(msg.content, str): st.markdown(msg.content)
                 elif isinstance(msg, AIMessage):
                     with st.chat_message("assistant"): st.markdown(msg.content)

        # --- Handle Chat Input ---
        if chat_permission:
            # Use a flag to check if we need to process the last message (avoids re-processing on unrelated reruns)
            needs_processing = ("process_last_message" in st.session_state and st.session_state.process_last_message)

            if prompt := st.chat_input("Enter your message (attached images will be sent)...", key="chat_input_main"):
                # 1. Prepare display message (text + current attachments)
                user_message_display_content = [{"type": "text", "text": prompt}]
                # Make a copy of files AT THIS MOMENT to associate with this message
                st.session_state.files_for_prompt = st.session_state.get("persistent_uploaded_files", [])[:]

                if st.session_state.files_for_prompt:
                    for file_info in st.session_state.files_for_prompt:
                        try:
                            base64_image = base64.b64encode(file_info["data"]).decode('utf-8')
                            mime_type = file_info["type"]
                            user_message_display_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            })
                        except Exception as e:
                            st.warning(f"Could not create preview for {file_info['name']}: {e}")

                # 2. Append display message to history
                st.session_state.messages_ig.append(HumanMessage(content=user_message_display_content))
                # 3. Set flag to process this message on the *next* run and rerun
                st.session_state.process_last_message = True
                st.rerun()

            # --- Processing logic (runs AFTER the rerun triggered by chat_input) ---
            if needs_processing:
                # Reset the flag
                st.session_state.process_last_message = False

                # Retrieve the associated files saved earlier
                attached_files_info = st.session_state.get("files_for_prompt", [])

                # Get the actual text prompt from the last message
                last_user_message_content = st.session_state.messages_ig[-1].content
                prompt_text = ""
                if isinstance(last_user_message_content, list):
                    for item in last_user_message_content:
                         if item['type'] == 'text':
                              prompt_text = item['text']
                              break
                elif isinstance(last_user_message_content, str): # Fallback
                     prompt_text = last_user_message_content

                # Prepare the actual content list for the LLM
                llm_message_content_list = [{"type": "text", "text": prompt_text}]

                if attached_files_info:
                    st.toast(f"Sending message with {len(attached_files_info)} image(s)...", icon="🖼️")
                    for file_info in attached_files_info:
                        try:
                            base64_image = base64.b64encode(file_info["data"]).decode('utf-8')
                            mime_type = file_info["type"]
                            llm_message_content_list.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            })
                        except Exception as e:
                            st.error(f"Error encoding image {file_info['name']} for LLM: {e}")

                # Update the last message in history to reflect what was *actually* sent to LLM
                # This is useful if encoding failed for some images, etc.
                st.session_state.messages_ig[-1] = HumanMessage(content=llm_message_content_list)

                # Invoke LLM
                with st.chat_message('assistant'):
                    stream_container = st.empty()
                    stream_handler = StreamHandler(stream_container)
                    try:
                        llm = get_llm_model(
                            chatmodel=llm_choice, model_name=model_choice,
                            param=param_config, stream_handler=stream_handler
                        )
                        # Send history up to and including the corrected user message
                        with tracing_v2_enabled(project_name="DEPLOY_MIND_Chat_With_LLM"):
                            if llm_choice == "Gemini":
                                for chunk in llm.stream(st.session_state.messages_ig):
                                    stream_handler.on_llm_new_token(chunk.content)
                                st.session_state.messages_ig.append(AIMessage(content=stream_handler.text))
                            else:
                                response = llm.invoke(st.session_state.messages_ig)
                                st.session_state.messages_ig.append(AIMessage(content=response.content))

                    except Exception as e:
                        stream_handler.on_llm_error(e) # Display error in stream container
                        st.session_state.messages_ig.append(AIMessage(content=f"Sorry, error processing request: {e}"))

                # Clear the main persistent file list AFTER sending
                if attached_files_info: # Only clear if files were sent with this message
                     st.session_state.persistent_uploaded_files = []
                     st.toast("Attached images cleared after sending.", icon="✅")
                     # Clear the temporary storage as well
                     if "files_for_prompt" in st.session_state:
                          del st.session_state.files_for_prompt

                # Final rerun to update the uploader display (now empty) and show AI response
                st.rerun()

            # --- End of chat input/processing handling ---

        elif not chat_permission:
             st.warning("Please configure the LLM in the sidebar to start chatting.")

# --- Entry point ---
if __name__ == "__main__":
    main()