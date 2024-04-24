import customtkinter as ctk
import threading
from pathlib import Path
from modules import textGen as tg

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("green") # Themes: blue (default), dark-blue, green

class ResponseWorker(threading.Thread):
    def __init__(self, user_input, context, update_callback, finish_callback):
        super().__init__()
        self.user_input = user_input
        self.context = context
        self.update_callback = update_callback
        self.finish_callback = finish_callback

    def run(self):
        print(">>> [Worker] Generating Response...")
        self.update_callback("Bukowski:\n")
        for token in tg.reader(self.user_input, self.context):
            print("Token", token)
            self.update_callback(token)
        print(">>> [Worker] Finished generating response.")
        self.finish_callback()

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Bukowski GPT')
        self.geometry('800x600')
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close

        self.context = self.load_context()

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill='both', expand=True)

        self.conversation_display = ctk.CTkTextbox(self.main_frame, state='disabled', width=780, height=540)
        self.conversation_display.pack(fill='both', expand=True)

        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill='x', pady=10)

        self.input_field = ctk.CTkEntry(self.bottom_frame, placeholder_text="Type here...", height=40)
        self.input_field.pack(side='left', fill='x', expand=True, padx=5)
        self.input_field.bind('<Return>', self.enter_press) 
        self.input_field.bind('<Shift-Return>', self.insert_newline) 

        self.send_button = ctk.CTkButton(self.bottom_frame, text='Send', command=self.send_query, height=40)
        self.send_button.pack(side='right', padx=5)

    def load_context(self):
        file_path = Path(__file__).resolve().parent.parent / 'data' / 'CharlesB2.pdf'
        print("PDF path:", file_path)
        return tg.load_pdf(str(file_path))

    def enter_press(self, event=None):
        if event.state & 0x0001:  # Shift key is pressed
            self.insert_newline(event)
        else:
            self.send_query()

    def insert_newline(self, event=None):
        self.input_field.insert(ctk.END, "\n")
        
    def send_query(self):
        user_input = self.input_field.get().strip()
        if user_input:
            self.append_conversation(f"You: {user_input}\n\n")
            self.input_field.delete(0, ctk.END)
            threading.Thread(target=self.start_response_thread, args=(user_input,), daemon=True).start()

    def start_response_thread(self, user_input):
        worker = ResponseWorker(user_input, self.context, self.append_conversation, self.finish_response)
        worker.start()

    def append_conversation(self, text):
        self.conversation_display.configure(state='normal')
        self.conversation_display.insert(ctk.END, text)
        self.conversation_display.configure(state='disabled')
        self.conversation_display.yview_moveto(1)  # Scroll to the bottom

    def finish_response(self):
        self.append_conversation("\n---\n\n")
    
    def on_close(self):
        # Ensure all threads are cleaned up before closing the application
        # Implement any necessary thread cleanup here
        self.destroy()

def run_app():
    app = ChatGPTUI()
    app.mainloop()
    
if __name__ == "__main__":
    run_app()
