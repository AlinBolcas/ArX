import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from modules.textGen import writer
import threading

class ChatGPTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatGPT Clone")
        
        self.create_widgets()

    def create_widgets(self):
        # Create the conversation display area
        self.conversation_display = ScrolledText(self.root, state='disabled', width=80, height=20)
        self.conversation_display.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Create the text entry field
        self.text_entry = tk.Text(self.root, height=4)
        self.text_entry.grid(row=1, column=0, sticky="ew")

        # Bind the enter key to the send_message function (Shift+Enter for newline)
        self.text_entry.bind("<Return>", self.send_message)
        self.text_entry.bind("<Shift-Return>", self.insert_newline)

        # Create the send button
        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, sticky="ew")

        # Make the grid expandable
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def send_message(self, event=None):
        # Prevent the function from running if the event is Shift+Return
        if event and event.state & 0x0001:
            return

        msg = self.text_entry.get("1.0", tk.END).strip()
        if msg:
            # Simulate displaying the user's message
            self.update_conversation("You", msg)
            # Clear the entry field
            self.text_entry.delete("1.0", tk.END)
            # Start a new thread for generating the response
            threading.Thread(target=self.generate_response, args=(msg,)).start()

        # Prevent the default behavior of the Return key
        return "break"

    def generate_response(self, user_input):
        # Here, call your function to generate the response. For example:
        gpt_response = writer(user_input, "", "", "")
        # Update the conversation in the main thread
        self.root.after(0, self.update_conversation, "GPT", gpt_response)

    def insert_newline(self, event=None):
        self.text_entry.insert(tk.END, "\n")
        # Move the cursor to the end
        self.text_entry.mark_set(tk.INSERT, tk.END)
        return "break"

    def update_conversation(self, sender, message):
        self.conversation_display.config(state='normal')
        self.conversation_display.insert(tk.END, f"{sender}: {message}\n")
        self.conversation_display.config(state='disabled')
        # Scroll to the bottom
        self.conversation_display.yview(tk.END)

def run_app():
    root = tk.Tk()
    app = ChatGPTApp(root)
    root.mainloop()
    
    
if __name__ == "__main__":
    run_app()
