import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import time
# Import the engine from rag_engine
from rag_engine import RagEngine

class ContinuousRagApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Continual RAG - TinyLlama Explorer")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize Engine in a separate thread to avoid UI freezing
        self.engine = None
        self.status_var = tk.StringVar(value="Initializing AI Models... Please wait.")
        
        self.setup_ui()
        
        # Start engine initialization
        threading.Thread(target=self.init_engine, daemon=True).start()

    def init_engine(self):
        try:
            self.engine = RagEngine()
            self.status_var.set("AI Ready.")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to load models: {str(e)}")

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1e1e1e", height=50)
        header.pack(fill=tk.X)
        tk.Label(header, text="Continual RAG: Local Learning Assistant", font=("Arial", 16, "bold"), fg="#ffffff", bg="#1e1e1e").pack(pady=10)

        # Main content area
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Column: Chat
        chat_frame = tk.LabelFrame(main_frame, text="Chat with AI", bg="#2b2b2b", fg="#ffffff", font=("Arial", 10, "bold"))
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#1e1e1e", fg="#e0e0e0", font=("Consolas", 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        input_frame = tk.Frame(chat_frame, bg="#2b2b2b")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.query_entry = tk.Entry(input_frame, bg="#3c3c3c", fg="#ffffff", insertbackground="white", font=("Arial", 11))
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.query_entry.bind("<Return>", lambda e: self.send_query())

        self.send_btn = tk.Button(input_frame, text="Send", command=self.send_query, bg="#007acc", fg="white", relief=tk.FLAT)
        self.send_btn.pack(side=tk.RIGHT)

        # Right Column: Knowledge Base
        kb_frame = tk.LabelFrame(main_frame, text="Continual Learning (Add Knowledge)", bg="#2b2b2b", fg="#ffffff", font=("Arial", 10, "bold"))
        kb_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        tk.Label(kb_frame, text="Paste new text/info below:", bg="#2b2b2b", fg="#cccccc").pack(anchor="w", padx=5)
        self.kb_input = scrolledtext.ScrolledText(kb_frame, wrap=tk.WORD, height=15, bg="#3c3c3c", fg="#ffffff", font=("Arial", 10))
        self.kb_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.add_btn = tk.Button(kb_frame, text="Learn This Information", command=self.add_knowledge, bg="#28a745", fg="white", relief=tk.FLAT)
        self.add_btn.pack(fill=tk.X, padx=5, pady=5)

        # Sources Display
        sources_frame = tk.LabelFrame(main_frame, text="Retrieved Context (Sources)", bg="#2b2b2b", fg="#ffffff", font=("Arial", 10, "bold"))
        sources_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=10)
        
        self.sources_display = scrolledtext.ScrolledText(sources_frame, wrap=tk.WORD, height=6, bg="#1e1e1e", fg="#a0a0a0", font=("Consolas", 9))
        self.sources_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status Bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333333", fg="#ffffff")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

    def log_chat(self, role, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{role}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def send_query(self):
        query = self.query_entry.get().strip()
        if not query: return
        if not self.engine:
            messagebox.showwarning("Wait", "AI Engine is still loading...")
            return

        self.query_entry.delete(0, tk.END)
        self.log_chat("You", query)
        self.status_var.set("AI is thinking...")
        
        threading.Thread(target=self.process_query, args=(query,), daemon=True).start()

    def process_query(self, query):
        try:
            result = self.engine.query(query)
            self.root.after(0, lambda: self.log_chat("AI", result["answer"]))
            
            # Update source display
            self.root.after(0, lambda: self.update_sources(result["sources"]))
            self.status_var.set("AI Ready.")
        except Exception as e:
            self.root.after(0, lambda: self.log_chat("AI", f"Error: {str(e)}"))
            self.status_var.set("Error during query.")

    def update_sources(self, sources):
        self.sources_display.config(state=tk.NORMAL)
        self.sources_display.delete(1.0, tk.END)
        if sources:
            for i, src in enumerate(sources):
                self.sources_display.insert(tk.END, f"Source {i+1}:\n{src}\n{'-'*30}\n")
        else:
            self.sources_display.insert(tk.END, "No specific context found.")
        self.sources_display.config(state=tk.DISABLED)

    def add_knowledge(self):
        text = self.kb_input.get(1.0, tk.END).strip()
        if not text: return
        if not self.engine:
            messagebox.showwarning("Wait", "AI Engine is still loading...")
            return

        self.kb_input.delete(1.0, tk.END)
        self.status_var.set("Learning new information...")
        
        threading.Thread(target=self.process_knowledge, args=(text,), daemon=True).start()

    def process_knowledge(self, text):
        try:
            self.engine.add_knowledge(text)
            self.root.after(0, lambda: messagebox.showinfo("Learned", "Information successfully added to memory!"))
            self.status_var.set("AI Ready.")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Learning Error", f"Failed to learn: {str(e)}"))
            self.status_var.set("Error during learning.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ContinuousRagApp(root)
    root.mainloop()
