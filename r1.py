import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import subprocess
import sys
import importlib.util
import queue
import time

# ==================== XP‑STYLE DEPENDENCY INSTALLER ====================
class XPSetupWizard(tk.Tk):
    """600x400 Windows XP‑style wizard that installs Python dependencies."""
    def __init__(self):
        super().__init__()
        self.title("DeepSeek‑V4 Setup Wizard")
        self.geometry("600x400")
        self.resizable(False, False)
        self.configure(bg="#ece9d8")  # classic XP beige

        # Variables
        self.installation_successful = False
        self.log_queue = queue.Queue()
        self.packages = ['torch', 'transformers', 'accelerate']
        self.install_needed = False

        # Style
        style = ttk.Style(self)
        style.theme_use('vista' if 'vista' in style.theme_names() else 'clam')
        style.configure("blue.Horizontal.TProgressbar", troughcolor='white', background='#3c7fb1')

        # ===== XP‑style title bar imitation =====
        title_frame = tk.Frame(self, bg="#0058e7", height=30)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(title_frame, text="DeepSeek‑V4 Setup Wizard", fg="white",
                                bg="#0058e7", font=("Segoe UI", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)

        # ===== Main content =====
        main = tk.Frame(self, bg="#ece9d8", padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instr = tk.Label(main, text="This wizard will install required Python packages for DeepSeek‑V4.",
                         bg="#ece9d8", font=("Segoe UI", 9), anchor="w", justify=tk.LEFT)
        instr.pack(fill=tk.X, pady=(0, 10))

        # Log area
        log_frame = tk.LabelFrame(main, text="Installation log", bg="#ece9d8", font=("Segoe UI", 8))
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg="white", fg="black",
                                                   font=("Consolas", 8), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(main, mode='indeterminate', style="blue.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Button frame
        btn_frame = tk.Frame(main, bg="#ece9d8")
        btn_frame.pack(fill=tk.X)

        self.install_btn = tk.Button(btn_frame, text="Install Dependencies", command=self.start_installation,
                                      bg="#d6d2c2", activebackground="#c0baa8", relief=tk.RAISED,
                                      font=("Segoe UI", 8), padx=10)
        self.install_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.destroy,
                                     bg="#d6d2c2", activebackground="#c0baa8", relief=tk.RAISED,
                                     font=("Segoe UI", 8), padx=10)
        self.cancel_btn.pack(side=tk.LEFT)

        self.launch_btn = tk.Button(btn_frame, text="Launch DeepSeek‑V4", state=tk.DISABLED,
                                     command=self.launch_main_app,
                                     bg="#3c7fb1", fg="white", activebackground="#1f4f7a",
                                     relief=tk.RAISED, font=("Segoe UI", 8, "bold"), padx=15)
        self.launch_btn.pack(side=tk.RIGHT)

        # Start checking dependencies automatically
        self.after(100, self.check_dependencies)

        # Start log updater
        self.after(100, self.process_log_queue)

    def log(self, message: str):
        """Thread‑safe logging."""
        self.log_queue.put(message + "\n")

    def process_log_queue(self):
        """Update log text from queue."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self.process_log_queue)

    def check_dependencies(self):
        """Check which packages are already installed."""
        self.log("Checking installed packages...")
        missing = []
        for pkg in self.packages:
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                missing.append(pkg)
                self.log(f"  {pkg} → NOT found")
            else:
                self.log(f"  {pkg} → found")
        if missing:
            self.log(f"\nMissing packages: {', '.join(missing)}")
            self.install_needed = True
            self.install_btn.config(state=tk.NORMAL)
        else:
            self.log("\n✅ All required packages are already installed.")
            self.install_btn.config(state=tk.DISABLED)
            self.launch_btn.config(state=tk.NORMAL)

    def start_installation(self):
        """Run pip install for missing packages in a thread."""
        self.install_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.log("\n=== Starting installation ===\n")
        thread = threading.Thread(target=self._install_thread, daemon=True)
        thread.start()

    def _install_thread(self):
        """Background installation using pip."""
        try:
            for pkg in self.packages:
                spec = importlib.util.find_spec(pkg)
                if spec is not None:
                    self.log(f"Skipping {pkg} (already installed)")
                    continue

                self.log(f"Installing {pkg}...")
                cmd = [sys.executable, "-m", "pip", "install", pkg]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           universal_newlines=True, bufsize=1)

                for line in process.stdout:
                    self.log(line.rstrip())
                process.wait()

                if process.returncode != 0:
                    self.log(f"❌ Failed to install {pkg}. Please install manually.")
                    self.after(0, self._installation_failed)
                    return
                else:
                    self.log(f"✅ {pkg} installed successfully.\n")

            self.log("\n🎉 All packages installed!")
            self.after(0, self._installation_complete)

        except Exception as e:
            self.log(f"❌ Unexpected error: {e}")
            self.after(0, self._installation_failed)

    def _installation_complete(self):
        """Called when installation finishes successfully."""
        self.progress.stop()
        self.progress.config(mode='determinate', value=100)
        self.cancel_btn.config(state=tk.NORMAL)
        self.launch_btn.config(state=tk.NORMAL)
        self.installation_successful = True

    def _installation_failed(self):
        """Called when installation fails."""
        self.progress.stop()
        self.cancel_btn.config(state=tk.NORMAL)
        self.install_btn.config(state=tk.NORMAL)
        self.log("\n⚠ Installation incomplete. You may need to install dependencies manually.")

    def launch_main_app(self):
        """Close wizard and start the main DeepSeek‑V4 chat app."""
        self.installation_successful = True
        self.destroy()


# ==================== REAL CHATBOT BACKEND (local LLM or rule-based) ====================
class RealChatBackend:
    """Real chatbot: tries local transformers model first, then rule-based responses."""

    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def __init__(self):
        self.history = []
        self._dualpath = None
        self.config = type("Config", (), {"max_tokens": 512, "temperature": 0.7})()
        self.tokenizer = None
        self.model = None
        self._ready = False
        self._load_error = None
        self._load_model()

    def _load_model(self):
        try:
            mod = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            self.tokenizer = getattr(mod, "AutoTokenizer").from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = getattr(mod, "AutoModelForCausalLM").from_pretrained(
                self.MODEL_ID,
                torch_dtype=getattr(torch, "float16", None),
                device_map="auto",
                trust_remote_code=True,
            )
            self._ready = True
        except Exception as e:
            self._ready = False
            self._load_error = str(e)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            return "Please type a message and I'll reply."
        if self._ready and self.model and self.tokenizer:
            try:
                torch = importlib.import_module("torch")
                system = "You are a helpful chatbot. Be concise and friendly."
                turns = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history[-6:]])
                text = f"{system}\n\n{turns}\nUser: {prompt}\nAssistant:" if turns else f"{system}\n\nUser: {prompt}\nAssistant:"
                inp = self.tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inp = {k: v.to(self.model.device) for k, v in inp.items()}
                with torch.no_grad():
                    out = self.model.generate(**inp, max_new_tokens=min(256, self.config.max_tokens), temperature=self.config.temperature, do_sample=True, top_p=0.9)
                reply = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if "Assistant:" in reply:
                    reply = reply.split("Assistant:", 1)[-1].strip()
                reply = reply.strip() or "I'm not sure what to say."
                self.history.append({"user": prompt, "assistant": reply})
                return reply
            except Exception as e:
                return self._rule_reply(prompt) + f"\n\n(Inference error: {e})"
        return self._rule_reply(prompt)

    def _rule_reply(self, prompt: str) -> str:
        """Rule-based replies when no model is loaded."""
        p = prompt.lower().strip()
        if not p:
            return "Say something and I'll try to help."
        if any(w in p for w in ("hello", "hi", "hey", "greetings")):
            return "Hello! How can I help you today?"
        if any(w in p for w in ("bye", "goodbye", "see you")):
            return "Goodbye! Take care."
        if any(w in p for w in ("thank", "thanks")):
            return "You're welcome!"
        if "name" in p or "who are you" in p:
            return "I'm a local chatbot. Install torch and transformers and I'll use a real language model."
        if "?" in prompt:
            return f"I'm thinking about your question: «{prompt[:60]}…». For full answers, install: pip install torch transformers accelerate, then restart."
        return f"I heard you: «{prompt[:80]}{'…' if len(prompt) > 80 else ''}». For real LLM replies, install torch and transformers."


# ==================== REAL CHATBOT GUI ====================
class RealChatbotApp(tk.Tk):
    """Full real chatbot window: local LLM when available, rule-based otherwise."""

    def __init__(self):
        super().__init__()
        self.title("BitNet + DeepSeek‑V4 · Real Chatbot")
        self.geometry("900x600")
        self.minsize(600, 400)
        self.configure(bg="#020617")
        self.llm = RealChatBackend()
        self._current_thread = None
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self, bg="#020617", height=44)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header, text="Real Chatbot", fg="#e5e7eb", bg="#020617", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=14, pady=8)
        tk.Label(header, text="Local LLM · BitNet + DeepSeek‑V4", fg="#9ca3af", bg="#111827", font=("Segoe UI", 9), padx=8, pady=4).pack(side=tk.LEFT, padx=(8, 0), pady=8)
        tk.Button(header, text="Clear chat", command=self._clear, bg="#020617", fg="#9ca3af", relief=tk.FLAT, padx=8).pack(side=tk.RIGHT, padx=14, pady=8)
        main = tk.Frame(self, bg="#020617", padx=12, pady=8)
        main.pack(fill=tk.BOTH, expand=True)
        self.chat_box = scrolledtext.ScrolledText(main, wrap=tk.WORD, state=tk.DISABLED, bg="#020617", fg="#e5e7eb", insertbackground="#e5e7eb", font=("Segoe UI", 10), relief=tk.FLAT)
        self.chat_box.pack(fill=tk.BOTH, expand=True)
        row = tk.Frame(self, bg="#020617", padx=12, pady=(0, 12))
        row.pack(side=tk.BOTTOM, fill=tk.X)
        self.input_text = tk.Text(row, height=3, bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb", font=("Segoe UI", 10), wrap=tk.WORD, relief=tk.FLAT)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), pady=4)
        self.input_text.bind("<Return>", self._on_enter)
        btns = tk.Frame(row, bg="#020617")
        btns.pack(side=tk.RIGHT, padx=(0, 4), pady=4)
        tk.Button(btns, text="Send", command=self._on_send, bg="#2563eb", fg="white", relief=tk.FLAT, padx=14, pady=6).pack(side=tk.TOP, fill=tk.X)
        status = "Real chatbot: local model loaded." if (self.llm._ready) else "Real chatbot (rule-based until torch/transformers installed)."
        tk.Label(row, text=status, fg="#6b7280", bg="#020617", font=("Segoe UI", 8)).pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        self._append("system", "Welcome. I'm a real chatbot — using a local LLM when available. Type below and press Enter to send (Shift+Enter for new line).")
        self.after(150, lambda: self.input_text.focus_set())

    def _append(self, who: str, msg: str):
        self.chat_box.config(state=tk.NORMAL)
        if self.chat_box.index("end-1c") != "1.0":
            self.chat_box.insert(tk.END, "\n")
        if who == "You":
            self.chat_box.insert(tk.END, f"You:\n", "user_label")
            self.chat_box.insert(tk.END, msg + "\n", "user")
        elif who == "Assistant":
            self.chat_box.insert(tk.END, f"Assistant:\n", "assistant_label")
            self.chat_box.insert(tk.END, msg + "\n", "assistant")
        else:
            self.chat_box.insert(tk.END, msg + "\n", "system")
        self.chat_box.tag_config("user_label", foreground="#a5b4fc", font=("Segoe UI", 9, "bold"))
        self.chat_box.tag_config("assistant_label", foreground="#6ee7b7", font=("Segoe UI", 9, "bold"))
        self.chat_box.tag_config("user", foreground="#e5e7eb", font=("Segoe UI", 10))
        self.chat_box.tag_config("assistant", foreground="#d1fae5", font=("Segoe UI", 10))
        self.chat_box.tag_config("system", foreground="#9ca3af", font=("Segoe UI", 9, "italic"))
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _on_enter(self, ev):
        if ev.state & 0x0001:
            return
        self._on_send()
        return "break"

    def _on_send(self):
        t = self.input_text.get("1.0", tk.END).strip()
        if not t:
            return
        self.input_text.delete("1.0", tk.END)
        self._append("You", t)
        if self._current_thread and self._current_thread.is_alive():
            return
        def run():
            reply = self.llm.generate(t)
            self.after(0, lambda: self._append("Assistant", reply))
        self._current_thread = threading.Thread(target=run, daemon=True)
        self._current_thread.start()

    def _clear(self):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)
        self.llm.history.clear()
        self._append("system", "Chat cleared. Continue the conversation below.")


# ==================== LOAD FULL CHAT APP FROM #r1.py OR REAL CHATBOT ====================
def _load_chat_app():
    """Prefer full app from #r1.py; otherwise run the built-in real chatbot."""
    import os
    _dir = os.path.dirname(os.path.abspath(__file__))
    r1_path = os.path.join(_dir, "#r1.py")
    if os.path.isfile(r1_path):
        try:
            spec = importlib.util.spec_from_file_location("r1_full", r1_path)
            r1_full = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(r1_full)
            return r1_full.DeepSeekV4ChatApp()
        except Exception:
            pass
    return RealChatbotApp()


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    wizard = XPSetupWizard()
    wizard.mainloop()

    if getattr(wizard, "installation_successful", False):
        app = _load_chat_app()
        app.mainloop()
