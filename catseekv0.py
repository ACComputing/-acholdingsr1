"""
CatBit1.x – Offline Version (Hugging Face removed)
ChatGPT-style GUI, local-only, no external dependencies beyond tkinter.
Uses a fixed mega-dataset for all responses.
"""
import tkinter as tk
from tkinter import scrolledtext
import threading
import re
from dataclasses import dataclass


# ==================== FIXED DATASET: Mega Distilled 7B Knowledge Base ====================
FIXED_WORD_DATASET = {
    # --- Original A-Z Everyday Terms ---
    "a": "A is for apple — a fruit that grows on trees.",
    "apple": "Apple: a round fruit, often red or green, that grows on trees.",
    "about": "About: concerning or regarding something.",
    "all": "All: the whole quantity or extent.",
    "and": "And: used to connect words or groups of words.",
    "animal": "Animal: a living creature that can move and feel.",
    "b": "B is for book — something you read.",
    "book": "Book: written or printed pages bound together.",
    "bird": "Bird: a creature with feathers and wings that can fly.",
    "blue": "Blue: a color like the sky on a clear day.",
    "big": "Big: large in size.",
    "c": "C is for cat — a furry animal that says meow.",
    "cat": "Cat: a small furry animal, often kept as a pet.",
    "car": "Car: a vehicle with wheels used for travel.",
    "computer": "Computer: an electronic machine that stores and processes data.",
    "cold": "Cold: low in temperature.",
    "d": "D is for dog — a loyal animal that barks.",
    "dog": "Dog: an animal often kept as a pet; barks and wags its tail.",
    "day": "Day: the time when the sun is up; 24 hours.",
    "door": "Door: something you open to enter or leave a room.",
    "e": "E is for elephant — a very large animal with a trunk.",
    "elephant": "Elephant: a very large gray animal with a long trunk.",
    "earth": "Earth: the planet we live on.",
    "eye": "Eye: the part of the body you see with.",
    "f": "F is for fish — an animal that lives in water.",
    "fish": "Fish: an animal that lives in water and has fins.",
    "friend": "Friend: someone you like and trust.",
    "food": "Food: what people and animals eat.",
    "g": "G is for garden — a place where plants grow.",
    "garden": "Garden: a place where flowers or vegetables are grown.",
    "green": "Green: the color of grass and leaves.",
    "good": "Good: of high quality or kind.",
    "h": "H is for house — a building where people live.",
    "house": "House: a building where people live.",
    "hello": "Hello: a friendly greeting. How can I help you today?",
    "help": "Help: I am ready to assist you. Ask me about a word, concept, or tech term.",
    "i": "I is for ice — frozen water.",
    "ice": "Ice: water that has frozen solid.",
    "idea": "Idea: a thought or suggestion.",
    "important": "Important: meaning a lot; significant.",
    "j": "J is for jump — to leave the ground with your feet.",
    "jump": "Jump: to push yourself off the ground into the air.",
    "joy": "Joy: a feeling of great happiness.",
    "k": "K is for kite — a toy that flies in the wind.",
    "kite": "Kite: a light toy that flies in the wind on a string.",
    "king": "King: a male ruler of a country.",
    "kind": "Kind: friendly and caring.",
    "l": "L is for light — what lets you see in the dark.",
    "light": "Light: the opposite of dark; what comes from the sun or a lamp.",
    "love": "Love: a strong feeling of caring for someone.",
    "learn": "Learn: to get new knowledge or skill.",
    "m": "M is for moon — what shines in the night sky.",
    "moon": "Moon: the round object that shines in the night sky.",
    "music": "Music: sounds made in a pattern that you can enjoy.",
    "mother": "Mother: a female parent.",
    "n": "N is for night — when the sun is down.",
    "night": "Night: the time when it is dark and the sun is down.",
    "name": "Name: what someone or something is called.",
    "new": "New: not old; just made or begun.",
    "o": "O is for ocean — a very large body of salt water.",
    "ocean": "Ocean: a very large body of salt water.",
    "open": "Open: not closed; allowing things to go in or out.",
    "p": "P is for plant — a living thing that grows from the ground.",
    "plant": "Plant: a living thing that grows from the ground and often has leaves.",
    "sun": "Sun: the star that gives Earth light and warmth.",
    "people": "People: human beings.",
    "please": "Please: a polite word when asking for something.",
    "q": "Q is for queen — a female ruler.",
    "queen": "Queen: a female ruler of a country.",
    "question": "Question: a sentence that asks something.",
    "quiet": "Quiet: making little or no sound.",
    "r": "R is for rain — water that falls from clouds.",
    "rain": "Rain: water that falls from clouds when it storms.",
    "red": "Red: a color like blood or a ripe apple.",
    "read": "Read: to look at words and understand them.",
    "s": "S is for star — a bright point in the night sky.",
    "star": "Star: a bright point of light in the night sky.",
    "sky": "Sky: the space above the earth where clouds and the sun are.",
    "school": "School: a place where people go to learn.",
    "t": "T is for tree — a tall plant with a trunk and leaves.",
    "tree": "Tree: a tall plant with a trunk, branches, and leaves.",
    "time": "Time: minutes, hours, days, and years. Also the 4th dimension.",
    "thank": "Thank: to say you are grateful. You're welcome!",
    "thanks": "Thanks: short for thank you. You're welcome!",
    "u": "U is for umbrella — something that keeps you dry in rain.",
    "umbrella": "Umbrella: something you hold over your head to stay dry in rain.",
    "under": "Under: below or beneath something.",
    "v": "V is for violin — a musical instrument you play with a bow.",
    "violin": "Violin: a musical instrument with strings and a bow.",
    "very": "Very: to a high degree; extremely.",
    "w": "W is for water — what we drink and that fills the ocean.",
    "water": "Water: a clear liquid (H2O) essential for life.",
    "world": "World: the earth and all the people and things on it.",
    "white": "White: the color of snow or milk.",
    "x": "X is for xylophone — a musical instrument with bars.",
    "xylophone": "Xylophone: a musical instrument with wooden or metal bars you hit.",
    "y": "Y is for yellow — a bright color like the sun.",
    "yellow": "Yellow: a bright color like the sun or a lemon.",
    "yes": "Yes: a word used to agree or say something is true.",
    "you": "You: the person I am talking to.",
    "z": "Z is for zebra — an animal with black and white stripes.",
    "zebra": "Zebra: an animal with black and white stripes that looks like a horse.",
    "zero": "Zero: the number 0; nothing.",
    "bye": "Bye: short for goodbye. See you later!",
    "goodbye": "Goodbye: a word you say when leaving. Take care!",
    "hi": "Hi: a casual way to say hello. Hello back!",
    "hey": "Hey: an informal greeting. Hi there!",

    # --- Expanded Tech, AI, and Science Terminology (DeepSeek 7B Optimization) ---
    "ai": "AI (Artificial Intelligence): The simulation of human intelligence processes by machines, especially computer systems.",
    "algorithm": "Algorithm: A step-by-step procedure or set of rules used for calculations, data processing, and automated reasoning.",
    "api": "API (Application Programming Interface): A set of protocols for building and integrating application software.",
    "attention": "Attention Mechanism: In neural networks, a technique that allows models to dynamically focus on different parts of the input text.",
    "blackhole": "Black Hole: A region of spacetime where gravity is so strong that nothing, not even light, can escape.",
    "byte": "Byte: A unit of digital data typically consisting of eight bits.",
    "cpu": "CPU (Central Processing Unit): The primary component of a computer that acts as its 'brain', executing instructions.",
    "data": "Data: Information translated into a form that is efficient for movement or processing.",
    "deepseek": "DeepSeek: An advanced series of AI models known for strong reasoning, coding, and mathematical capabilities, ranging up to hundreds of billions of parameters.",
    "deeplearning": "Deep Learning: A subset of machine learning based on artificial neural networks with multiple layers.",
    "dimension": "Dimension: A measurable extent of some kind, such as length, breadth, depth, or time.",
    "dna": "DNA (Deoxyribonucleic Acid): The molecule that carries genetic instructions in all known living organisms.",
    "energy": "Energy: The quantitative property that must be transferred to a body or physical system to perform work (E=mc²).",
    "entropy": "Entropy: A measure of disorder, randomness, or uncertainty within a closed system.",
    "evolution": "Evolution: The process by which different kinds of living organisms developed and diversified from earlier forms.",
    "galaxy": "Galaxy: A gravitationally bound system of stars, stellar remnants, interstellar gas, dust, and dark matter.",
    "gpu": "GPU (Graphics Processing Unit): A specialized electronic circuit designed to rapidly manipulate memory, highly useful in AI training.",
    "gravity": "Gravity: A fundamental interaction which causes mutual attraction between all things with mass or energy.",
    "hardware": "Hardware: The physical components that make up a computer system.",
    "html": "HTML: HyperText Markup Language, the standard markup language for documents designed to be displayed in a web browser.",
    "internet": "Internet: A global system of interconnected computer networks that uses the standard Internet protocol suite.",
    "java": "Java: A high-level, class-based, object-oriented programming language.",
    "javascript": "JavaScript: A programming language that is one of the core technologies of the World Wide Web.",
    "linux": "Linux: A family of open-source Unix-like operating systems based on the Linux kernel.",
    "llm": "LLM (Large Language Model): A deep learning algorithm that can recognize, summarize, translate, predict and generate text and other content based on knowledge gained from massive datasets.",
    "machinelearning": "Machine Learning: A branch of AI based on the idea that systems can learn from data, identify patterns and make decisions.",
    "math": "Math: The abstract science of number, quantity, and space.",
    "matrix": "Matrix: A rectangular array or table of numbers, symbols, or expressions, arranged in rows and columns.",
    "molecule": "Molecule: A group of atoms bonded together, representing the smallest fundamental unit of a chemical compound.",
    "neural": "Neural Network: A computational model inspired by the human brain, used extensively in AI and deep learning.",
    "network": "Network: A collection of computers, servers, or other devices connected to one another to share data.",
    "os": "Operating System (OS): System software that manages computer hardware, software resources, and provides common services for programs.",
    "parameter": "Parameter: In AI, the variables (weights and biases) that a model learns during training. e.g., '7B' means 7 Billion parameters.",
    "physics": "Physics: The natural science that studies matter, its fundamental constituents, its motion and behavior through space and time.",
    "python": "Python: A high-level, general-purpose programming language known for its readability and widespread use in AI/data science.",
    "quantum": "Quantum: The minimum amount of any physical entity involved in an interaction.",
    "ram": "RAM (Random Access Memory): A form of computer memory that can be read and changed in any order, typically used to store working data.",
    "robot": "Robot: A machine capable of carrying out a complex series of actions automatically.",
    "science": "Science: The intellectual and practical activity encompassing the systematic study of the physical and natural world.",
    "software": "Software: A set of instructions, data or programs used to operate computers and execute specific tasks.",
    "space": "Space: The boundless three-dimensional extent in which objects and events have relative position and direction.",
    "sql": "SQL (Structured Query Language): A domain-specific language used in programming and designed for managing data held in a relational database.",
    "temperature": "Temperature (AI): A hyperparameter that controls the randomness of an AI's responses. Lower = predictable, Higher = creative.",
    "token": "Token: The basic unit of data processed by an LLM, roughly equivalent to a word or part of a word.",
    "transformer": "Transformer: A deep learning architecture introduced in 2017 that relies entirely on self-attention mechanisms, forming the basis of modern LLMs.",
    "universe": "Universe: All of space and time and their contents, including planets, stars, galaxies, and all other forms of matter and energy.",
    "velocity": "Velocity: The directional speed of an object in motion as an indication of its rate of change in position.",
    "virus": "Virus (Computer): A type of malicious software that, when executed, replicates itself by modifying other computer programs.",
    "web": "Web: The World Wide Web, an information system where documents and other web resources are identified by URLs.",
    "wifi": "Wi-Fi: A family of wireless network protocols based on the IEEE 802.11 family of standards.",
    "zetta": "Zettabyte: A multiple of the unit byte for digital information, equivalent to one sextillion bytes."
}

FIXED_DATASET_DEFAULT = (
    "I am operating in offline mode using the distilled mega-dataset. "
    "I recognize basic A-Z words, math, programming, and AI terms (e.g., Python, deepseek, 7b, algorithm, physics). "
    "Try a single keyword to get a definition."
)


# ==================== CatBit1.x – Offline LLM (Hugging Face removed) ====================
@dataclass
class CatBit1Config:
    name: str = "CatBit1.x (Offline)"
    description: str = "Fixed dataset only – no Hugging Face"
    max_tokens: int = 512  # Not used, kept for compatibility
    temperature: float = 0.7
    top_p: float = 0.9


class CatBit1xLLM:
    """
    Offline version – no Hugging Face dependencies.
    Responds exclusively from the built‑in FIXED_WORD_DATASET.
    """

    def __init__(self, config: CatBit1Config | None = None):
        self.config = config or CatBit1Config()
        self.history: list[dict[str, str]] = []
        self._dataset = FIXED_WORD_DATASET
        self._default = FIXED_DATASET_DEFAULT

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            return "Type a message. I'll reply using the offline knowledge base."

        reply = self._lookup_fixed(prompt.strip())
        self.history.append({"user": prompt, "assistant": reply})
        return reply

    def _lookup_fixed(self, text: str) -> str:
        """
        Advanced lookup from mega-dataset.
        Matches full string first, then keyword extraction,
        and provides procedural definitions for completely unknown terms.
        """
        lower = text.lower().strip()

        # 1. Exact Match Check
        if lower in self._dataset:
            return self._dataset[lower]

        # 2. Extract alphanumeric word match
        word = re.sub(r"[^\w]", "", lower)
        if word and word in self._dataset:
            return self._dataset[word]

        # 3. Multi-word scan
        words = re.findall(r"[a-z0-9]+", lower)

        # Simple math evaluation (safe subset)
        if any(char.isdigit() for char in lower) and any(op in lower for op in ['+', '-', '*', '/']):
            try:
                # Restrict to numbers and basic operators
                safe_math = re.sub(r"[^0-9\+\-\*\/\(\)\.]", "", lower)
                result = eval(safe_math)
                return f"Math Calculation: {safe_math} = {result}"
            except Exception:
                pass

        # Check existing dictionary words
        for w in words:
            if w in self._dataset:
                return f"I found a concept in your query:\n{self._dataset[w]}"

        # 4. Procedural fallback (structured definition placeholder)
        if len(words) > 0 and len(words) < 5:
            subject = " ".join(words).title()
            return f"{subject}: I am operating offline with a fixed knowledge base. {self._default}"

        return self._default


# ==================== ChatGPT-style GUI (unchanged) ====================
class CatBitChatApp(tk.Tk):
    """ChatGPT-like UI: header, scrollable chat, bottom input with Send."""

    def __init__(self):
        super().__init__()
        self.title("CatBit1.x · Offline (Hugging Face removed)")
        self.geometry("900x650")
        self.minsize(600, 450)
        self.configure(bg="#ffffff")
        self.llm = CatBit1xLLM()
        self._thread: threading.Thread | None = None
        self._build_gui()

    def _build_gui(self):
        # Top bar
        top = tk.Frame(self, bg="#f7f7f8", height=52)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top, text="CatBit1.x", fg="#0d0d0d", bg="#f7f7f8",
            font=("Segoe UI", 14, "bold")
        ).pack(side=tk.LEFT, padx=16, pady=14)

        self._subtitle = tk.Label(
            top, text="Offline · Fixed Mega‑Dataset", fg="#6e6e80", bg="#f7f7f8",
            font=("Segoe UI", 10)
        )
        self._subtitle.pack(side=tk.LEFT, padx=(0, 12), pady=14)

        tk.Button(
            top, text="New chat", command=self._clear_chat,
            bg="#f7f7f8", fg="#0d0d0d", relief=tk.FLAT, font=("Segoe UI", 9),
            activebackground="#e5e5e5", padx=10, pady=4
        ).pack(side=tk.RIGHT, padx=16, pady=14)

        # Chat area
        chat_frame = tk.Frame(self, bg="#ffffff", padx=16, pady=12)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#ffffff", fg="#0d0d0d", insertbackground="#0d0d0d",
            font=("Segoe UI", 11), relief=tk.FLAT, borderwidth=0
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        # Bottom input
        bottom = tk.Frame(self, bg="#ffffff", padx=16, pady=12)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        input_frame = tk.Frame(bottom, bg="#f7f7f8", highlightbackground="#c5c5d1", highlightthickness=1)
        input_frame.pack(fill=tk.X)

        self.input_text = tk.Text(
            input_frame, height=3, wrap=tk.WORD, bg="#f7f7f8", fg="#0d0d0d",
            insertbackground="#0d0d0d", font=("Segoe UI", 11), relief=tk.FLAT,
            padx=12, pady=10
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_text.bind("<Return>", self._on_return)

        send_btn = tk.Button(
            input_frame, text="Send", command=self._send,
            bg="#10a37f", fg="white", relief=tk.FLAT, font=("Segoe UI", 10, "bold"),
            activebackground="#0d8c6d", padx=16, pady=8, cursor="hand2"
        )
        send_btn.pack(side=tk.RIGHT, padx=8, pady=8)

        self._append_system(
            "Welcome to CatBit1.x (Hugging Face removed). I'm running offline on the fixed Mega‑Dataset. "
            "Ask about words, concepts, AI terms, or simple math equations!"
        )
        self.after(100, lambda: self.input_text.focus_set())

    def _append_system(self, msg: str):
        self._append("system", "", msg)

    def _append(self, role: str, label: str, msg: str):
        self.chat.config(state=tk.NORMAL)
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert(tk.END, "\n")

        if role == "user":
            self.chat.insert(tk.END, "You\n", "user_label")
            self.chat.insert(tk.END, msg + "\n\n", "user")
        elif role == "assistant":
            self.chat.insert(tk.END, "CatBit1.x\n", "assistant_label")
            self.chat.insert(tk.END, msg + "\n\n", "assistant")
        else:
            self.chat.insert(tk.END, msg + "\n\n", "system")

        self.chat.tag_config("user_label", font=("Segoe UI", 10, "bold"), foreground="#0d0d0d")
        self.chat.tag_config("assistant_label", font=("Segoe UI", 10, "bold"), foreground="#10a37f")
        self.chat.tag_config("user", font=("Segoe UI", 11), foreground="#0d0d0d")
        self.chat.tag_config("assistant", font=("Segoe UI", 11), foreground="#0d0d0d")
        self.chat.tag_config("system", font=("Segoe UI", 10), foreground="#6e6e80")

        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _on_return(self, ev):
        if ev.state & 0x0001:
            return
        self._send()
        return "break"

    def _send(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return

        self.input_text.delete("1.0", tk.END)
        self._append("user", "You", text)

        if self._thread and self._thread.is_alive():
            return

        def run():
            reply = self.llm.generate(text)
            self.after(0, lambda: self._append("assistant", "CatBit1.x", reply))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _clear_chat(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.config(state=tk.DISABLED)
        self.llm.history.clear()
        self._append_system("Chat cleared. Start a new conversation.")


if __name__ == "__main__":
    app = CatBitChatApp()
    app.mainloop()
