import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import subprocess
import sys
import importlib.util
import queue
import time
import random
import re
import math
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ==================== XP‑STYLE DEPENDENCY INSTALLER ====================
class CatR1SetupWizard(tk.Tk):
    """600x400 Windows XP‑style wizard that installs Python dependencies."""
    def __init__(self):
        super().__init__()
        self.title("CatR1 1.X Setup Wizard")
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

        # ===== XP‑style title bar =====
        title_frame = tk.Frame(self, bg="#0058e7", height=30)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(title_frame, text="CatR1 1.X Setup Wizard", fg="white",
                                bg="#0058e7", font=("Segoe UI", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)

        # ===== Main content =====
        main = tk.Frame(self, bg="#ece9d8", padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instr = tk.Label(main, text="This wizard will install required Python packages for CatR1 1.X.",
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

        self.launch_btn = tk.Button(btn_frame, text="Launch CatR1 1.X", state=tk.DISABLED,
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
        """Close wizard and start the main CatR1 1.X chat app."""
        self.installation_successful = True
        self.destroy()


# ==================== CatR1 1.X ENGINE – FULL SIMULATION ====================
# Based on the earlier DeepSeek‑V4 framework, now rebranded as CatR1 1.X.
# PD‑disaggregated, DualPath KV‑cache, 1M context, bilingual.

@dataclass
class CatR1Config:
    name: str = "CatR1‑1.X‑14B"
    max_tokens: int = 4096
    temperature: float = 0.8
    top_k: int = 40
    reasoning_depth: int = 3
    max_history: int = 10
    # PD‑disaggregated hardware specs
    num_prefill_engines: int = 4
    num_decode_engines: int = 8
    storage_bandwidth_gbps: float = 200.0
    rdma_bandwidth_gbps: float = 800.0
    kv_cache_hit_rate: float = 0.96
    hbm_bandwidth_gbps: float = 3000.0

class DualPathEngine:
    """
    Simulates the CatR1 1.X DualPath inference framework:
    - Prefill (PE) and Decode (DE) engines with disaggregated architecture
    - Dual‑path KV‑cache loading: Storage→PE (legacy) vs Storage→DE→PE (novel)
    - Central traffic manager with load‑aware path selection
    - Performance multipliers: 1.87x offline, 1.96x online throughput
    """
    def __init__(self, config: CatR1Config):
        self.config = config
        self.pe_nodes = [f"PE_{i}" for i in range(config.num_prefill_engines)]
        self.de_nodes = [f"DE_{i}" for i in range(config.num_decode_engines)]
        self.pe_load = {node: random.uniform(0.15, 0.35) for node in self.pe_nodes}
        self.de_load = {node: random.uniform(0.10, 0.25) for node in self.de_nodes}
        self.storage_nic_usage = {node: 0.0 for node in self.pe_nodes + self.de_nodes}
        self.requests_processed = 0
        self.total_path_a_time = 0.0
        self.total_path_b_time = 0.0
        self.path_a_count = 0
        self.path_b_count = 0

    def _compute_context_penalty(self, context_length: int) -> float:
        return math.log2(1 + context_length / 2048) * 0.15

    def select_path(self, context_length: int, kv_cache_size_mb: float) -> Tuple[str, float, str]:
        pe_pressure = sum(self.pe_load.values()) / len(self.pe_nodes)
        de_pressure = sum(self.de_load.values()) / len(self.de_nodes)

        # Path A: Storage → PE
        path_a_time = (kv_cache_size_mb * 8 / self.config.storage_bandwidth_gbps) * 1000
        path_a_time += self._compute_context_penalty(context_length)
        path_a_time *= (1 + 0.8 * pe_pressure)

        # Path B: Storage → DE → PE
        effective_bw = min(self.config.storage_bandwidth_gbps, self.config.rdma_bandwidth_gbps)
        path_b_time = (kv_cache_size_mb * 8 / effective_bw) * 1000
        path_b_time += self._compute_context_penalty(context_length) * 0.7
        path_b_time *= (1 + 0.3 * de_pressure)

        # Long‑context speedups
        if context_length > 500000:
            path_b_time *= 0.53
        elif context_length > 100000:
            path_b_time *= 0.65
        elif context_length > 32000:
            path_b_time *= 0.8

        # Cache hit benefit
        if random.random() < self.config.kv_cache_hit_rate:
            path_a_time *= 0.5
            path_b_time *= 0.5

        path_a_time *= random.uniform(0.95, 1.05)
        path_b_time *= random.uniform(0.95, 1.05)

        if path_b_time < path_a_time * 1.1:
            path = "B (Storage→DE→PE)"
            est_time = path_b_time
            node = min(self.de_nodes, key=lambda n: self.de_load[n])
            self.storage_nic_usage[node] += kv_cache_size_mb * 8 / est_time
            self.de_load[node] += 0.03
            self.total_path_b_time += est_time
            self.path_b_count += 1
        else:
            path = "A (Storage→PE)"
            est_time = path_a_time
            node = min(self.pe_nodes, key=lambda n: self.pe_load[n])
            self.storage_nic_usage[node] += kv_cache_size_mb * 8 / est_time
            self.pe_load[node] += 0.03
            self.total_path_a_time += est_time
            self.path_a_count += 1

        self.requests_processed += 1
        return path, est_time, node

    def get_performance_stats(self) -> Dict[str, Any]:
        if self.requests_processed == 0:
            return {}
        avg_a = self.total_path_a_time / self.path_a_count if self.path_a_count else 0
        avg_b = self.total_path_b_time / self.path_b_count if self.path_b_count else 0
        speedup = avg_a / avg_b if avg_b > 0 else 1.0
        return {
            "requests_processed": self.requests_processed,
            "path_a_count": self.path_a_count,
            "path_b_count": self.path_b_count,
            "avg_path_a_time_ms": round(avg_a, 2),
            "avg_path_b_time_ms": round(avg_b, 2),
            "speedup_vs_traditional": round(speedup, 2),
            "paper_offline_speedup": 1.87,
            "paper_online_speedup": 1.96,
            "storage_nic_util": {
                node: round(usage / self.config.storage_bandwidth_gbps, 2)
                for node, usage in self.storage_nic_usage.items()
            },
            "pe_load_avg": round(sum(self.pe_load.values()) / len(self.pe_nodes), 2),
            "de_load_avg": round(sum(self.de_load.values()) / len(self.de_nodes), 2),
        }

class NGramModel:
    """Bilingual n‑gram generator."""
    def __init__(self, sentences: List[str], max_order: int = 3):
        self.max_order = max_order
        self.ngrams: List[Dict[Tuple[str, ...], Counter]] = [defaultdict(Counter) for _ in range(max_order + 1)]
        self.vocab = set()
        self._build(sentences)

    def _build(self, sentences: List[str]):
        for sent in sentences:
            words = sent.split()
            if len(words) < 2:
                continue
            self.vocab.update(words)
            for order in range(1, self.max_order + 1):
                for i in range(len(words) - order):
                    key = tuple(words[i:i+order])
                    next_word = words[i+order]
                    self.ngrams[order][key][next_word] += 1

    def _get_prob_dist(self, context: List[str], order: int, temperature: float) -> Dict[str, float]:
        for o in range(min(order, self.max_order), 0, -1):
            if len(context) < o:
                continue
            key = tuple(context[-o:])
            if key in self.ngrams[o]:
                counts = self.ngrams[o][key]
                total = sum(counts.values())
                vocab_size = len(self.vocab)
                smoothed = {w: (c + 0.01) / (total + 0.01 * vocab_size) for w, c in counts.items()}
                if temperature != 1.0:
                    logits = {w: math.log(p + 1e-10) / temperature for w, p in smoothed.items()}
                    max_logit = max(logits.values())
                    exp_logits = {w: math.exp(l - max_logit) for w, l in logits.items()}
                    sum_exp = sum(exp_logits.values())
                    return {w: v / sum_exp for w, v in exp_logits.items()}
                return smoothed
        uniform_prob = 1.0 / len(self.vocab)
        return {w: uniform_prob for w in self.vocab}

    def generate(self, seed: Optional[List[str]] = None, length: int = 20,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        if not self.vocab:
            return "..."
        if seed is None:
            if self.ngrams[2]:
                start_key = random.choice(list(self.ngrams[2].keys()))
                output = list(start_key)
            else:
                output = [random.choice(list(self.vocab))]
        else:
            output = seed[:]
        for _ in range(length):
            context = output[-(self.max_order):]
            probs = self._get_prob_dist(context, self.max_order, temperature)
            if top_k > 0 and len(probs) > top_k:
                sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                probs = dict(sorted_words[:top_k])
                total = sum(probs.values())
                probs = {w: p / total for w, p in probs.items()}
            words = list(probs.keys())
            probs_list = list(probs.values())
            next_word = random.choices(words, weights=probs_list)[0]
            output.append(next_word)
        return " ".join(output)

class CatR1:
    """
    CatR1 1.X – bilingual model with PD‑disaggregated DualPath inference.
    Supports 1M token context (simulated) and paper‑reported speedups.
    """
    def __init__(self, config: CatR1Config | None = None):
        self.config = config or CatR1Config()
        self.history: deque = deque(maxlen=self.config.max_history)

        # Expanded bilingual corpora
        en_corpus = [
            "I think therefore I am.", "The quick brown fox jumps over the lazy dog.",
            "To be or not to be that is the question.", "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step.",
            "The only thing we have to fear is fear itself.",
            "Ask not what your country can do for you ask what you can do for your country.",
            "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",
            "Life is like riding a bicycle to keep your balance you must keep moving.",
            "The important thing is not to stop questioning.",
            "I can calculate the motion of heavenly bodies but not the madness of people.",
            "Two things are infinite the universe and human stupidity and I am not sure about the universe.",
            "Be yourself everyone else is already taken.", "So many books so little time.",
            "You only live once but if you do it right once is enough.",
            "The future belongs to those who believe in the beauty of their dreams.",
            "It does not matter how slowly you go as long as you do not stop.",
            "Everything you can imagine is real.", "The only true wisdom is in knowing you know nothing.",
            "The journey not the arrival matters.", "Not all those who wander are lost.",
            "We are what we repeatedly do excellence then is not an act but a habit.",
            "Happiness is not something readymade It comes from your own actions.",
            "The purpose of our lives is to be happy.", "Get busy living or get busy dying.",
            "You have within you right now everything you need to deal with whatever the world can throw at you.",
            "Believe you can and you are halfway there.", "The future depends on what you do today.",
            "Do what you can with what you have where you are.", "It always seems impossible until it is done.",
            "Success is not final failure is not fatal It is the courage to continue that counts.",
        ]
        zh_corpus = [
            "学而时习之不亦说乎。", "有朋自远方来不亦乐乎。", "三人行必有我师焉。",
            "温故而知新可以为师矣。", "学而不思则罔思而不学则殆。",
            "知之者不如好之者好之者不如乐之者。", "逝者如斯夫不舍昼夜。",
            "君子坦荡荡小人长戚戚。", "道可道非常道名可名非常名。",
            "天下难事必作于易天下大事必作于细。", "千里之行始于足下。",
            "天行健君子以自强不息。", "地势坤君子以厚德载物。",
            "随风潜入夜润物细无声。", "床前明月光疑是地上霜举头望明月低头思故乡。",
            "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少。",
            "白日依山尽黄河入海流欲穷千里目更上一层楼。",
            "日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天。",
            "两个黄鹂鸣翠柳一行白鹭上青天窗含西岭千秋雪门泊东吴万里船。",
            "千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪。",
            "故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流。",
            "劝君更尽一杯酒西出阳关无故人。", "海内存知己天涯若比邻。",
            "独在异乡为异客每逢佳节倍思亲。", "但愿人长久千里共婵娟。",
            "人生自古谁无死留取丹心照汗青。", "先天下之忧而忧后天下之乐而乐。",
            "采菊东篱下悠然见南山。", "大漠孤烟直长河落日圆。",
            "会当凌绝顶一览众山小。", "问渠那得清如许为有源头活水来。",
        ]

        self.ngram_model = {
            "en": NGramModel(en_corpus, max_order=3),
            "zh": NGramModel(zh_corpus, max_order=2),
        }

        # Knowledge graph (CatR1‑branded)
        self.knowledge = {
            "greeting": {
                "en": {"responses": ["Hello! How can I assist you today?", "Hi there! What's on your mind?"],
                       "keywords": ["hello", "hi", "hey", "greetings"]},
                "zh": {"responses": ["你好！今天我能帮你什么？", "嗨！有什么想法吗？"],
                       "keywords": ["你好", "您好", "嗨"]}
            },
            "farewell": {
                "en": {"responses": ["Goodbye! Feel free to return anytime.", "Take care!"],
                       "keywords": ["bye", "goodbye", "see you"]},
                "zh": {"responses": ["再见！随时欢迎再来。", "保重！"],
                       "keywords": ["再见", "拜拜", "明天见"]}
            },
            "thanks": {
                "en": {"responses": ["You're welcome! Happy to help.", "My pleasure!"],
                       "keywords": ["thank", "thanks", "appreciate"]},
                "zh": {"responses": ["不客气！很高兴帮忙。", "我的荣幸！"],
                       "keywords": ["谢谢", "感谢", "多谢"]}
            },
            "identity": {
                "en": {"responses": ["I'm CatR1 1.X, a 14B parameter model with PD‑disaggregated DualPath inference and 1M context. [C] A.C Holdings 1999-2026"],
                       "keywords": ["your name", "who are you", "call you"]},
                "zh": {"responses": ["我是CatR1 1.X，一个140亿参数的模型，采用PD分离式双路推理架构，支持1M上下文。[C] A.C Holdings 1999-2026"],
                       "keywords": ["你叫什么", "你是谁", "怎么称呼你"]}
            },
            "capabilities": {
                "en": {"responses": ["I can handle 1M tokens, answer questions, generate text, and use DualPath for ultra‑fast inference."],
                       "keywords": ["what can you do", "capabilities", "help"]},
                "zh": {"responses": ["我能处理1M上下文、回答问题、生成文本，并通过双路推理实现极速响应。"],
                       "keywords": ["能做什么", "功能", "帮助"]}
            },
            "joke": {
                "en": {"responses": ["Why don't cats play poker in the jungle? Too many cheetahs!"],
                       "keywords": ["joke", "funny", "laugh"]},
                "zh": {"responses": ["为什么猫不喜欢玩扑克？因为有很多猎豹（cheetah，也指作弊者）！"],
                       "keywords": ["笑话", "搞笑", "幽默"]}
            },
            "time": {
                "en": {"responses": [f"The simulated time is {time.strftime('%I:%M %p')}."],
                       "keywords": ["time", "clock", "what time"]},
                "zh": {"responses": [f"模拟时间是 {time.strftime('%H:%M')}。"],
                       "keywords": ["时间", "几点", "现在几点"]}
            },
            "advice": {
                "en": {"responses": ["Always trust your instincts, but verify with data."],
                       "keywords": ["advice", "suggest", "what should"]},
                "zh": {"responses": ["相信你的直觉，但也要用数据验证。"],
                       "keywords": ["建议", "怎么办", "如何"]}
            },
            "story": {
                "en": {"responses": ["Once upon a time, in a land far away, there was a curious cat who loved to explore..."],
                       "keywords": ["story", "tell me a story", "tale"]},
                "zh": {"responses": ["从前，有一只喜欢探险的猫。一天，它发现了一座神奇的图书馆..."],
                       "keywords": ["故事", "讲个故事", "从前"]}
            }
        }

        self.fallback_templates = {
            "en": ["That's interesting. Could you tell me more about {topic}?",
                   "Let me generate something about {topic}: {markov}"],
            "zh": ["关于「{topic}」这很有趣。能再多说一点吗？",
                   "让我用「{topic}」生成一点内容：{markov}"]
        }

        # DualPath engine (always enabled)
        self.dualpath = DualPathEngine(self.config)

    def _detect_language(self, text: str) -> str:
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return "zh"
        return "en"

    def _extract_keywords(self, text: str, lang: str) -> List[str]:
        if lang == "en":
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
            stopwords = {"a","an","the","is","are","was","were","i","you","he","she","it","we","they",
                         "and","or","but","if","because","as","what","which","this","that","these","those",
                         "then","just","so","too","very","can","will","be","have","do","to","for","with",
                         "about","on","at","by","from","up","down","in","out","of","off","over","under"}
            return [w for w in words if w not in stopwords and len(w) > 2]
        else:
            chars = list(text)
            stopwords = {"的","了","是","在","我","你","他","她","它","我们","你们","他们",
                         "和","与","或","但是","如果","因为","所以","这","那","这些","那些",
                         "然后","就","太","很","能","会","将","有","做","也","都","被","把",
                         "对","于","而","并","便","虽","然","但","还","已","经"}
            return [ch for ch in chars if ch not in stopwords and not ch.isspace()]

    def _match_intent(self, text: str, lang: str) -> Optional[str]:
        text_lower = text.lower() if lang == "en" else text
        for node_id, node in self.knowledge.items():
            for kw in node[lang]["keywords"]:
                if kw in text_lower:
                    return node_id
        return None

    def _reason(self, prompt: str, lang: str, history: List[Dict]) -> str:
        lines = [f"🧠 Reasoning (depth {self.config.reasoning_depth}):"]
        lines.append(f"  • Query in {lang.upper()}: '{prompt[:50]}{'...' if len(prompt)>50 else ''}'")
        if history:
            last = history[-1] if history else None
            if last and last["sender"] == "user":
                lines.append(f"  • Previous: '{last['text'][:30]}...'")
        intent = self._match_intent(prompt, lang)
        if intent:
            lines.append(f"  • Detected intent: '{intent}' (confidence 0.9)")
        else:
            keywords = self._extract_keywords(prompt, lang)
            if keywords:
                lines.append(f"  • Extracted keywords: {keywords[:3]}")
            else:
                lines.append(f"  • No keywords found, using generative fallback.")
        # DualPath simulation for long contexts
        if len(prompt) > 500:
            context_len = len(prompt) * 4
            kv_size = context_len * 0.1
            path, est_time, node = self.dualpath.select_path(context_len, kv_size)
            lines.append(f"  • DualPath: Using path {path} via {node} (est. {est_time:.1f}ms)")
            lines.append(f"  • KV‑cache hit rate: {self.config.kv_cache_hit_rate*100:.0f}%")
        for i in range(self.config.reasoning_depth):
            if i == 0:
                lines.append(f"  • Step {i+1}: Analysing query structure...")
            elif i == 1:
                lines.append(f"  • Step {i+1}: Considering context window (1M)...")
            elif i == 2:
                lines.append(f"  • Step {i+1}: Formulating response with PD‑disaggregated engines...")
            else:
                lines.append(f"  • Step {i+1}: Refining...")
        return "\n".join(lines)

    def _generate_with_ngram(self, lang: str, seed_words: Optional[List[str]] = None) -> str:
        return self.ngram_model[lang].generate(
            seed=seed_words,
            length=15,
            temperature=self.config.temperature,
            top_k=self.config.top_k
        )

    def generate(self, prompt: str) -> str:
        base_sleep = random.uniform(0.2, 0.8)
        if len(prompt) > 1000:
            base_sleep *= 0.55
        time.sleep(base_sleep)

        if not prompt.strip():
            return "🐱 CatR1 1.X: Please say something – I'm listening. / 请说点什么 – 我在听。"

        lang = self._detect_language(prompt)
        intent = self._match_intent(prompt, lang)
        keywords = self._extract_keywords(prompt, lang)

        show_reasoning = random.random() < 0.3
        reasoning = self._reason(prompt, lang, list(self.history)) if show_reasoning else ""

        if intent:
            responses = self.knowledge[intent][lang]["responses"]
            base = random.choice(responses)
        else:
            if keywords:
                topic = keywords[0]
                if random.random() < 0.5:
                    markov_text = self._generate_with_ngram(lang, seed_words=[topic])
                    base = markov_text
                else:
                    templates = self.fallback_templates[lang]
                    template = random.choice(templates)
                    if "{markov}" in template:
                        markov_text = self._generate_with_ngram(lang)
                        base = template.format(topic=topic, markov=markov_text)
                    else:
                        base = template.format(topic=topic)
            else:
                base = self._generate_with_ngram(lang)

        if len(prompt) > 500 and not show_reasoning and random.random() < 0.3:
            stats = self.dualpath.get_performance_stats()
            if stats:
                speedup = stats.get("speedup_vs_traditional", 1.0)
                base += f"\n\n[⚡ DualPath: {speedup}x speedup vs traditional PD‑disaggregated (paper: 1.87–1.96x)]"

        emoji = "🐱"
        lang_tag = lang.upper()
        if show_reasoning:
            output = f"{emoji} CatR1 1.X (14B, {lang_tag}, 1M ctx):\n{reasoning}\n\n💬 {base}"
        else:
            output = f"{emoji} CatR1 1.X (14B, {lang_tag}, 1M ctx):\n💬 {base}"

        self.history.append({"sender": "user", "text": prompt, "lang": lang})
        self.history.append({"sender": "assistant", "text": output, "lang": lang})

        return output


# ==================== GUI (CatR1 1.X Chat) ====================
class CatR1ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CatR1 1.X [C] A.C Holdings 1999-2026")
        self.geometry("950x650")
        self.minsize(700, 500)

        try:
            self.style = ttk.Style(self)
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        self.configure(bg="#020617")

        self.llm = CatR1()
        self._current_thread: Optional[threading.Thread] = None

        self._build_layout()

    def _build_layout(self):
        header = tk.Frame(self, bg="#020617", height=48)
        header.pack(side=tk.TOP, fill=tk.X)

        title_label = tk.Label(
            header,
            text="CatR1 1.X",
            fg="#e5e7eb",
            bg="#020617",
            font=("Segoe UI", 14, "bold"),
        )
        title_label.pack(side=tk.LEFT, padx=14, pady=(10, 6))

        model_pill = tk.Label(
            header,
            text="14B · 1M ctx · PD‑disaggregated · DualPath KV‑cache · 1.96x throughput",
            fg="#e5e7eb",
            bg="#111827",
            font=("Segoe UI", 9),
            padx=10,
            pady=3,
        )
        model_pill.pack(side=tk.LEFT, padx=(8, 0), pady=(12, 6))

        stats_btn = tk.Button(
            header,
            text="DualPath Stats",
            command=self._show_dualpath_stats,
            bg="#020617",
            fg="#9ca3af",
            activebackground="#111827",
            activeforeground="#e5e7eb",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        stats_btn.pack(side=tk.RIGHT, padx=14, pady=(10, 6))

        clear_btn = tk.Button(
            header,
            text="Clear chat",
            command=self._clear_chat,
            bg="#020617",
            fg="#9ca3af",
            activebackground="#111827",
            activeforeground="#e5e7eb",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        clear_btn.pack(side=tk.RIGHT, padx=14, pady=(10, 6))

        main_frame = tk.Frame(self, bg="#020617")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(2, 4))

        self.chat_box = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        input_container = tk.Frame(self, bg="#020617")
        input_container.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 12))

        input_frame = tk.Frame(input_container, bg="#020617")
        input_frame.pack(side=tk.TOP, fill=tk.X)

        self.input_text = tk.Text(
            input_frame,
            height=3,
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4), pady=4)
        self.input_text.bind("<Return>", self._on_enter_pressed)

        buttons_frame = tk.Frame(input_frame, bg="#020617")
        buttons_frame.pack(side=tk.RIGHT, padx=(4, 0), pady=4)

        stop_btn = tk.Button(
            buttons_frame,
            text="Stop",
            command=self._stop_response,
            bg="#111827",
            fg="#e5e7eb",
            activebackground="#1f2937",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        )
        stop_btn.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        send_btn = tk.Button(
            buttons_frame,
            text="Send",
            command=self._on_send_click,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        )
        send_btn.pack(side=tk.TOP, fill=tk.X)

        hint_label = tk.Label(
            input_container,
            text="🐱 CatR1 1.X [C] A.C Holdings 1999-2026 · PD‑disaggregated + DualPath · 1.87–1.96x throughput · 1M context",
            fg="#6b7280",
            bg="#020617",
            font=("Segoe UI", 8),
            anchor="w",
        )
        hint_label.pack(side=tk.TOP, fill=tk.X, padx=(4, 4), pady=(2, 0))

        self._append_message(
            "system",
            "Welcome to CatR1 1.X – production inference framework simulation.\n"
            "• PD‑disaggregated: separate prefill (PE) and decode (DE) engines\n"
            "• Dual‑path KV‑cache loading: Storage→PE (legacy) vs Storage→DE→PE (new)\n"
            "• Uses idle DE NICs + RDMA to boost throughput by up to 1.96x\n"
            "• 1M token context – try long messages to see DualPath in action!\n"
            "[C] A.C Holdings 1999-2026"
        )

        self.after(200, lambda: self.input_text.focus_set())

    def _show_dualpath_stats(self):
        if self.llm.dualpath:
            stats = self.llm.dualpath.get_performance_stats()
            if stats:
                msg = "📊 CatR1 DualPath Performance:\n"
                msg += f"Requests processed: {stats['requests_processed']}\n"
                msg += f"Path A (Storage→PE) count: {stats['path_a_count']}, avg time: {stats['avg_path_a_time_ms']} ms\n"
                msg += f"Path B (Storage→DE→PE) count: {stats['path_b_count']}, avg time: {stats['avg_path_b_time_ms']} ms\n"
                msg += f"Speedup vs traditional: {stats['speedup_vs_traditional']}x\n"
                msg += f"(Paper reports: {stats['paper_offline_speedup']}x offline, {stats['paper_online_speedup']}x online)\n"
                msg += f"PE load avg: {stats['pe_load_avg']}, DE load avg: {stats['de_load_avg']}\n"
                msg += "Storage NIC utilisation:\n"
                for node, util in stats['storage_nic_util'].items():
                    msg += f"  {node}: {util*100:.1f}%\n"
                self._append_message("system", msg)
            else:
                self._append_message("system", "No DualPath stats yet – send some messages first!")
        else:
            self._append_message("system", "DualPath is disabled (should not happen).")

    def _append_message(self, sender: str, message: str):
        self.chat_box.config(state=tk.NORMAL)
        if sender == "user":
            label = "You"
            tag = "user"
        elif sender == "assistant":
            label = "CatR1"
            tag = "assistant"
        else:
            label = ""
            tag = "system"
        if self.chat_box.index("end-1c") != "1.0":
            self.chat_box.insert(tk.END, "\n")
        if label:
            self.chat_box.insert(tk.END, f"{label}:\n", (f"{tag}_label",))
        self.chat_box.insert(tk.END, message + "\n", (tag,))
        self.chat_box.tag_config("user_label", foreground="#a5b4fc", font=("Segoe UI", 9, "bold"))
        self.chat_box.tag_config("assistant_label", foreground="#6ee7b7", font=("Segoe UI", 9, "bold"))
        self.chat_box.tag_config("user", foreground="#e5e7eb", font=("Segoe UI", 10))
        self.chat_box.tag_config("assistant", foreground="#d1fae5", font=("Segoe UI", 10))
        self.chat_box.tag_config("system", foreground="#9ca3af", font=("Segoe UI", 9, "italic"))
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _on_enter_pressed(self, event):
        if event.state & 0x0001:
            return
        self._on_send_click()
        return "break"

    def _on_send_click(self):
        user_text = self.input_text.get("1.0", tk.END).strip()
        if not user_text:
            return
        self.input_text.delete("1.0", tk.END)
        self._append_message("user", user_text)
        if self._current_thread and self._current_thread.is_alive():
            return
        self._current_thread = threading.Thread(
            target=self._handle_llm_response,
            args=(user_text,),
            daemon=True,
        )
        self._current_thread.start()

    def _handle_llm_response(self, user_text: str):
        try:
            reply = self.llm.generate(user_text)
        except Exception as e:
            reply = f"(CatR1 backend error: {e})"
        self.after(0, lambda: self._append_message("assistant", reply))

    def _clear_chat(self):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)
        self.llm.history.clear()
        if self.llm.dualpath:
            self.llm.dualpath = DualPathEngine(self.llm.config)

    def _stop_response(self):
        pass


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    wizard = CatR1SetupWizard()
    wizard.mainloop()

    if wizard.installation_successful:
        app = CatR1ChatApp()
        app.mainloop()
