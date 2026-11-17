"""
CSS premium — ATS Intelligent
Design glassmorphism / 3D depth / dark space theme
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

/* ══════════════════════════════════════════════
   VARIABLES & RESET
══════════════════════════════════════════════ */
:root {
  --bg:        #030711;
  --bg2:       #080f1f;
  --bg3:       #0d1526;
  --glass:     rgba(255,255,255,0.04);
  --glass2:    rgba(255,255,255,0.07);
  --border:    rgba(255,255,255,0.08);
  --border2:   rgba(0,229,195,0.3);
  --teal:      #00e5c3;
  --teal2:     #00c4a8;
  --indigo:    #6366f1;
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #64748b;
  --red:       #f43f5e;
  --yellow:    #fbbf24;
  --shadow:    0 8px 32px rgba(0,0,0,0.6);
  --shadow2:   0 20px 60px rgba(0,0,0,0.7), 0 0 40px rgba(0,229,195,0.06);
  --radius:    16px;
  --radius2:   12px;
}

/* ══════════════════════════════════════════════
   BACKGROUND ANIMÉ
══════════════════════════════════════════════ */
@keyframes bgShift {
  0%   { background-position: 0% 0%;   }
  50%  { background-position: 100% 100%; }
  100% { background-position: 0% 0%;   }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0);    }
}
@keyframes glow {
  0%, 100% { box-shadow: 0 0 20px rgba(0,229,195,0.3); }
  50%       { box-shadow: 0 0 40px rgba(0,229,195,0.6); }
}
@keyframes spin { to { transform: rotate(360deg); } }

html, body {
  font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
  background-color: var(--bg) !important;
}
.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 10% 10%, rgba(99,102,241,0.08) 0%, transparent 50%),
    radial-gradient(ellipse 60% 50% at 90% 80%, rgba(0,229,195,0.06) 0%, transparent 50%),
    radial-gradient(ellipse 100% 100% at 50% 50%, #030711 0%, #030711 100%);
  background-attachment: fixed !important;
  font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
}
.stApp > * { animation: fadeUp 0.4s ease both; }

/* Force white text everywhere */
*, *::before, *::after {
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important;
}

/* ══════════════════════════════════════════════
   FOOTER / BOTTOM
══════════════════════════════════════════════ */
.stApp > footer, footer,
div[data-testid="stBottom"],
div[data-testid="stBottom"] > * {
  background: linear-gradient(to top, var(--bg) 60%, transparent) !important;
  border: none !important;
}

/* ══════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060c1a 0%, #030711 100%) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] .sidebar-title {
  font-size: 1.1rem; font-weight: 800 !important;
  color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important;
  letter-spacing: 0.5px; display: block; margin-bottom: 20px;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
  color: var(--text2) !important; -webkit-text-fill-color: var(--text2) !important;
  font-size: 0.82rem !important; font-weight: 500 !important;
  text-transform: uppercase !important; letter-spacing: 0.8px !important;
}
/* Slider */
div[data-baseweb="slider"] [role="slider"] {
  background: var(--teal) !important; border-color: var(--teal) !important;
  box-shadow: 0 0 10px rgba(0,229,195,0.5) !important;
}
div[data-baseweb="slider"] div[data-testid="stSliderTrackActive"] {
  background: linear-gradient(90deg, var(--indigo), var(--teal)) !important;
}
/* Sidebar select */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.05) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important;
}
section[data-testid="stSidebar"] textarea {
  background: rgba(255,255,255,0.05) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important; resize: vertical !important;
}
section[data-testid="stSidebar"] textarea:focus {
  border-color: var(--teal) !important; box-shadow: 0 0 0 2px rgba(0,229,195,0.15) !important;
}
section[data-testid="stSidebar"] button {
  background: rgba(244,63,94,0.1) !important; color: var(--red) !important;
  -webkit-text-fill-color: var(--red) !important;
  border: 1px solid rgba(244,63,94,0.3) !important; border-radius: var(--radius2) !important;
  font-weight: 700 !important; transition: all 0.2s !important;
}
section[data-testid="stSidebar"] button:hover {
  background: rgba(244,63,94,0.2) !important; transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════════════
   LAYOUT
══════════════════════════════════════════════ */
.block-container {
  max-width: 1100px !important;
  padding: 2.5rem 2rem 7rem 2rem !important;
  margin: 0 auto !important;
}

/* ══════════════════════════════════════════════
   HEADER
══════════════════════════════════════════════ */
.chat-header { text-align: center; padding: 1rem 0 2rem 0; }
.chat-header .logo-wrap {
  display: flex; align-items: center; justify-content: center;
  gap: 14px; margin-bottom: 10px;
}
.chat-header h1 {
  font-size: 2.4rem !important; font-weight: 900 !important;
  background: linear-gradient(135deg, var(--teal) 0%, #a5b4fc 60%, var(--teal) 100%) !important;
  background-size: 200% auto !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  animation: textShimmer 4s linear infinite !important;
  letter-spacing: 2px; margin: 0 !important; line-height: 1.1;
}
@keyframes textShimmer {
  to { background-position: 200% center; }
}
.chat-header .caption { font-size: 0.85rem; color: var(--text3) !important; -webkit-text-fill-color: var(--text3) !important; }
.chat-header .caption code {
  background: rgba(0,229,195,0.1); color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important;
  padding: 2px 8px; border-radius: 6px; font-size: 0.82rem; border: 1px solid rgba(0,229,195,0.2);
}

/* ══════════════════════════════════════════════
   TITRES
══════════════════════════════════════════════ */
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container h5, .block-container h6,
.block-container div.stMarkdown h1, .block-container div.stMarkdown h2,
.block-container div.stMarkdown h3, .block-container div.stMarkdown h4 {
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
  font-weight: 800 !important; letter-spacing: -0.3px !important;
}

/* ══════════════════════════════════════════════
   TABS
══════════════════════════════════════════════ */
div[data-testid="stTabs"] {
  border-bottom: 1px solid var(--border) !important;
}
div[data-testid="stTabs"] button {
  color: var(--text3) !important; -webkit-text-fill-color: var(--text3) !important;
  font-weight: 600 !important; font-size: 0.88rem !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 10px 16px !important;
  transition: all 0.2s !important;
}
div[data-testid="stTabs"] button:hover {
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  background: rgba(255,255,255,0.04) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
  background: rgba(0,229,195,0.05) !important;
}

/* ══════════════════════════════════════════════
   CHAT BUBBLES
══════════════════════════════════════════════ */
.msg-row { display: flex; align-items: flex-start; gap: 12px; margin: 14px 0; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
  width: 40px; height: 40px; flex-shrink: 0;
  background: linear-gradient(135deg, rgba(0,229,195,0.15), rgba(99,102,241,0.15));
  border: 1.5px solid rgba(0,229,195,0.4); border-radius: 50%;
  box-shadow: 0 0 15px rgba(0,229,195,0.2);
  display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.bubble-assistant {
  display: inline-block;
  background: linear-gradient(135deg, rgba(0,229,195,0.12), rgba(99,102,241,0.08)) !important;
  border: 1px solid rgba(0,229,195,0.2);
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  border-radius: 4px 18px 18px 18px; padding: 14px 18px;
  font-size: 0.95rem; font-weight: 400; line-height: 1.65; max-width: 74%;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.bubble-assistant * { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
.bubble-user {
  display: inline-block;
  background: linear-gradient(135deg, #1e2d4a, #161f30) !important;
  border: 1px solid rgba(99,102,241,0.2);
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
  border-radius: 18px 4px 18px 18px; padding: 14px 18px;
  font-size: 0.95rem; line-height: 1.65; max-width: 65%;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.bubble-user * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
.bubble-welcome {
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  font-size: 0.97rem; line-height: 1.65; padding: 8px 0 0 0;
}
.bubble-welcome strong { color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important; }

/* ══════════════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════════════ */
div[data-testid="stBottom"],
div[data-testid="stChatInputContainer"],
div[data-testid="stChatInputContainer"] > div,
div[data-testid="stChatInputContainer"] > div > div {
  background: transparent !important;
}
div[data-testid="stBottom"] {
  background: linear-gradient(to top, var(--bg) 70%, transparent) !important;
  border: none !important;
  padding: 10px 1rem 20px 1rem !important;
}
div[data-testid="stChatInput"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(0,229,195,0.4) !important;
  border-radius: 50px !important;
  backdrop-filter: blur(20px) !important;
  box-shadow: 0 0 0 1px rgba(0,229,195,0.1), 0 0 30px rgba(0,229,195,0.1) !important;
  transition: box-shadow 0.3s !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(0,229,195,0.15), 0 0 40px rgba(0,229,195,0.2) !important;
}
div[data-testid="stChatInput"] textarea {
  color: #ffffff !important; background: transparent !important;
  -webkit-text-fill-color: #ffffff !important; font-size: 0.95rem !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text3) !important; -webkit-text-fill-color: var(--text3) !important;
}
div[data-testid="stChatInput"] button {
  background: linear-gradient(135deg, var(--teal), var(--indigo)) !important;
  border-radius: 50% !important; border: none !important;
  box-shadow: 0 0 15px rgba(0,229,195,0.4) !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stChatInput"] button:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 0 25px rgba(0,229,195,0.6) !important;
}
div[data-testid="stChatInput"] button * {
  color: #030711 !important; -webkit-text-fill-color: #030711 !important;
}
.block-container { padding-bottom: 130px !important; }

/* ══════════════════════════════════════════════
   CARTES OFFRES
══════════════════════════════════════════════ */
.job-card {
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 20px 24px !important; margin-bottom: 16px !important;
  backdrop-filter: blur(20px) !important;
  box-shadow: var(--shadow) !important;
  transition: transform 0.25s cubic-bezier(.4,0,.2,1), box-shadow 0.25s, border-color 0.25s !important;
}
.job-card:hover {
  transform: translateY(-5px) !important;
  box-shadow: var(--shadow2) !important;
  border-color: rgba(0,229,195,0.25) !important;
}
.job-title {
  font-size: 1.02rem; font-weight: 700;
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}
.job-meta {
  font-size: 0.82rem; color: var(--text2) !important; -webkit-text-fill-color: var(--text2) !important;
  margin: 5px 0 10px 0;
}
.job-snippet {
  font-size: 0.88rem; color: var(--text2) !important; -webkit-text-fill-color: var(--text2) !important;
  line-height: 1.6;
}
.job-link { font-size: 0.82rem; color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important; }
.score-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 5px 14px; border-radius: 50px;
  font-weight: 700; font-size: 0.88rem; margin: 8px 0;
  letter-spacing: 0.3px;
}
.score-high {
  background: linear-gradient(135deg, rgba(0,229,195,0.12), rgba(0,229,195,0.06));
  color: var(--teal) !important; -webkit-text-fill-color: var(--teal) !important;
  border: 1px solid rgba(0,229,195,0.3);
  box-shadow: 0 0 12px rgba(0,229,195,0.1);
}
.score-mid {
  background: linear-gradient(135deg, rgba(251,191,36,0.12), rgba(251,191,36,0.06));
  color: var(--yellow) !important; -webkit-text-fill-color: var(--yellow) !important;
  border: 1px solid rgba(251,191,36,0.3);
}
.score-low {
  background: linear-gradient(135deg, rgba(244,63,94,0.12), rgba(244,63,94,0.06));
  color: var(--red) !important; -webkit-text-fill-color: var(--red) !important;
  border: 1px solid rgba(244,63,94,0.3);
}
.analysis-box {
  background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important; padding: 16px 20px !important; margin-top: 12px !important;
  font-size: 0.9rem; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  line-height: 1.75; backdrop-filter: blur(10px) !important;
}
.analysis-box * { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }

/* ══════════════════════════════════════════════
   BOUTONS
══════════════════════════════════════════════ */
.block-container div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, var(--teal), #00c4a8) !important;
  color: #030711 !important; -webkit-text-fill-color: #030711 !important;
  border: none !important; font-weight: 700 !important;
  border-radius: var(--radius2) !important;
  padding: 10px 24px !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  box-shadow: 0 4px 20px rgba(0,229,195,0.25) !important;
}
.block-container div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(0,229,195,0.4) !important;
}

/* ══════════════════════════════════════════════
   MENUS DÉROULANTS (selectbox)
══════════════════════════════════════════════ */
div[data-baseweb="select"] > div {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important; color: var(--text) !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(0,229,195,0.15) !important;
}
div[data-baseweb="select"] * {
  background-color: transparent !important;
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
  background: var(--bg2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important; overflow: hidden !important;
  box-shadow: var(--shadow) !important;
}
div[data-baseweb="menu"] li {
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
  transition: background 0.15s !important;
}
div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] li[aria-selected="true"] {
  background: rgba(0,229,195,0.08) !important;
}
div[data-baseweb="select"] svg { fill: var(--teal) !important; }

/* ══════════════════════════════════════════════
   INPUT TEXT & FILE UPLOAD
══════════════════════════════════════════════ */
div[data-testid="stTextInput"] input {
  background: var(--bg3) !important; border: 1px solid var(--border) !important;
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
  border-radius: var(--radius2) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(0,229,195,0.15) !important;
}
div[data-testid="stTextInput"] input::placeholder {
  color: var(--text3) !important; -webkit-text-fill-color: var(--text3) !important;
}
div[data-testid="stFileUploader"] {
  background: var(--glass) !important; border: 1px dashed rgba(0,229,195,0.3) !important;
  border-radius: var(--radius) !important;
}
div[data-testid="stFileUploaderDropzone"] {
  background: transparent !important; border: none !important;
}

/* ══════════════════════════════════════════════
   ALERTES / INFO / SUCCESS
══════════════════════════════════════════════ */
div[data-testid="stAlert"] {
  background: rgba(255,255,255,0.04) !important;
  border-radius: var(--radius2) !important;
  backdrop-filter: blur(10px) !important;
}

/* ══════════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════════ */
div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--indigo), var(--teal)) !important;
  border-radius: 50px !important;
}

/* ══════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════ */
div[data-testid="stExpander"] {
  background: var(--glass) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important; overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
  color: var(--text2) !important; -webkit-text-fill-color: var(--text2) !important;
  font-weight: 600 !important;
}

/* ══════════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════════ */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
  margin: 2rem 0 !important;
}

/* ══════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,229,195,0.3); border-radius: 50px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal); }

/* ══════════════════════════════════════════════
   CAPTIONS ET TEXTES SECONDAIRES
══════════════════════════════════════════════ */
[data-testid="stCaption"], [data-testid="stCaption"] *,
.stCaption, .stCaption * {
  color: var(--text3) !important;
  -webkit-text-fill-color: var(--text3) !important;
  font-size: 0.8rem !important;
}
</style>
"""
