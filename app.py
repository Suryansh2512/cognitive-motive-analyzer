"""Gradio MVP for the Cognitive Motive Analyzer."""

import spaces
import gradio as gr

from src.model.inference import analyze


LAMP_OFF = """
<div class="lamp-state lamp-off" role="status" aria-label="Lamp is off">
  <div class="lamp-scene">
    <div class="lamp-post"><div class="lamp-head"><span></span></div></div>
    <div class="lamp-pool"></div>
    <div class="street-line"></div>
  </div>
  <div class="lamp-caption"><span class="status-dot"></span> Waiting for a case</div>
</div>
"""

LAMP_FLICKER = """
<div class="lamp-state lamp-flicker" role="status" aria-label="Lamp is flickering">
  <div class="lamp-scene">
    <div class="lamp-post"><div class="lamp-head"><span></span></div></div>
    <div class="lamp-pool"></div>
    <div class="street-line"></div>
  </div>
  <div class="lamp-caption"><span class="status-dot"></span> Reading the evidence</div>
</div>
"""

LAMP_ON = """
<div class="lamp-state lamp-on" role="status" aria-label="Lamp is on">
  <div class="lamp-scene">
    <div class="lamp-post"><div class="lamp-head"><span></span></div></div>
    <div class="lamp-pool"></div>
    <div class="street-line"></div>
  </div>
  <div class="lamp-caption"><span class="status-dot"></span> Assessment ready</div>
</div>
"""


CSS = """
:root {
  --ink: #f2eee5;
  --muted: #a7a9a8;
  --panel: rgba(24, 29, 31, 0.82);
  --line: rgba(226, 218, 197, 0.16);
  --amber: #f0b45a;
  --amber-bright: #ffe0a2;
  --blue-black: #0b1115;
}

body, .gradio-container {
  background: #0b1115 !important;
  color: var(--ink) !important;
  font-family: Georgia, 'Times New Roman', serif !important;
}

.gradio-container { max-width: 1180px !important; }
#app-shell { padding: 28px 18px 44px; }

.hero { padding: 10px 0 26px; max-width: 760px; }
.kicker { color: var(--amber); font: 600 11px/1.2 Arial, sans-serif; letter-spacing: .18em; text-transform: uppercase; }
h1 { font-size: clamp(38px, 6vw, 74px) !important; line-height: .96 !important; letter-spacing: 0 !important; margin: 14px 0 16px !important; font-weight: 500 !important; }
.hero-copy { color: #c5c7c4; font-size: 18px; line-height: 1.55; max-width: 650px; }

.scene-panel, .input-panel, .output-panel { border: 1px solid var(--line); background: var(--panel); border-radius: 6px; overflow: hidden; }
.scene-panel { min-height: 390px; background: linear-gradient(150deg, #151c20 0%, #0b1115 66%); }

.lamp-state { min-height: 390px; position: relative; overflow: hidden; }
.lamp-scene { height: 340px; position: relative; background: radial-gradient(ellipse at 30% 94%, rgba(240,180,90,.09), transparent 39%), linear-gradient(164deg, transparent 0 68%, rgba(1,4,5,.66) 68% 100%); }
.lamp-scene:before { content: ''; position: absolute; inset: 0; background: linear-gradient(116deg, transparent 0 46%, rgba(42,49,50,.19) 46% 47%, transparent 47% 100%); }
.lamp-post { position: absolute; left: 29%; top: 48px; width: 8px; height: 272px; background: linear-gradient(90deg, #222a2b, #72706a 48%, #171d1e); box-shadow: 0 0 0 1px rgba(0,0,0,.35); }
.lamp-post:after { content: ''; position: absolute; width: 94px; height: 8px; left: 3px; top: 0; transform: rotate(-5deg); transform-origin: left; background: #3b4140; }
.lamp-head { position: absolute; left: 67px; top: -13px; width: 35px; height: 32px; border-radius: 16px 16px 7px 7px; background: #252b2b; border: 2px solid #656762; }
.lamp-head span { position: absolute; width: 12px; height: 12px; left: 9px; top: 8px; border-radius: 50%; background: #101516; }
.lamp-pool { position: absolute; left: 8%; bottom: 22px; width: 49%; height: 72px; transform: skewX(-15deg); border-radius: 50%; background: rgba(240,180,90,.07); filter: blur(12px); }
.street-line { position: absolute; left: -5%; right: 0; bottom: 21px; height: 1px; background: rgba(190,184,166,.16); transform: rotate(-4deg); }
.lamp-caption { position: absolute; left: 24px; bottom: 19px; color: var(--muted); font: 11px Arial, sans-serif; letter-spacing: .12em; text-transform: uppercase; }
.status-dot { display: inline-block; width: 7px; height: 7px; margin-right: 8px; border-radius: 50%; background: #4c5555; vertical-align: 1px; }
.lamp-flicker .lamp-head span { background: var(--amber); box-shadow: 0 0 18px 8px rgba(240,180,90,.58), 0 0 54px 20px rgba(240,180,90,.19); animation: flicker .72s steps(2, end) infinite; }
.lamp-flicker .lamp-pool { background: rgba(240,180,90,.26); animation: pool-flicker .72s steps(2, end) infinite; }
.lamp-flicker .status-dot { background: var(--amber); box-shadow: 0 0 9px var(--amber); }
.lamp-on .lamp-head span { background: var(--amber-bright); box-shadow: 0 0 20px 9px rgba(240,180,90,.72), 0 0 70px 27px rgba(240,180,90,.28); }
.lamp-on .lamp-pool { background: rgba(240,180,90,.36); filter: blur(15px); }
.lamp-on .status-dot { background: var(--amber-bright); box-shadow: 0 0 9px var(--amber); }
@keyframes flicker { 0%, 100% { opacity: 1; } 17% { opacity: .2; } 35% { opacity: .88; } 63% { opacity: .35; } 80% { opacity: .96; } }
@keyframes pool-flicker { 0%, 100% { opacity: 1; } 20% { opacity: .25; } 62% { opacity: .8; } }

.input-panel, .output-panel { padding: 22px; }
.input-panel { margin-top: 18px; }
.output-panel { min-height: 430px; }
.panel-label { color: var(--amber); font: 600 11px Arial, sans-serif; letter-spacing: .16em; text-transform: uppercase; }
textarea, input { background: rgba(9, 13, 15, .78) !important; border: 1px solid var(--line) !important; color: var(--ink) !important; border-radius: 4px !important; }
textarea { font: 17px/1.5 Georgia, serif !important; }
button.primary { background: var(--amber) !important; color: #171513 !important; border: 0 !important; border-radius: 4px !important; font-weight: 700 !important; }
button.primary:hover { background: var(--amber-bright) !important; }
#answer textarea { min-height: 360px !important; font-size: 16px !important; }
.disclaimer { color: #8e9694; font: 12px/1.5 Arial, sans-serif; margin-top: 13px; }
footer { color: #626b6a; font: 11px Arial, sans-serif; letter-spacing: .08em; text-transform: uppercase; padding-top: 24px; }

@media (max-width: 760px) {
  #app-shell { padding: 18px 10px 30px; }
  .scene-panel { min-height: 300px; }
  .lamp-state { min-height: 300px; }
  .lamp-scene { height: 250px; }
  .lamp-post { top: 35px; height: 207px; }
  .output-panel { margin-top: 18px; }
}
"""


def context_from_fields(religion: str, trauma: str, relationships: str, career: str) -> dict:
    return {
        "religion": religion.strip(),
        "trauma": trauma.strip(),
        "relationships": relationships.strip(),
        "career": career.strip(),
    }


def begin_analysis():
    return LAMP_FLICKER, ""


@spaces.GPU(duration=60)
def run_analysis(action: str, religion: str, trauma: str, relationships: str, career: str) -> str:
    if not action or not action.strip():
        raise gr.Error("Describe the behavior before starting the assessment.")
    history = context_from_fields(religion, trauma, relationships, career)
    return analyze(action.strip(), history or None)


def finish_analysis():
    return LAMP_ON


with gr.Blocks() as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <div class="hero">
              <div class="kicker">Cognitive Motive Analyzer / Field Notes</div>
              <h1>What might be<br>happening beneath it?</h1>
              <div class="hero-copy">A reflective tool for comparing possible motives, frameworks, and blind spots in human behavior.</div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes="scene-panel"):
                lamp = gr.HTML(LAMP_OFF)
            with gr.Column(scale=7, elem_classes="input-panel"):
                gr.HTML('<div class="panel-label">Describe the case</div>')
                action = gr.Textbox(
                    placeholder="Someone keeps apologizing, but repeats the same betrayal...",
                    lines=5,
                    show_label=False,
                )
                with gr.Row():
                    religion = gr.Textbox(label="Belief / culture", placeholder="Optional")
                    relationships = gr.Textbox(label="Relationships", placeholder="Optional")
                with gr.Row():
                    trauma = gr.Textbox(label="Past events", placeholder="Optional")
                    career = gr.Textbox(label="Work / money", placeholder="Optional")
                send = gr.Button("Illuminate the case", variant="primary", elem_classes="primary")
                gr.HTML('<div class="disclaimer">Exploratory interpretation only. This tool does not diagnose people or establish facts about their inner motives.</div>')

        with gr.Column(elem_classes="output-panel"):
            gr.HTML('<div class="panel-label">Field assessment</div>')
            answer = gr.Textbox(value="The assessment will appear here.", show_label=False, lines=15, elem_id="answer", interactive=False)

        gr.Examples(
            examples=[
                ["A colleague takes credit for shared work, then privately sends you encouraging messages."],
                ["A parent stops speaking to their adult child after a disagreement about religion."],
                ["Someone gives away meaningful possessions while insisting it is only decluttering."],
            ],
            inputs=[action],
            label="Try a case",
        )
        gr.HTML("<footer>Local inference / Llama 3.1 8B Instruct / uncertainty made visible</footer>")

    request = send.click(begin_analysis, outputs=[lamp, answer])
    request.then(
        run_analysis,
        inputs=[action, religion, trauma, relationships, career],
        outputs=answer,
    ).then(finish_analysis, outputs=lamp)


if __name__ == "__main__":
  demo.launch(css=CSS, server_name="0.0.0.0", server_port=7860)
