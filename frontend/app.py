import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/dialogue"

EXAMPLES = [
    "I got ghosted after 3 months",
    "My boss took credit for my work",
    "My best friend betrayed my trust after years",
    "My parents don't understand my dreams",
    "I'm starting over after a big failure",
]


def get_dialogue(situation):
    if not situation or not situation.strip():
        yield _error_html("Please describe a situation first.")
        return

    yield '<div style="color:#aaa;font-style:italic;padding:16px 0">🎬 Searching the filmy archives...</div>'

    try:
        response = requests.post(API_URL, json={"situation": situation}, timeout=180)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            yield _error_html(data["error"])
            return

        dialogue = data.get("dialogue", "")
        movie = data.get("movie", "")
        context = data.get("context", "")
        is_fallback = data.get("is_fallback", False)

        if not dialogue:
            yield _error_html("Couldn't find a matching dialogue. Try describing your situation differently.")
            return

        yield _result_html(dialogue, movie, context, is_fallback)

    except requests.exceptions.ConnectionError:
        yield _error_html("Could not connect to the backend. Make sure the API server is running.")
    except requests.exceptions.Timeout:
        yield _error_html("The request timed out. The model might still be loading — please try again.")
    except Exception:
        yield _error_html("Something went wrong. Please try again.")


def _error_html(msg):
    return f'''
    <div style="background:#2a1a1a;border:1px solid #c0392b;border-radius:12px;padding:16px 20px;color:#e74c3c;font-size:14px">
        ⚠️ {msg}
    </div>'''


def _result_html(dialogue, movie, context, is_fallback):
    warning = ''
    if is_fallback:
        warning = '''
        <div style="background:#2a2000;border:1px solid #f39c12;border-radius:8px;padding:10px 16px;margin-bottom:16px;color:#f39c12;font-size:13px">
            ⚠️ <strong>Low confidence match</strong> — No strong dialogue found for your situation. Here's a Bollywood classic instead.
        </div>'''

    context_block = ''
    if context:
        context_block = f'''
        <div style="margin-top:16px;padding-top:14px;border-top:1px solid #333;color:#aaa;font-size:13px;font-style:italic">
            💬 {context}
        </div>'''

    return f'''
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #4a4a8a;border-radius:16px;padding:28px 32px;margin:8px 0;box-shadow:0 4px 24px rgba(0,0,0,0.4)">
        {warning}
        <div style="font-size:22px;font-weight:700;color:#e8e8ff;line-height:1.5;letter-spacing:0.3px">
            &ldquo;{dialogue}&rdquo;
        </div>
        <div style="margin-top:14px;font-size:15px;color:#9b8fcf;font-weight:600;letter-spacing:1px">
            — {movie}
        </div>
        {context_block}
    </div>'''


with gr.Blocks(title="RAG to Riches", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 RAG to Riches\n### *Bollywood has a dialogue for every situation*")

    situation_input = gr.Textbox(
        label="Describe your situation",
        placeholder="e.g. I got ghosted after 3 months...",
        lines=2,
    )

    submit_btn = gr.Button("Find My Dialogue 🎭", variant="primary")

    output = gr.HTML()

    gr.Examples(
        examples=EXAMPLES,
        inputs=[situation_input],
        outputs=[output],
        fn=get_dialogue,
        cache_examples=False,
        label="✨ Quick examples — click to try",
    )

    submit_btn.click(fn=get_dialogue, inputs=situation_input, outputs=output)
    situation_input.submit(fn=get_dialogue, inputs=situation_input, outputs=output)


if __name__ == "__main__":
    demo.launch()
