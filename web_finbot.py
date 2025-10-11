from flask import Flask, request, render_template_string, session, redirect, url_for, send_file
from finbot import load_data, get_tool_call
import re
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FinanceBot Web</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #e0e7ff 0%, #f7fafc 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 540px;
            margin: 48px auto;
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 8px 32px rgba(60,60,120,0.15);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 75vh;
        }
        .header {
            background: linear-gradient(90deg, #6366f1 0%, #60a5fa 100%);
            color: #fff;
            padding: 28px 36px 18px 36px;
            display: flex;
            align-items: center;
        }
        .header img {
            width: 44px;
            height: 44px;
            margin-right: 18px;
        }
        .header h2 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .chat-history {
            flex: 1;
            padding: 32px 36px;
            overflow-y: auto;
            background: #f3f4f6;
            display: flex;
            flex-direction: column;
        }
        .bubble {
            max-width: 85%;
            padding: 16px 22px;
            margin-bottom: 18px;
            border-radius: 20px;
            font-size: 1.13em;
            line-height: 1.7;
            word-break: break-word;
            box-shadow: 0 2px 8px rgba(60,60,120,0.07);
        }
        .user-bubble {
            background: #6366f1;
            color: #fff;
            margin-left: auto;
            border-bottom-right-radius: 6px;
        }
        .bot-bubble {
            background: #fff;
            color: #222;
            margin-right: auto;
            border-bottom-left-radius: 6px;
            border: 1px solid #e5e7eb;
        }
        .loading-bubble {
            background: #fff;
            color: #222;
            margin-right: auto;
            border-bottom-left-radius: 6px;
            border: 1px solid #e5e7eb;
            max-width: 85%;
            padding: 16px 22px;
            margin-bottom: 18px;
            border-radius: 20px;
            font-size: 1.13em;
            line-height: 1.7;
            word-break: break-word;
            box-shadow: 0 2px 8px rgba(60,60,120,0.07);
            display: flex;
            align-items: center;
            min-height: 44px;
        }
        .dot-anim {
            display: inline-block;
            width: 8px;
            height: 8px;
            margin: 0 3px;
            border-radius: 50%;
            background: #6366f1;
            opacity: 0.5;
            animation: blink 1.2s infinite;
        }
        .dot-anim:nth-child(2) { animation-delay: 0.2s; }
        .dot-anim:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0.5; }
            40% { opacity: 1; }
        }
        h2, h3, h1 {
            margin: 18px 0 10px 0;
            line-height: 1.3;
            font-weight: 600;
        }
        h2 { font-size: 1.25em; }
        h3 { font-size: 1.1em; }
        h1 { font-size: 1.5em; }
        ul {
            margin: 12px 0 12px 24px;
            padding-left: 18px;
        }
        li {
            margin-bottom: 8px;
            line-height: 1.6;
        }
        hr {
            margin: 24px 0;
            border: none;
            border-top: 1.5px solid #e5e7eb;
        }
        .input-bar {
            background: #f9fafb;
            padding: 22px 36px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            gap: 14px;
        }
        .input-bar input[type=text] {
            flex: 1;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            font-size: 1.08em;
            outline: none;
            transition: border 0.2s;
        }
        .input-bar input[type=text]:focus {
            border: 1.5px solid #6366f1;
        }
        .input-bar input[type=submit] {
            padding: 14px 28px;
            border-radius: 10px;
            border: none;
            background: linear-gradient(90deg, #6366f1 0%, #60a5fa 100%);
            color: #fff;
            font-size: 1.08em;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .input-bar input[type=submit]:hover {
            background: linear-gradient(90deg, #60a5fa 0%, #6366f1 100%);
        }
        @media (max-width: 600px) {
            .container { max-width: 100vw; margin: 0; border-radius: 0; }
            .header, .chat-history, .input-bar { padding: 16px; }
            .chat-history { padding: 20px 8px; }
            .input-bar { padding: 16px 8px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/2721/2721304.png" alt="FinanceBot Logo">
            <h2>FinanceBot</h2>
        </div>
        <div class="chat-history" id="chat-history">
            {% if chat_history %}
                {% for msg in chat_history %}
                    {% if msg.role == 'user' %}
                        <div class="bubble user-bubble"><b>You:</b> {{ msg.text }}</div>
                    {% else %}
                        <div class="bubble bot-bubble">
                            <b>FinanceBot:</b> {{ msg.text|safe }}
                            {% if msg.image %}
                                <div style="margin-top:16px;">
                                    <img src="/{{ msg.image }}" alt="Generated Chart" style="max-width:100%;border-radius:12px;box-shadow:0 2px 8px #ccc;">
                                </div>
                            {% endif %}
                        </div>
                    {% endif %}
                {% endfor %}
            {% else %}
                <div class="bubble bot-bubble"><b>FinanceBot:</b> How can I help you with your finances today?</div>
            {% endif %}
        </div>
        <form id="chat-form" class="input-bar" autocomplete="off">
            <input type="text" name="user_query" id="user_query" placeholder="Type your question..." autofocus required>
            <input type="submit" value="Ask">
        </form>
    </div>
    <script>
        // Auto-scroll chat to bottom on new message
        function scrollChatToBottom() {
            var chat = document.getElementById('chat-history');
            chat.scrollTop = chat.scrollHeight;
        }
        window.onload = scrollChatToBottom;

        // AJAX chat logic
        document.getElementById('chat-form').onsubmit = function(e) {
            e.preventDefault();
            var input = document.getElementById('user_query');
            var text = input.value.trim();
            if (!text) return;
            input.value = '';
            // Add user bubble immediately
            var chat = document.getElementById('chat-history');
            var userBubble = document.createElement('div');
            userBubble.className = 'bubble user-bubble';
            userBubble.innerHTML = '<b>You:</b> ' + text;
            chat.appendChild(userBubble);
            // Add loading bubble
            var loadingBubble = document.createElement('div');
            loadingBubble.className = 'loading-bubble';
            loadingBubble.innerHTML = '<span class="dot-anim"></span><span class="dot-anim"></span><span class="dot-anim"></span>';
            chat.appendChild(loadingBubble);
            scrollChatToBottom();
            // Send AJAX request
            fetch("/", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "user_query=" + encodeURIComponent(text)
            })
            .then(response => response.text())
            .then(html => {
                // Replace chat-history with new content
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, "text/html");
                var newChat = doc.getElementById('chat-history');
                chat.innerHTML = newChat.innerHTML;
                scrollChatToBottom();
            });
        };
    </script>
</body>
</html>
"""

def markdown_to_html(text):
    # Headings
    text = re.sub(r'^\s*### (.*)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*## (.*)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*# (.*)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic (avoid bold inside italic)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?<!\*)', r'<i>\1</i>', text)
    # Horizontal rule
    text = re.sub(r'-{5,}', r'<hr>', text)
    # Lists
    text = re.sub(r'^\s*-\s+(.*\S.*)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Remove empty <li></li> (fixes extra blank lines in lists)
    text = re.sub(r'<li>\s*</li>', '', text)
    # Wrap consecutive <li> in <ul>
    def ul_wrap(match):
        items = match.group(0).replace('\n', '')
        return f'<ul>{items}</ul>'
    text = re.sub(r'(<li>.*?</li>)+', ul_wrap, text, flags=re.DOTALL)
    # Remove <ul></ul> with no <li> inside
    text = re.sub(r'<ul>\s*</ul>', '', text)
    # Remove blank lines directly before/after <ul> blocks
    text = re.sub(r'(<ul>.*?</ul>)\s*<br>', r'\1', text)
    text = re.sub(r'<br>\s*(<ul>)', r'\1', text)
    # Line breaks for paragraphs (avoid inside lists)
    text = re.sub(r'\n{2,}', r'<br>', text)
    # Remove single blank lines between list items
    text = re.sub(r'(<li>.*?</li>)\s*\n\s*(<li>)', r'\1\2', text)
    # Remove blank lines after last list item
    text = re.sub(r'(</ul>)<br>', r'\1', text)
    return text

def extract_image_filename(text):
    # Looks for: saved as *filename.png* or spending_chart.png
    match = re.search(r'\*?(spending_chart\.png)\*?', text)
    if match:
        return match.group(1)
    return None

def chatbot_response(user_query, current_data):
    tool_call = get_tool_call(user_query, current_data)
    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments", {})
    # Import tool functions from finbot
    from finbot import (
        get_summary, get_financial_total, get_top_spending_category,
        find_peak_spending_day_for_category, visualize_spending,
        find_transaction_date, get_financial_advice, add_transaction,
        get_balance, check_budgets, check_goals, contribute_to_goal,
        calculate_savings_plan, add_savings_goal, identify_unnecessary_spending
    )
    tool_belt = {
        "get_summary": get_summary,
        "get_financial_total": get_financial_total,
        "get_top_spending_category": get_top_spending_category,
        "find_peak_spending_day_for_category": find_peak_spending_day_for_category,
        "visualize_spending": visualize_spending,
        "find_transaction_date": find_transaction_date,
        "get_financial_advice": get_financial_advice,
        "add_transaction": add_transaction,
        "get_balance": get_balance,
        "check_budgets": check_budgets,
        "check_goals": check_goals,
        "contribute_to_goal": contribute_to_goal,
        "calculate_savings_plan": calculate_savings_plan,
        "add_savings_goal": add_savings_goal,
        "identify_unnecessary_spending": identify_unnecessary_spending,
    }
    if tool_name in tool_belt:
        try:
            response = tool_belt[tool_name](data=current_data, **arguments)
        except TypeError:
            response = tool_belt[tool_name](current_data)
        except Exception as e:
            response = f"An error occurred while running the tool: {e}"
    else:
        response = "I'm not sure how to do that. Please try rephrasing."
    # Convert markdown-like output to HTML for structured display
    return markdown_to_html(response)

@app.route("/spending_chart.png")
def serve_chart():
    # Serve the chart image from the project directory
    # Fix: Use 'send_file' instead of 'send_from_directory' to avoid matplotlib threading issues
    from flask import send_file
    chart_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "spending_chart.png")
    if os.path.exists(chart_path):
        return send_file(chart_path, mimetype="image/png")
    return "Chart not found", 404

@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []
    data = load_data()
    if request.method == "POST":
        user_query = request.form.get("user_query", "")
        session["chat_history"].append({"role": "user", "text": user_query})
        bot_response = chatbot_response(user_query, data)
        image_filename = extract_image_filename(bot_response)
        session["chat_history"].append({
            "role": "bot",
            "text": bot_response,
            "image": image_filename
        })
        session.modified = True
        return redirect(url_for("home"))
    return render_template_string(
        HTML_TEMPLATE,
        chat_history=session.get("chat_history", [])
    )

@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
