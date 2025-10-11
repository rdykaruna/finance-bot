from flask import Flask, request, render_template_string
from finbot import load_data, get_tool_call

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FinanceBot Web</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }
        .chatbox { max-width: 700px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        .user, .bot { margin: 10px 0; }
        .user { color: #333; }
        .bot { color: #0077cc; }
        .input-area { margin-top: 20px; }
        input[type=text] { width: 80%; padding: 10px; border-radius: 4px; border: 1px solid #ccc; }
        input[type=submit] { padding: 10px 20px; border-radius: 4px; border: none; background: #0077cc; color: #fff; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chatbox">
        <h2>🤖 FinanceBot Web</h2>
        {% if user_query %}
            <div class="user"><b>You:</b> {{ user_query }}</div>
            <div class="bot"><b>FinanceBot:</b> {{ bot_response|safe }}</div>
        {% else %}
            <div class="bot">How can I help you with your finances today?</div>
        {% endif %}
        <form method="post" class="input-area">
            <input type="text" name="user_query" placeholder="Type your question..." autofocus required>
            <input type="submit" value="Ask">
        </form>
    </div>
</body>
</html>
"""

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
    return response

@app.route("/", methods=["GET", "POST"])
def home():
    user_query = None
    bot_response = None
    data = load_data()
    if request.method == "POST":
        user_query = request.form.get("user_query", "")
        bot_response = chatbot_response(user_query, data)
    return render_template_string(HTML_TEMPLATE, user_query=user_query, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
