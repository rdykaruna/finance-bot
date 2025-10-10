import google.generativeai as genai
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import json
import os
import matplotlib.pyplot as plt
import math # Import for ceiling function

# --- 1. Configuration ---
try:
    # NOTE: The provided key is a placeholder and should be replaced with a valid one.
    genai.configure(api_key="AIzaSyB2ROPgddv5SzQtSZXDlt4igvXyS0rR5GU")
except Exception:
    print("🚨 Error: The API key seems to be invalid. Please check it and try again.")
    # exit() 

model = genai.GenerativeModel('gemini-flash-latest')
DATA_FILE = "data.json"
CURRENCY_SYMBOL = "₹"

# --- 2. Persistent Data Storage ---
def load_data():
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print("No data file found. Creating a new default dataset.")
        ref_date = datetime.now()
        default_data = {
            "transactions": [
                {"date": (ref_date - timedelta(days=70)).isoformat(), "description": "July Salary", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=65)).isoformat(), "description": "Myntra Purchase", "amount": -4000.0, "category": "Shopping"},
                {"date": (ref_date - timedelta(days=40)).isoformat(), "description": "August Salary", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=32)).isoformat(), "description": "September Salary", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=30)).isoformat(), "description": "Grocery Mart", "amount": -4500.0, "category": "Groceries"},
                {"date": (ref_date - timedelta(days=28)).isoformat(), "description": "Electricity Bill", "amount": -1200.0, "category": "Bills & Utilities"},
                {"date": (ref_date - timedelta(days=25)).isoformat(), "description": "Flipkart Purchase", "amount": -8500.0, "category": "Shopping"},
                {"date": (ref_date - timedelta(days=15)).isoformat(), "description": "Swiggy Dinner", "amount": -550.0, "category": "Food & Drink"},
                {"date": (ref_date - timedelta(days=2)).isoformat(), "description": "October Salary", "amount": 65000.0, "category": "Income"},
            ],
            "budgets": {"Groceries": 8000.0, "Shopping": 10000.0, "Food & Drink": 5000.0, "Transport": 3000.0, "Bills & Utilities": 15000.0},
            "goals": [{"name": "Rainy Day Fund", "target_amount": 50000.0, "saved_amount": 15000.0}]
        }
        save_data(default_data)
        return default_data
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- 3. Core AI Function (The "Orchestrator") ---
def get_tool_call(user_query, data):
    prompt = f"""
    You are an AI orchestrator. Your job is to analyze the user's request and decide which of the available tools to call.
    You must respond with a single, clean JSON object specifying the 'tool_name' and its 'arguments'.

    *AVAILABLE TOOLS:*
    - get_summary(category: str, time_period: str = None): Get the TOTAL for a specific category.
    - get_top_spending_category(time_period: str = None): Find the category with the highest TOTAL spending.
    - find_peak_spending_day_for_category(category: str, time_period: str = None): Finds the SINGLE BIGGEST transaction/day within a category.
    - visualize_spending(time_period: str = None): Creates a pie chart of spending.
    - find_transaction_date(description: str): Find the date of a specific past transaction.
    - add_transaction(description: str, amount: float, category: str, date: str = None): **FIXED!** Add a new transaction. Automatically negates amount for expense categories.
    - get_financial_advice(): Provide advice on where to save money (based on past overspending).
    - check_budgets(): Get the status of all budgets.
    - get_balance(): Get the current total account balance.
    - contribute_to_goal(name: str, amount: float): Add savings to a goal.
    - check_goals(name: str = None): Check progress on one or all savings goals.
    - calculate_savings_plan(target_amount: float): Calculates a realistic monthly savings amount based on spending history and suggests where to cut truly non-essential expenses.

    *CRITICAL RULES:*
    - If user asks "on which day", "what was my biggest purchase", or "when did I spend the most" for a specific category, you MUST call find_peak_spending_day_for_category.
    - If user asks for a "total" or "how much" for a category, call get_summary.
    - If the user asks a forward-looking question about a *target amount* and *saving*, such as "how much do I need to save" or "I want to buy X, how can I save for it," you MUST call **calculate_savings_plan**.

    *User Query:* "{user_query}"
    *JSON Output:*
    """
    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('`', '').replace('json', '')
        return json.loads(json_string)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Error decoding tool call JSON: {e}")
        return {"tool_name": "unknown", "arguments": {}}

# --- 4. The "Tool Belt" (Backend Functions) ---

def get_df(data):
    if not data.get('transactions'): return pd.DataFrame()
    df = pd.DataFrame(data['transactions'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def parse_time_period(time_period_str):
    today = datetime.now()
    if not time_period_str: return None, None
    time_period_str = time_period_str.lower()
    if 'this month' in time_period_str:
        start_date, end_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0), today
    elif 'last month' in time_period_str:
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date, end_date = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0), last_month_end.replace(hour=23, minute=59, second=59)
    else: return None, None
    return start_date, end_date

# MODIFIED: Tool for adding a transaction (Now with auto-negation for expenses)
def add_transaction(data, description: str, amount: float, category: str, date: str = None):
    if not all([description, amount, category]): return "I need a description, amount, and category."
    
    # Categories that should generally NOT be automatically negated
    GAIN_CATEGORIES = ['Income', 'Savings'] 
    
    # Auto-negation logic for expenses
    final_amount = amount
    # If the category is NOT income/savings AND the amount is positive (i.e., user entered 20000 for a payment)
    if category not in GAIN_CATEGORIES and final_amount > 0:
        final_amount = -amount # Store as negative
    
    try:
        transaction_date = parse_date(date) if date else datetime.now()
    except (ValueError, TypeError):
        transaction_date = datetime.now()
        
    new_trans = { "date": transaction_date.isoformat(), "description": description, "amount": final_amount, "category": category }
    data['transactions'].append(new_trans)
    save_data(data)
    
    # Always display the absolute value to the user for better readability
    return f"Added: '{description}' ({CURRENCY_SYMBOL}{abs(final_amount):,.2f}) to '{category}' for {transaction_date.strftime('%Y-%m-%d')}."

# --- All other tools are also fully defined (unchanged) ---

def calculate_savings_plan(data, target_amount: float):
    """Calculates a realistic monthly saving needed based on available discretionary spending and suggests truly non-essential categories for cutting."""
    
    if target_amount <= 0:
        return "Please specify a positive amount for your target purchase."

    df = get_df(data)
    
    # 1. Analyze last month's spending to find discretionary capacity
    start_date, end_date = parse_time_period('last month')
    
    # Define essential/non-discretionary categories to EXCLUDE from cuts
    non_discretionary = ['Bills & Utilities', 'Income', 'Savings', 'Groceries', 'Transport'] 
    
    if not start_date or df.empty:
        # Fallback if no data
        return (f"To buy an item worth {CURRENCY_SYMBOL}{target_amount:,.2f}, I recommend a target saving of "
                f"*{CURRENCY_SYMBOL}{500:,.2f} per month*. Focus on reducing 'Shopping' or 'Food & Drink'.")

    last_month_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    expenses = last_month_df[last_month_df['amount'] < 0]
    
    # Filter for truly discretionary spending only
    discretionary = expenses[~expenses['category'].isin(non_discretionary)].copy()
    
    if discretionary.empty:
        realistic_monthly_saving = 500.0
        suggestion = "You're already very frugal! To meet this goal, you may need to reduce a non-essential subscription or try to earn a little extra."
    else:
        discretionary_total = abs(discretionary['amount'].sum())
        realistic_monthly_saving = discretionary_total * 0.30 
        
        if realistic_monthly_saving < 200:
             realistic_monthly_saving = 200.0

        discretionary['amount'] = abs(discretionary['amount'])
        category_totals = discretionary.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_categories = category_totals.head(2).index.tolist()
        
        if len(top_categories) == 1:
            suggestion = f"The best area to look for cuts is in '{top_categories[0]}'."
        elif len(top_categories) > 1:
            suggestion = f"The best areas to look for cuts are your highest non-essential expenses: '{top_categories[0]}' and '{top_categories[1]}'."
        else:
             suggestion = "Try reducing spending in 'Shopping' or 'Food & Drink', as these are often flexible."

    # Calculate time to reach the target
    months_needed_raw = target_amount / realistic_monthly_saving
    months_needed_rounded = int(math.ceil(months_needed_raw)) 
    
    return (f"Based on your spending habits, a realistic monthly saving amount is "
            f"*{CURRENCY_SYMBOL}{realistic_monthly_saving:,.2f} per month*. This means you will reach your goal "
            f"of {CURRENCY_SYMBOL}{target_amount:,.2f} in approximately **{months_needed_rounded} months**. "
            f"To achieve this, {suggestion}")


def find_peak_spending_day_for_category(data, category: str, time_period: str = None):
    """Finds the single largest transaction within a specific category."""
    df = get_df(data)
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: 
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    category_expenses = df[(df['category'].str.lower() == category.lower()) & (df['amount'] < 0)]

    if category_expenses.empty:
        return f"No spending found for '{category}' in this period."

    peak_transaction = category_expenses.loc[category_expenses['amount'].idxmin()]
    
    return (f"Your highest spending on '{category}' was for '{peak_transaction['description']}' "
            f"on {peak_transaction['date'].strftime('%Y-%m-%d')} for {CURRENCY_SYMBOL}{abs(peak_transaction['amount']):,.2f}.")


def get_balance(data):
    df = get_df(data)
    total = df['amount'].sum()
    return f"Your current total balance is {CURRENCY_SYMBOL}{total:,.2f}."

def check_budgets(data):
    df = get_df(data)
    this_month_df = df[df['date'].dt.month == datetime.now().month]
    report = "Budget Status (This Month):\n"
    for category, budget_amount in data.get('budgets', {}).items():
        spent = abs(this_month_df[this_month_df['category'] == category]['amount'].sum())
        remaining = budget_amount - spent
        report += f"- {category}: Spent {CURRENCY_SYMBOL}{spent:,.2f} of {CURRENCY_SYMBOL}{budget_amount:,.2f} ({CURRENCY_SYMBOL}{abs(remaining):,.2f} {'left' if remaining >=0 else 'over'}).\n"
    return report.strip()

def check_goals(data, name: str = None):
    goals = data.get('goals', [])
    if not goals: return "You haven't set any savings goals yet."
    if name:
        for goal in goals:
            if goal.get('name', '').lower() == name.lower():
                saved, target = goal.get('saved_amount', 0), goal.get('target_amount', 0)
                progress = (saved / target) * 100 if target > 0 else 0
                return f"For your '{name}' goal, you have saved {CURRENCY_SYMBOL}{saved:,.2f} of {CURRENCY_SYMBOL}{target:,.2f} ({progress:.0f}% complete)."
        return f"Sorry, I couldn't find a goal named '{name}'."
    else:
        report = "Here are your current savings goals:\n"
        for goal in goals:
            g_name, saved, target = goal.get('name'), goal.get('saved_amount', 0), goal.get('target_amount', 0)
            report += f"- {g_name}: {CURRENCY_SYMBOL}{saved:,.2f} / {CURRENCY_SYMBOL}{target:,.2f} saved.\n"
        return report.strip()

def contribute_to_goal(data, name: str, amount: float):
    goals = data.get('goals', [])
    for goal in goals:
        if goal.get('name', '').lower() == name.lower():
            goal['saved_amount'] += amount
            # When contributing to a goal, it's an expense out of the main account. The add_transaction function handles the sign now.
            add_transaction(data, f"Contribution to {name}", amount, "Savings") 
            return f"Great! I've added {CURRENCY_SYMBOL}{amount:,.2f} to your '{name}' goal."
    return f"Sorry, I couldn't find a goal named '{name}'."

def get_summary(data, category: str, time_period: str = None):
    df = get_df(data)
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df_filtered = df[df['category'].str.lower() == category.lower()]
    if df_filtered.empty: return f"No transactions found for '{category}' in this period."
    total = abs(df_filtered['amount'].sum())
    return f"Total for '{category}' is {CURRENCY_SYMBOL}{total:,.2f}."

def get_top_spending_category(data, time_period: str = None):
    df = get_df(data)
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    expenses = df[df['amount'] < 0].copy()
    if expenses.empty: return "No expenses found for this period."
    expenses['amount'] = abs(expenses['amount'])
    category_totals = expenses.groupby('category')['amount'].sum()
    top_category, top_amount = category_totals.idxmax(), category_totals.max()
    return f"Your top spending category was *'{top_category}'*, with a total of {CURRENCY_SYMBOL}{top_amount:,.2f}."

def find_transaction_date(data, description: str):
    df = get_df(data)
    matches = df[df['description'].str.contains(description, case=False)]
    if matches.empty: return f"I couldn't find any transaction matching '{description}'."
    latest = matches.sort_values(by='date', ascending=False).iloc[0]
    return f"The last transaction for '{description}' was on *{latest['date'].strftime('%Y-%m-%d')}* for {CURRENCY_SYMBOL}{abs(latest['amount']):,.2f}."

def get_financial_advice(data):
    df = get_df(data)
    budgets = data.get('budgets', {})
    if not budgets: return "Set some budgets first for me to give advice."
    last_month_str = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%B")
    start_date, end_date = parse_time_period('last month')
    if not start_date: return "Not enough time has passed to analyze last month's data."
    last_month_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if last_month_df.empty: return f"I don't have enough data from {last_month_str} to give advice."
    overspent_categories = [(cat, abs(last_month_df[last_month_df['category'] == cat]['amount'].sum()) - budget) for cat, budget in budgets.items() if abs(last_month_df[last_month_df['category'] == cat]['amount'].sum()) > budget]
    if not overspent_categories:
        expenses = last_month_df[last_month_df['amount'] < 0]
        discretionary = expenses[~expenses['category'].isin(['Bills & Utilities', 'Income', 'Savings'])]
        if discretionary.empty: return "You did a great job staying within budget last month!"
        top_category = discretionary.groupby('category')['amount'].sum().abs().idxmax()
        return f"You stayed within all budgets last month! Your highest discretionary spending was in *'{top_category}'*, which could be an area to watch."
    top_offender = max(overspent_categories, key=lambda item: item[1])
    return f"Based on last month, the best place to save money might be in *'{top_offender[0]}'*. You went over your budget by {CURRENCY_SYMBOL}{top_offender[1]:,.2f}."

def visualize_spending(data, time_period: str = None):
    df = get_df(data)
    period_text = "All Time"
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: 
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            period_text = time_period.title()
    expenses = df[df['amount'] < 0].copy()
    if expenses.empty: return "No spending data found for this period to visualize."
    expenses['amount'] = abs(expenses['amount'])
    category_totals = expenses.groupby('category')['amount'].sum()
    plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(10, 8))
    plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    plt.title(f'Spending Breakdown for {period_text}', fontsize=16)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    file_name = "spending_chart.png"
    plt.savefig(file_name)
    plt.close()
    return f"I've generated a pie chart of your spending and saved it as *{file_name}*."

def generate_proactive_insight(data):
    # This function is included but simplified for the main loop
    return ""

# --- 5. The Main Executor Loop ---
def main():
    data = load_data()
    proactive_insight = generate_proactive_insight(data)
    
    print("🤖 AI Financial Assistant (God Level v1.1)")
    if proactive_insight:
        print(f"   {proactive_insight}")
    print("   My analytical precision has been upgraded. How can I help?")
    print("   Type 'exit' to quit.")

    # MODIFIED: Added the new tool to the tool_belt
    tool_belt = {
        "get_summary": get_summary,
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
    }

    while True:
        current_data = load_data()
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("🤖 Goodbye!")
            break

        tool_call = get_tool_call(user_query, current_data)
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        print(f"🧠 (Debug) AI decided to call tool: '{tool_name}' with arguments: {arguments}")

        if tool_name in tool_belt:
            # Handle the 'data' argument and keyword arguments for the tool call
            try:
                # Most tools accept data and keyword arguments
                response = tool_belt[tool_name](data=current_data, **arguments)
            except TypeError:
                # For tools like get_financial_advice, check_budgets, which only take 'data'
                 response = tool_belt[tool_name](current_data)
            except Exception as e:
                response = f"An error occurred while running the tool: {e}"
        else:
            response = "I'm not sure how to do that. Please try rephrasing."
            
        print(f"🤖 Assistant: {response}")

if __name__ == "__main__":
    main()
