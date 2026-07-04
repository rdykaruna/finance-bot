FinanceBot: AI-Powered Personal Finance Assistant

An intelligent, tool-augmented personal finance assistant that analyzes transaction data, tracks budgets, sets savings goals, and visualizes spending habits. The system utilizes Gemini Flash as an orchestrator to dynamically route user queries to specific backend python tools and is equipped with both a CLI interface and a web-based Flask dashboard.

Features

AI Orchestration: Dynamic function routing via the Gemini API based on user intent.

Automated Financial Tracking: Income, spending, net savings, and account balance metrics.

Budgeting & Goals Engine: Set custom thresholds per category and track real-time progression percentages for target milestones.

Smart Savings Planner: Evaluates historical discrete spending over a rolling 6-month window to suggest realistic, non-essential avenues for cutting costs.

Anomaly Detection: Tracks short-term spending irregularities (e.g., specific 2x category spikes) occurring in the last completed calendar month.

Data Visualizations: Generates and embeds dynamically rendered Matplotlib pie charts showing all-time or scoped period breakdown allocations.

Hybrid Interface: Run natively inside a terminal command loop or access an interactive web experience with auto-scrolling AJAX chat bubbles.

Setup & Installation

1. Clone the Repository
git clone https://github.com/rdykaruna/finance-bot.git
cd finance-bot

2. Configure Dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install google-generativeai pandas python-dateutil matplotlib flask

3. API Key Configuration
The orchestrator initialization pulls configurations dynamically. Ensure your API environment variable is accessible by export:
export GEMINI_API_KEY="your_actual_gemini_api_key_here"

Running the Application
Option A: Command-Line Interface (CLI)
To run the lightweight native assistant shell directly in your terminal:
python finbot.py

Type 'exit' inside the execution loop to terminate the assistant safely.

Option B: Flask Web Interface
To launch the responsive browser dashboard complete with asynchronous updates and image render bindings:
python web_finbot.py

Open your browser and navigate to http://127.0.0.1:5000. To clear your active session parameters, navigate to http://127.0.0.1:5000/reset.
