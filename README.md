# 🏆 Chess Performance Analytics 
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
  ![Chess.com](https://img.shields.io/badge/Data-Chess.com_API-orange.svg)
  ![Analytics](https://img.shields.io/badge/Analytics-Advanced-purple.svg)

  *Professional chess performance analysis with stunning visualizations*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Examples](#-examples) • [Documentation](#-documentation)

---

## 📋 Overview

A comprehensive chess performance analytics tool that fetches data from Chess.com and generates professional-grade visualizations and insights. Perfect for chess players looking to analyze their game patterns, identify strengths/weaknesses, and track improvement over time.

### 🎯 What it does:
- **Fetches** complete game history from Chess.com API
- **Analyzes** performance across multiple dimensions
- **Generates** comprehensive dashboard (14 Visualizations)
- **Creates** interactive world maps and reports
- **Exports** data for further analysis
---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Libraries
```bash
pip install pandas seaborn matplotlib plotly geopandas numpy chess.com asyncio
```

---

## 💻 Usage

### 1. **Data Fetching**

First, fetch your chess game data using the DataFetcher:

```python
from DataFetcher import DataFetcher
import asyncio

### Analytics Parameters
```python
ChessPerformanceAnalyzer(
    save_file='game_results.json',     # Input data file
    output_folder='output'             # Folder for generated files
)

# Configure your analysis
usernames = ["YourChessUsername"]  # Replace with your Chess.com username (can handle multiple users: ["player1", "player2"])
fetcher = DataFetcher(
    usernames=usernames,          
    verbose=2,                     # Detailed logging (0: quiet, 1: summary, 2: detailed)
    tts='auto',                    # Rate limiting ('auto' or float)
    chunk_size=10,                 # Games per processing batch
    save_file='game_results.json'  # Output file for game data
)

# Run the data fetching
asyncio.run(fetcher.main())
```

### 2. **Generate Analytics**

Then run the performance analysis:

```python
from ChessAnalytics import ChessPerformanceAnalyzer

# Initialize analyzer
analyzer = ChessPerformanceAnalyzer(
    save_file='game_results.json',
    output_folder='chess_analytics_output'
)

# Filter for your username
analyzer.filter_data(['YourChessUsername'])

# Generate all visualizations
analyzer.create_performance_dashboard()
analyzer.create_advanced_analytics()
analyzer.create_interactive_world_map()

# Generate text report
report = analyzer.generate_performance_report()
print(report)
```

### 3. **Output Files**

The analysis generates the following files in your output folder:

```
chess_analytics_output/
├── chess_performance_dashboard.png     # Main 14-panel dashboard
├── chess_advanced_analytics.png        # 6-panel advanced analysis  
├── chess_world_map.html                # Interactive world map
├── chess_world_map.png                 # Static world map
├── performance_report.txt              # Detailed text report
└── chess_performance_data.csv          # Raw analyzed data
```

---

## 📸 Examples

### Dashboard Preview
![Chess Dashboard]()
*Professional 14-panel performance dashboard with modern dark theme*

### Interactive World Map
![World Map]()
*Geographic performance analysis with interactive tooltips*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Made with ❤️ for the chess community*