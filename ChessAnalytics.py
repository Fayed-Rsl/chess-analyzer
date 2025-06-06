
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Optional
import plotly.express as px
import re
import warnings
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# Set modern dark theme styling
plt.style.use('dark_background')
sns.set_palette("bright")

class ChessPerformanceAnalyzer:
    """
    A comprehensive chess performance analyzer for Chess.com game data.
    
    This class provides advanced analytics and beautiful visualizations
    for chess game performance analysis including:
    - Win rate analysis by various dimensions
    - Geographic performance mapping
    - Time-based performance trends
    - Rating progression analysis
    - Interactive dashboards
    """
    
    def __init__(self, save_file: str = 'game_results.json', output_folder: str = 'chess_analytics_output'):
        """
        Initialize the Chess Performance Analyzer.
        
        Args:
            save_file (str): Path to the JSON file containing game data
            output_folder (str): Folder to save all generated visualizations
        """
        self.save_file = save_file
        self.output_folder = output_folder
        self.data = None
        self.world_map = None
        self.country_mapping = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.load_data()
        self._setup_styling()

        # Create mapping dictionary for common mismatches
        self.country_mapping = {
            'United States': 'United States',
            'Czech Republic': 'Czech Republic',
            'Ivory Coast': 'Ivory Coast',
            'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
            # 'DR Congo': 'Democratic Republic of the Congo',
            'North Macedonia': 'Macedonia',
            'Scotland': 'United Kingdom',
            'England': 'United Kingdom',
            'Wales': 'United Kingdom',
            'Türkiye': 'Turkey',
            'TÃ¼rkiye': 'Turkey',
            'Hong Kong': 'Hong Kong',
            'Palestine': 'Palestinian Territory',
            'Sao Tome/Principe': 'Sao Tome and Principe',
            'Saint Kitts/Nevis': 'Saint Kitts and Nevis',
            'Saint Pierre/Miquelon': 'Saint Pierre and Miquelon',
            'Trinidad/Tobago': 'Trinidad and Tobago',
            'The Gambia': 'Gambia',
            'Timor-Leste': 'East Timor',
            'US Virgin Islands': 'U.S. Virgin Islands',
            'Vatican City': 'Vatican',
            # Regional identifiers that should be mapped to countries
            'Basque Country': 'Spain',
            'Catalonia': 'Spain',
            'Canary Islands': 'Spain',
            'Galicia': 'Spain',
            # Skip these as they're not countries
            'International': None,
            'European Union': None,
        }


    def _setup_styling(self):
        """Setup modern dark theme styling for all visualizations."""
        # Modern dark theme color palette
        self.colors = {
            'primary': '#00D4FF',      # Bright cyan
            'secondary': '#FF6B9D',    # Pink
            'accent': '#FFD93D',       # Yellow
            'success': '#6BCF7F',      # Green
            'warning': '#FF8C42',      # Orange
            'danger': '#FF5E5B',       # Red
            'neutral': '#95A5A6',      # Gray
            'background': '#1E1E2E',   # Dark blue
            'surface': '#2A2D3A',      # Dark gray
            'text': '#FFFFFF'          # White text
        }
        
        # Set matplotlib dark theme
        plt.rcParams.update({
            'figure.facecolor': '#1E1E2E',
            'axes.facecolor': '#2A2D3A',
            'axes.edgecolor': '#404552',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.bottom': True,
            'axes.spines.left': True,
            'axes.grid': True,
            'grid.color': '#404552',
            'grid.alpha': 0.3,
            'text.color': '#FFFFFF',
            'axes.labelcolor': '#FFFFFF',
            'xtick.color': '#FFFFFF',
            'ytick.color': '#FFFFFF',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10,
            'legend.frameon': False,
            'legend.facecolor': 'none'
        })

    def load_data(self) -> None:
        """
        Load and preprocess chess game data from JSON file.
        """
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            
            # Flatten nested data structure
            all_games = [game for games in data.values() for game in games]
            self.data = pd.DataFrame(all_games)
            
            # Clean and preprocess data
            self._preprocess_data()
            
            print(f"✅ Loaded {len(self.data)} chess games successfully")
            if 'date' in self.data.columns:
                print(f"📊 Data spans from {self.data['date'].min()} to {self.data['date'].max()}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Could not find data file: {self.save_file}")
        except Exception as e:
            raise Exception(f"❌ Error loading data: {str(e)}")

    def _preprocess_data(self) -> None:
        """
        Clean and preprocess the loaded data.
        """
        # Extract date from archive_url
        if 'archive_url' in self.data.columns:
            # Extract year/month from URLs like '/player/libetue/games/2021/12'
            self.data['date'] = self.data['archive_url'].apply(self._extract_date_from_url)
        
        # Map result values to wins/losses/draws
        self.data['result_category'] = self.data['result'].map(self._categorize_result)
        
        # Calculate derived metrics
        self.data['rating_difference'] = self.data['player_rating'] - self.data['opponent_rating']
        
        # Map results to numeric values for analysis
        self.data['result_numeric'] = self.data['result_category'].map({
            'win': 1, 'loss': 0, 'draw': 0.5
        })
        
        # Clean country names and apply mapping
        self.data['opponent_country'] = self.data['opponent_country'].str.strip()
        self.data['opponent_country_mapped'] = self.data['opponent_country'].map(
            lambda x: self.country_mapping.get(x, x)
        )
        
        # Remove entries that couldn't be mapped (like 'International', 'European Union')
        self.data = self.data[self.data['opponent_country_mapped'].notna()]

    def _extract_date_from_url(self, url: str) -> Optional[pd.Timestamp]:
        """Extract date from archive URL format."""
        try:
            # Pattern: /player/username/games/YYYY/MM
            match = re.search(r'/games/(\d{4})/(\d{1,2})', str(url))
            if match:
                year, month = match.groups()
                return pd.Timestamp(f"{year}-{month}-01")
        except:
            pass
        return None

    def _categorize_result(self, result: str) -> str:
        """
        Categorize chess.com results into win/loss/draw.
        
        Args:
            result: Original result from chess.com
            
        Returns:
            Categorized result: 'win', 'loss', or 'draw'
        """
        # Define result categories based on chess.com result types
        win_results = ['win']
        loss_results = ['timeout', 'checkmated', 'resigned', 'abandoned']
        draw_results = ['agreed', 'stalemate', 'insufficient', 'timevsinsufficient', 
                       'repetition', '50move']
        # Special game modes that count as wins if you achieved the objective
        special_wins = ['kingofthehill', 'threecheck']
        
        if result in win_results or result in special_wins:
            return 'win'
        elif result in loss_results:
            return 'loss'
        elif result in draw_results:
            return 'draw'
        else:
            # Default unknown results to loss (conservative approach)
            return 'loss'

    def filter_data(self, usernames: List[str]) -> None:
        """
        Filter data for specific players.
        
        Args:
            usernames (List[str]): List of player usernames to analyze
        """
        original_count = len(self.data)
        self.data = self.data[self.data['player_username'].isin(usernames)]
        filtered_count = len(self.data)
        
        print(f"🔍 Filtered data: {original_count} → {filtered_count} games")
        print(f"👤 Analyzing players: {', '.join(usernames)}")

    def create_performance_dashboard(self) -> None:
        """
        Create a modern, professional performance dashboard with multiple visualizations.
        """
        # Create a large figure with custom layout
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#1E1E2E')
        
        # Create custom grid layout
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3, 
                     left=0.06, right=0.96, top=0.92, bottom=0.08)
        
        # Create subplots with modern styling
        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])  # Performance Summary
        ax2 = fig.add_subplot(gs[0, 1])  # Win Rate by Time Control
        ax3 = fig.add_subplot(gs[0, 2])  # Win Rate by Color
        ax4 = fig.add_subplot(gs[0, 3])  # Key Performance Metrics
        
        # Row 2
        ax5 = fig.add_subplot(gs[1, :2])  # Rating Difference Analysis (wider)
        ax6 = fig.add_subplot(gs[1, 2])  # Top Countries Performance
        ax7 = fig.add_subplot(gs[1, 3])  # Game Frequency Analysis
        
        # Row 3
        ax8 = fig.add_subplot(gs[2, :2])  # Monthly Performance Trend (wider)
        ax9 = fig.add_subplot(gs[2, 2])  # Rating Distribution
        ax10 = fig.add_subplot(gs[2, 3])  # Opponent Strength Analysis
        
        # Row 4
        ax11 = fig.add_subplot(gs[3, 0])  # Win Streak Analysis
        ax12 = fig.add_subplot(gs[3, 1])  # Time Control Matrix
        ax13 = fig.add_subplot(gs[3, 2])  # Result Breakdown
        ax14 = fig.add_subplot(gs[3, 3])  # Performance Insights
        
        # Apply modern styling to all axes
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]
        for ax in axes:
            self._style_axis(ax)
        
        # Plot all visualizations
        self._plot_performance_summary(ax1)
        self._plot_win_rate_by_time_control(ax2)
        self._plot_win_rate_by_color(ax3)
        self._plot_key_metrics(ax4)
        self._plot_rating_difference_analysis(ax5)
        self._plot_top_countries_performance(ax6)
        self._plot_game_frequency(ax7)
        
        if 'date' in self.data.columns and self.data['date'].notna().any():
            self._plot_monthly_trends(ax8)
        else:
            self._plot_rating_progression(ax8)
            
        self._plot_rating_distributions(ax9)
        self._plot_opponent_strength_analysis(ax10)
        self._plot_win_streak_analysis(ax11)
        self._plot_time_control_matrix(ax12)
        self._plot_rating_performance_heatmap(ax13)
        self._plot_performance_insights(ax14)
        
        # Add main title with modern styling
        fig.suptitle('🏆 CHESS PERFORMANCE ANALYTICS DASHBOARD', 
                    fontsize=28, fontweight='bold', color='white') # y=0.96
        
        # Save the dashboard
        dashboard_path = os.path.join(self.output_folder, 'chess_performance_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', 
                   facecolor='#1E1E2E', edgecolor='none')
        print(f"📊 Dashboard saved to: {dashboard_path}")
        
        plt.show()

    def _style_axis(self, ax):
        """Apply modern dark theme styling to an axis."""
        ax.set_facecolor('#2A2D3A')
        for spine in ax.spines.values():
            spine.set_color('#404552')
            spine.set_linewidth(1.2)
        ax.tick_params(colors='white', which='both')
        ax.grid(True, color='#404552', alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)

    def _plot_performance_summary(self, ax) -> None:
        """Plot modern donut chart performance summary."""
        total_games = len(self.data)
        wins = len(self.data[self.data['result_category'] == 'win'])
        losses = len(self.data[self.data['result_category'] == 'loss'])
        draws = len(self.data[self.data['result_category'] == 'draw'])
        
        win_rate = wins / total_games if total_games > 0 else 0
        
        # Create modern donut chart
        sizes = [wins, losses, draws]
        labels = ['Wins', 'Losses', 'Draws']
        colors = [self.colors['success'], self.colors['danger'], self.colors['neutral']]
        
        # Only include non-zero categories
        non_zero_data = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero_data:
            sizes, labels, colors = zip(*non_zero_data)
            
            # Create donut with modern styling
            wedges, texts = ax.pie(sizes, labels=None, colors=colors, startangle=90, 
                                  wedgeprops=dict(width=0.4, edgecolor='#1E1E2E', linewidth=2))
            
            # Add center circle with gradient effect
            centre_circle = plt.Circle((0,0), 0.6, fc='#1E1E2E', ec='#404552', linewidth=2)
            ax.add_patch(centre_circle)
            
            # Add win rate in center with modern typography
            ax.text(0, 0.1, f'{win_rate:.1%}', ha='center', va='center', 
                    fontsize=24, fontweight='bold', color=self.colors['primary'])
            ax.text(0, -0.2, 'WIN RATE', ha='center', va='center', 
                    fontsize=10, color='white', alpha=0.8)
        
        # Add legend with custom styling
        if non_zero_data:
            legend_labels = [f'{l}: {s}' for s, l in zip(sizes, labels)]
            ax.legend(wedges, legend_labels, loc='center', bbox_to_anchor=(0, -0.8),
                     frameon=False, fontsize=9)
        
        ax.set_title('📊 PERFORMANCE OVERVIEW', fontweight='bold', pad=20, 
                    color='white', fontsize=12)

    def _plot_key_metrics(self, ax) -> None:
        """Plot key performance metrics in a card-style layout."""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Calculate key metrics
        total_games = len(self.data)
        avg_rating = self.data['player_rating'].mean()
        rating_change = self.data['player_rating'].iloc[-1] - self.data['player_rating'].iloc[0] if len(self.data) > 1 else 0
        best_streak = self._calculate_best_streak()
        
        # Create metric cards
        metrics = [
            {'label': 'TOTAL GAMES', 'value': f'{total_games:,}', 'color': self.colors['primary']},
            {'label': 'AVG RATING', 'value': f'{avg_rating:.0f}', 'color': self.colors['accent']},
            {'label': 'RATING CHANGE', 'value': f'{rating_change:+.0f}', 'color': self.colors['success'] if rating_change >= 0 else self.colors['danger']},
            {'label': 'BEST STREAK', 'value': f'{best_streak}', 'color': self.colors['warning']}
        ]
        
        y_positions = [0.8, 0.6, 0.4, 0.2]
        for i, metric in enumerate(metrics):
            # Add colored indicator
            ax.add_patch(Rectangle((0.02, y_positions[i] - 0.05), 0.04, 0.1, 
                                 facecolor=metric['color'], alpha=0.8))
            
            # Add metric value
            ax.text(0.1, y_positions[i] + 0.02, metric['value'], 
                   fontsize=14, fontweight='bold', color='white', va='center')
            
            # Add metric label
            ax.text(0.1, y_positions[i] - 0.03, metric['label'], 
                   fontsize=8, color='white', alpha=0.7, va='center')
        
        ax.set_title('📈 KEY METRICS', fontweight='bold', color='white',
                    loc='left',
                    fontsize=12, pad=10)

    def _calculate_best_streak(self) -> int:
        """Calculate the best win streak."""
        if 'date' in self.data.columns:
            data_sorted = self.data.sort_values('date')
        else:
            data_sorted = self.data
            
        wins = (data_sorted['result_category'] == 'win').astype(int).values
        
        max_streak = current_streak = 0
        for win in wins:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak

    def _plot_win_rate_by_time_control(self, ax) -> None:
        """Plot modern win rate analysis by time control."""
        time_stats = self.data.groupby('time_class').agg({
            'result_numeric': ['mean', 'count']
        }).round(3)
        time_stats.columns = ['win_rate', 'games']
        time_stats = time_stats.reset_index()
        
        # Create modern gradient bars
        bars = ax.bar(time_stats['time_class'], time_stats['win_rate'], 
                     color=self.colors['primary'], alpha=0.8, 
                     edgecolor='#1E1E2E', linewidth=1.5)
        
        # Add gradient effect to bars
        for bar, rate in zip(bars, time_stats['win_rate']):
            # Add glow effect
            ax.bar(bar.get_x(), rate, width=bar.get_width(), 
                  color=self.colors['primary'], alpha=0.3, 
                  edgecolor=self.colors['primary'], linewidth=2)
            
            # Add value label with modern styling
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_title('⏱️ TIME CONTROL PERFORMANCE', fontweight='bold', 
                    color='white', fontsize=12, pad=15)
        ax.set_ylabel('Win Rate', color='white', fontsize=10)
        ax.set_ylim(0, max(time_stats['win_rate']) * 1.3 if len(time_stats) > 0 else 1)
        
        # Style tick labels
        ax.tick_params(axis='x', rotation=45, labelsize=9)

    def _plot_win_rate_by_color(self, ax) -> None:
        """Plot modern win rate by player color."""
        color_stats = self.data.groupby('player_color').agg({
            'result_numeric': ['mean', 'count']
        }).round(3)
        color_stats.columns = ['win_rate', 'games']
        color_stats = color_stats.reset_index()
        
        # Modern color mapping
        colors_map = {
            'white': '#E8E8E8', 
            'black': '#2C2C2C'
        }
        bar_colors = [colors_map.get(color, self.colors['neutral']) for color in color_stats['player_color']]
        
        bars = ax.bar(color_stats['player_color'], color_stats['win_rate'], 
                     color=bar_colors, alpha=0.9, 
                     edgecolor=self.colors['primary'], linewidth=2)
        
        # Add value labels with contrasting colors
        for bar, rate, games, piece_color in zip(bars, color_stats['win_rate'], 
                                                color_stats['games'], color_stats['player_color']):
            height = bar.get_height()
            text_color = 'black' if piece_color == 'white' else 'white'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}\n({games} games)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_title('♟️ COLOR PERFORMANCE', fontweight='bold', 
                    color='white', fontsize=12, pad=25)
        ax.set_ylabel('Win Rate', color='white', fontsize=10)

    def _plot_rating_difference_analysis(self, ax) -> None:
        """Plot modern rating difference vs performance analysis."""
        # Create bins for rating differences
        self.data['rating_diff_bin'] = pd.cut(self.data['rating_difference'], 
                                            bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
                                            labels=['<-200', '-200 to -100', '-100 to -50', '-50 to 0', 
                                                   '0 to 50', '50 to 100', '100 to 200', '>200'])
        
        rating_perf = self.data.groupby('rating_diff_bin').agg({
            'result_numeric': ['mean', 'count']
        })
        rating_perf.columns = ['win_rate', 'count']
        rating_perf = rating_perf.reset_index()
        
        # Create modern area plot with gradient
        x_pos = range(len(rating_perf))
        y_values = rating_perf['win_rate']
        
        # Fill area under curve with gradient effect
        ax.fill_between(x_pos, 0, y_values, alpha=0.3, color=self.colors['primary'])
        ax.fill_between(x_pos, 0, y_values, alpha=0.6, color=self.colors['primary'])
        
        # Main line plot with enhanced styling
        ax.plot(x_pos, y_values, marker='o', linewidth=3, markersize=8, 
               color=self.colors['accent'], markerfacecolor=self.colors['accent'],
               markeredgecolor='white', markeredgewidth=2)
        
        # Add value labels
        for x, y, count in zip(x_pos, y_values, rating_perf['count']):
            ax.text(x, y + 0.03, f'{y:.1%}\n({count})', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='white')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rating_perf['rating_diff_bin'], rotation=45, ha='right')
        ax.set_title('📈 PERFORMANCE VS RATING DIFFERENCE', fontweight='bold', 
                    color='white', fontsize=12, pad=15)
        ax.set_ylabel('Win Rate', color='white', fontsize=10)
        ax.set_ylim(0, max(y_values) * 1.2 if len(y_values) > 0 else 1)

    def _plot_performance_insights(self, ax) -> None:
        """Plot performance insights instead of correlation matrix."""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Calculate insights
        total_games = len(self.data)
        win_rate = len(self.data[self.data['result_category'] == 'win']) / total_games
        
        # Rating advantage analysis
        avg_rating_diff = self.data['rating_difference'].mean()
        favorable_games = len(self.data[self.data['rating_difference'] > 0])
        favorable_rate = favorable_games / total_games
        
        # Best time control
        time_performance = self.data.groupby('time_class')['result_numeric'].mean()
        best_time_control = time_performance.idxmax() if len(time_performance) > 0 else 'N/A'
        best_performance = time_performance.max() if len(time_performance) > 0 else 0
        
        # Color preference
        color_performance = self.data.groupby('player_color')['result_numeric'].mean()
        white_performance = color_performance.get('white', 0)
        black_performance = color_performance.get('black', 0)
        
        # Create insight cards
        insights = [
            f"🎯 Playing against {favorable_rate:.0%} higher-rated opponents",
            f"⭐ Best format: {best_time_control} ({best_performance:.1%} win rate)",
            f"♟️ Color bias: {'White' if white_performance > black_performance else 'Black'} (+{abs(white_performance - black_performance):.1%})",
            f"📊 Rating trend: {'+' if avg_rating_diff > 0 else ''}{avg_rating_diff:.0f} avg difference"
        ]
        
        # Display insights with modern styling
        y_positions = [0.8, 0.6, 0.4, 0.2]
        for i, insight in enumerate(insights):
            # Add colored bullet point
            ax.plot(0.05, y_positions[i], 'o', markersize=8, 
                   color=self.colors['accent'], alpha=0.8)
            
            # Add insight text
            ax.text(0.12, y_positions[i], insight, fontsize=10, 
                   color='white', va='center', ha='left')
        
        ax.set_title('💡 PERFORMANCE INSIGHTS', fontweight='bold', 
                    color='white', fontsize=12, pad=10)

    def _plot_top_countries_performance(self, ax) -> None:
        """Plot performance against top countries with modern styling."""
        country_stats = self.data.groupby('opponent_country').agg({
            'result_numeric': ['mean', 'count']
        })
        country_stats.columns = ['win_rate', 'games']
        country_stats = country_stats[country_stats['games'] >= 3].sort_values('games', ascending=True).tail(8)
        
        if len(country_stats) > 0:
            # Create horizontal bars with gradient effect
            bars = ax.barh(range(len(country_stats)), country_stats['win_rate'], 
                          color=self.colors['secondary'], alpha=0.8,
                          edgecolor=self.colors['secondary'], linewidth=1.5)
            
            # Add glow effect
            ax.barh(range(len(country_stats)), country_stats['win_rate'], 
                   color=self.colors['secondary'], alpha=0.3)
            
            # Add value labels
            for i, (_, row) in enumerate(country_stats.iterrows()):
                ax.text(row['win_rate'] + 0.02, i, f"{row['win_rate']:.1%} ({int(row['games'])})", 
                       va='center', fontsize=9, color='white', fontweight='bold')
            
            ax.set_yticks(range(len(country_stats)))
            ax.set_yticklabels(country_stats.index, fontsize=9)
            ax.set_xlabel('Win Rate', color='white', fontsize=10)
            ax.set_xlim(0, 1.2)
        
        ax.set_title('🌍 TOP COUNTRIES', fontweight='bold', 
                    color='white', fontsize=12, pad=15)

    def _plot_game_frequency(self, ax) -> None:
        """Plot game frequency with modern styling."""
        if 'date' in self.data.columns and self.data['date'].notna().any():
            self.data['day_of_week'] = self.data['date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.data['day_of_week'].value_counts().reindex(day_order, fill_value=0)
            
            # Create modern bar chart
            bars = ax.bar(range(len(day_counts)), day_counts.values, 
                         color=self.colors['accent'], alpha=0.8,
                         edgecolor=self.colors['accent'], linewidth=1.5)
            
            # Add glow effect
            ax.bar(range(len(day_counts)), day_counts.values, 
                  color=self.colors['accent'], alpha=0.3)
            
            ax.set_xticks(range(len(day_counts)))
            ax.set_xticklabels([day[:3] for day in day_order], fontsize=9)
            ax.set_title('📅 GAMES BY DAY', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
        else:
            # Fallback: show games per opponent
            top_opponents = self.data['opponent_username'].value_counts().head(6)
            if len(top_opponents) > 0:
                bars = ax.bar(range(len(top_opponents)), top_opponents.values, 
                             color=self.colors['warning'], alpha=0.8,
                             edgecolor=self.colors['warning'], linewidth=1.5)
                
                ax.set_xticks(range(len(top_opponents)))
                ax.set_xticklabels(top_opponents.index, rotation=45, ha='right', fontsize=8)
            ax.set_title('🎯 FREQUENT OPPONENTS', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
        
        ax.set_ylabel('Games', color='white', fontsize=10)

    def _plot_monthly_trends(self, ax) -> None:
        """Plot monthly performance trends with modern styling."""
        if 'date' in self.data.columns and self.data['date'].notna().any():
            monthly_data = self.data.groupby(self.data['date'].dt.to_period('M')).agg({
                'result_numeric': 'mean',
                'player_rating': 'mean'
            }).reset_index()
            
            if len(monthly_data) > 0:
                x_pos = range(len(monthly_data))
                
                # Performance line
                ax.plot(x_pos, monthly_data['result_numeric'], 
                       marker='o', linewidth=3, markersize=6, 
                       color=self.colors['success'], alpha=0.9,
                       markerfacecolor=self.colors['success'],
                       markeredgecolor='white', markeredgewidth=1.5,
                       label='Win Rate')
                
                # Fill area under curve
                ax.fill_between(x_pos, 0, monthly_data['result_numeric'], 
                               alpha=0.2, color=self.colors['success'])
                
                # Add trend line
                if len(monthly_data) > 2:
                    z = np.polyfit(x_pos, monthly_data['result_numeric'], 1)
                    p = np.poly1d(z)
                    ax.plot(x_pos, p(x_pos), "--", color=self.colors['accent'], 
                           alpha=0.8, linewidth=2, label='Trend')
                
                ax.set_xticks(range(0, len(monthly_data), max(1, len(monthly_data)//6)))
                ax.set_xticklabels([str(monthly_data['date'].iloc[i]) for i in range(0, len(monthly_data), max(1, len(monthly_data)//6))],
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Win Rate', color='white', fontsize=10)
                ax.legend(loc='upper left', fontsize=8)
            
            ax.set_title('📊 MONTHLY TRENDS', fontweight='bold', 
                        color='white', fontsize=12, pad=15)

    def _plot_rating_progression(self, ax) -> None:
        """Plot rating progression with modern styling."""
        # Sort by game sequence and calculate moving average
        self.data['game_number'] = range(len(self.data))
        self.data['rating_ma'] = self.data['player_rating'].rolling(window=min(10, len(self.data)), min_periods=1).mean()
        
        x_pos = self.data['game_number']
        
        # Rating scatter plot
        ax.scatter(x_pos, self.data['player_rating'], 
                  alpha=0.4, color=self.colors['primary'], s=20, label='Game Rating')
        
        # Moving average line
        ax.plot(x_pos, self.data['rating_ma'], 
               linewidth=3, color=self.colors['accent'], 
               label=f'{min(10, len(self.data))}-Game Average')
        
        # Fill area
        ax.fill_between(x_pos, self.data['rating_ma'], 
                       alpha=0.2, color=self.colors['accent'])
        
        ax.set_title('📈 RATING PROGRESSION', fontweight='bold', 
                    color='white', fontsize=12, pad=15)
        ax.set_xlabel('Game Number', color='white', fontsize=10)
        ax.set_ylabel('Rating', color='white', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)

    def _plot_mini_rating_progression(self, ax):
        """Mini rating progression chart as fallback."""
        if len(self.data) > 1:
            x_pos = range(len(self.data))
            ratings = self.data['player_rating'].values
            
            # Simple line plot
            ax.plot(x_pos, ratings, color=self.colors['primary'], linewidth=2, alpha=0.8)
            
            # Add trend line
            if len(self.data) > 5:
                z = np.polyfit(x_pos, ratings, 1)
                p = np.poly1d(z)
                ax.plot(x_pos, p(x_pos), "--", color=self.colors['accent'], 
                    alpha=0.8, linewidth=2)
            
            # Fill area
            ax.fill_between(x_pos, min(ratings), ratings, alpha=0.2, color=self.colors['primary'])
            
            ax.set_title('📈 RATING PROGRESSION', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            ax.set_xlabel('Game Number', color='white', fontsize=10)
            ax.set_ylabel('Rating', color='white', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center', 
                transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title('📈 RATING PROGRESSION', fontweight='bold', 
                        color='white', fontsize=12, pad=15)


    def _plot_rating_distributions(self, ax) -> None:
        """Plot modern rating distributions."""
        # Create overlapping histograms with transparency
        bins = 20
        
        ax.hist(self.data['player_rating'], bins=bins, alpha=0.7, 
               label='Your Rating', color=self.colors['primary'],
               edgecolor='white', linewidth=1)
        ax.hist(self.data['opponent_rating'], bins=bins, alpha=0.7, 
               label='Opponent Rating', color=self.colors['secondary'],
               edgecolor='white', linewidth=1)
        
        # Add vertical lines for averages
        avg_player = self.data['player_rating'].mean()
        avg_opponent = self.data['opponent_rating'].mean()
        
        ax.axvline(avg_player, color=self.colors['primary'], linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'Your Avg: {avg_player:.0f}')
        ax.axvline(avg_opponent, color=self.colors['secondary'], linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'Opp Avg: {avg_opponent:.0f}')
        
        ax.set_title('📊 RATING DISTRIBUTION', fontweight='bold', 
                    color='white', fontsize=12, pad=15)
        ax.set_xlabel('Rating', color='white', fontsize=10)
        ax.set_ylabel('Games', color='white', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

    def _plot_opponent_strength_analysis(self, ax) -> None:
        """Analyze performance against different opponent strengths."""
        self.data['opponent_strength'] = pd.cut(self.data['opponent_rating'], 
                                              bins=[0, 1000, 1200, 1400, 1600, 1800, 2000, 3000],
                                              labels=['<1000', '1000-1200', '1200-1400', '1400-1600', 
                                                     '1600-1800', '1800-2000', '2000+'])
        
        strength_perf = self.data.groupby('opponent_strength').agg({
            'result_numeric': ['mean', 'count']
        })
        strength_perf.columns = ['win_rate', 'games']
        strength_perf = strength_perf.reset_index()
        
        if len(strength_perf) > 0:
            # Create gradient bars
            bars = ax.bar(range(len(strength_perf)), strength_perf['win_rate'], 
                         color=self.colors['warning'], alpha=0.8,
                         edgecolor=self.colors['warning'], linewidth=1.5)
            
            # Add glow effect
            ax.bar(range(len(strength_perf)), strength_perf['win_rate'], 
                  color=self.colors['warning'], alpha=0.3)
            
            # Add value labels
            for i, (_, row) in enumerate(strength_perf.iterrows()):
                if row['games'] > 0:
                    ax.text(i, row['win_rate'] + 0.02, f"{row['win_rate']:.1%}\n({int(row['games'])})", 
                           ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
            
            ax.set_xticks(range(len(strength_perf)))
            ax.set_xticklabels(strength_perf['opponent_strength'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Win Rate', color='white', fontsize=10)
        
        ax.set_title('💪 VS OPPONENT STRENGTH', fontweight='bold', 
                    color='white', fontsize=12, pad=15)

    def _plot_win_streak_analysis(self, ax) -> None:
        """Analyze win streaks with modern styling."""
        # Calculate consecutive wins
        data_sorted = self.data.sort_values('date') if 'date' in self.data.columns else self.data
        data_sorted['is_win'] = (data_sorted['result_category'] == 'win').astype(int)
        wins = data_sorted['is_win'].values
        
        # Find streaks
        streaks = []
        current_streak = 0
        for win in wins:
            if win:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        if streaks:
            # Create modern histogram
            bins = max(1, len(set(streaks)))
            n, bins_edges, patches = ax.hist(streaks, bins=bins, 
                                           color=self.colors['success'], alpha=0.8,
                                           edgecolor='white', linewidth=1.5)
            
            # Add glow effect
            ax.hist(streaks, bins=bins, color=self.colors['success'], alpha=0.3)
            
            max_streak = max(streaks)
            ax.set_title(f'🔥 WIN STREAKS (Max: {max_streak})', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            ax.set_xlabel('Streak Length', color='white', fontsize=10)
            ax.set_ylabel('Frequency', color='white', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Win Streaks\nFound', ha='center', va='center', 
                   transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title('🔥 WIN STREAKS', fontweight='bold', 
                        color='white', fontsize=12, pad=15)

    def _plot_time_control_matrix(self, ax) -> None:
        """Create a modern matrix showing performance across time controls and colors."""
        if len(self.data['time_class'].unique()) > 1 and len(self.data['player_color'].unique()) > 1:
            matrix_data = self.data.pivot_table(values='result_numeric', 
                                              index='time_class', 
                                              columns='player_color', 
                                              aggfunc='mean')
            
            # Create modern heatmap
            im = ax.imshow(matrix_data.values, cmap='plasma', aspect='auto', vmin=0, vmax=1)
            
            ax.set_xticks(range(len(matrix_data.columns)))
            ax.set_xticklabels(matrix_data.columns, fontsize=10)
            ax.set_yticks(range(len(matrix_data.index)))
            ax.set_yticklabels(matrix_data.index, fontsize=9)
            
            # Add text annotations with modern styling
            for i in range(len(matrix_data.index)):
                for j in range(len(matrix_data.columns)):
                    value = matrix_data.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.1%}', ha='center', va='center', 
                               color='white', fontweight='bold', fontsize=10)
            
            ax.set_title('🎯 TIME × COLOR MATRIX', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\nfor Matrix Analysis', 
                   ha='center', va='center', transform=ax.transAxes, 
                   color='white', fontsize=12)
            ax.set_title('🎯 TIME × COLOR MATRIX', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            
    # Option 1: Rating Performance Heatmap
    def _plot_rating_performance_heatmap(self, ax) -> None:
        """Create a heatmap showing performance across different rating ranges and time controls."""
        # Create rating bins
        self.data['rating_bin'] = pd.cut(self.data['player_rating'], 
                                    bins=[0, 1200, 1400, 1600, 1800, 2000, 3000],
                                    labels=['<1200', '1200-1400', '1400-1600', 
                                            '1600-1800', '1800-2000', '2000+'])
        
        # Create pivot table
        if len(self.data['time_class'].unique()) > 1:
            heatmap_data = self.data.pivot_table(
                values='result_numeric', 
                index='rating_bin', 
                columns='time_class', 
                aggfunc='mean'
            )
            
            if not heatmap_data.empty:
                # Create modern heatmap
                im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                
                # Set labels
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
                ax.set_yticks(range(len(heatmap_data.index)))
                ax.set_yticklabels(heatmap_data.index, fontsize=9)
                
                # Add value annotations
                for i in range(len(heatmap_data.index)):
                    for j in range(len(heatmap_data.columns)):
                        value = heatmap_data.iloc[i, j]
                        if not np.isnan(value):
                            text_color = 'white' if value < 0.5 else 'black'
                            ax.text(j, i, f'{value:.1%}', ha='center', va='center', 
                                color=text_color, fontweight='bold', fontsize=9)
                
                ax.set_title('🔥 RATING × TIME HEATMAP', fontweight='bold', 
                            color='white', fontsize=12, pad=15)
            else:
                ax.text(0.5, 0.5, 'Insufficient Data\nfor Heatmap', ha='center', va='center', 
                    transform=ax.transAxes, color='white', fontsize=12)
                ax.set_title('🔥 RATING × TIME HEATMAP', fontweight='bold', 
                            color='white', fontsize=12, pad=15)
        else:
            ax.text(0.5, 0.5, 'Need Multiple\nTime Controls', ha='center', va='center', 
                transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title('🔥 RATING × TIME HEATMAP', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
                    

    def create_interactive_world_map(self) -> None:
        """
        Create an interactive world map showing performance by country.
        """
        try:
            # Calculate win rates by country using mapped names
            country_stats = self.data.groupby('opponent_country_mapped').agg({
                'result_numeric': ['mean', 'count'],
                'player_rating': 'mean',
                'opponent_rating': 'mean'
            }).round(3)
            
            country_stats.columns = ['win_rate', 'games', 'avg_player_rating', 'avg_opponent_rating']
            country_stats = country_stats.reset_index()
            
            # Filter countries with at least 2 games and valid mappings
            country_stats = country_stats[
                (country_stats['games'] >= 2) & 
                (country_stats['opponent_country_mapped'].notna())
            ]
            
            if len(country_stats) == 0:
                print("⚠️ No countries with sufficient data for mapping")
                return
            
            # Create modern choropleth map
            fig = px.choropleth(
                country_stats,
                locations='opponent_country_mapped',
                locationmode='country names',
                color='win_rate',
                hover_name='opponent_country_mapped',
                hover_data={
                    'games': True,
                    'avg_player_rating': ':.0f',
                    'avg_opponent_rating': ':.0f',
                    'win_rate': ':.1%'
                },
                color_continuous_scale='Viridis',
                range_color=[0, 1],
                title='🌍 Chess Performance by Country',
                labels={'win_rate': 'Win Rate', 'opponent_country_mapped': 'Country'}
            )
            
            # Apply modern dark theme to map
            fig.update_layout(
                title_font_size=24,
                title_font_color='white',
                title_x=0.5,
                paper_bgcolor='#1E1E2E',
                plot_bgcolor='#1E1E2E',
                font_color='white',
                geo=dict(
                    showframe=False, 
                    showcoastlines=True,
                    bgcolor='#2A2D3A',
                    coastlinecolor='#404552',
                    projection_type='equirectangular'
                ),
                height=700,
                width=1200,
                coloraxis_colorbar=dict(
                    title_font_color='white',
                    tickfont_color='white'
                )
            )
            
            # Save the map
            map_path = os.path.join(self.output_folder, 'chess_world_map.html')
            fig.write_html(map_path)
            print(f"🗺️ Interactive world map saved to: {map_path}")
            
            # Also save as PNG
            try:
                map_png_path = os.path.join(self.output_folder, 'chess_world_map.png')
                fig.write_image(map_png_path, width=1200, height=700)
                print(f"🗺️ World map PNG saved to: {map_png_path}")
            except:
                print("📝 Note: PNG export requires kaleido. Install with: pip install kaleido")
            
            fig.show()
            
            # Print mapping diagnostics
            print("\n🗺️ Country Mapping Results:")
            mapped_countries = country_stats['opponent_country_mapped'].unique()
            print(f"✅ Successfully mapped {len(mapped_countries)} countries")
            
            # Show unmapped countries
            original_countries = set(self.data['opponent_country'].unique())
            mapped_originals = set(self.data[self.data['opponent_country_mapped'].notna()]['opponent_country'].unique())
            unmapped = original_countries - mapped_originals
            if unmapped:
                print(f"⚠️ Unmapped countries: {list(unmapped)}")
            
        except Exception as e:
            print(f"⚠️ Could not create world map: {str(e)}")
            print("📝 Note: This requires plotly and proper country name matching")

    def create_advanced_analytics(self) -> None:
        """
        Create additional advanced analytics visualizations with modern styling.
        """
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#1E1E2E')
        
        # Create custom grid
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25, 
                     left=0.06, right=0.96, top=0.92, bottom=0.1)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Detailed Results
        ax2 = fig.add_subplot(gs[0, 1])  # Rating Progression
        ax3 = fig.add_subplot(gs[0, 2])  # Time Analysis
        ax4 = fig.add_subplot(gs[1, 0])  # Head-to-head
        ax5 = fig.add_subplot(gs[1, 1])  # Performance Radar
        ax6 = fig.add_subplot(gs[1, 2])  # Weekly Heatmap
        
        # Apply modern styling
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        for ax in axes:
            self._style_axis(ax)
        
        # Plot visualizations
        self._plot_detailed_results_modern(ax1)
        # self._plot_rating_progression_advanced(ax2)
        self._plot_mini_rating_progression(ax2)
        self._plot_time_analysis_modern(ax3)
        self._plot_head_to_head_modern(ax4)
        self._plot_performance_radar(ax5)
        self._plot_weekly_heatmap(ax6)
        
        # Add title
        fig.suptitle('🔬 ADVANCED CHESS ANALYTICS', 
                    fontsize=24, fontweight='bold', color='white') # y=0.96
        
        
        # Save advanced analytics
        advanced_path = os.path.join(self.output_folder, 'chess_advanced_analytics.png')
        plt.savefig(advanced_path, dpi=300, bbox_inches='tight', 
                   facecolor='#1E1E2E', edgecolor='none')
        print(f"🔬 Advanced analytics saved to: {advanced_path}")
        
        plt.show()

    def _plot_detailed_results_modern(self, ax) -> None:
        """Plot detailed breakdown with modern styling."""
        result_counts = self.data['result'].value_counts().head(8)
        
        # Modern gradient bars
        bars = ax.bar(range(len(result_counts)), result_counts.values, 
                     color=self.colors['primary'], alpha=0.8,
                     edgecolor='white', linewidth=1.5)
        
        # Add glow effect
        for i, (bar, count) in enumerate(zip(bars, result_counts.values)):
            ax.bar(i, count, color=self.colors['primary'], alpha=0.3, width=0.8)
            
            # Add value labels
            ax.text(i, count + 0.5, str(count), ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_xticks(range(len(result_counts)))
        ax.set_xticklabels(result_counts.index, rotation=45, ha='right', fontsize=9)
        ax.set_title('🎯 DETAILED RESULTS', fontweight='bold', 
                    color='white', fontsize=12, pad=15)
        ax.set_ylabel('Count', color='white', fontsize=10)

    def _plot_rating_progression_advanced(self, ax) -> None:
        """Advanced rating progression with trend analysis."""
        if 'date' in self.data.columns and self.data['date'].notna().any():
            data_sorted = self.data.sort_values('date')
            
            # Calculate various moving averages
            data_sorted['rating_ma_5'] = data_sorted['player_rating'].rolling(window=5, min_periods=1).mean()
            data_sorted['rating_ma_10'] = data_sorted['player_rating'].rolling(window=10, min_periods=1).mean()
            
            x_pos = range(len(data_sorted))
            
            # Plot actual ratings as scatter
            wins = data_sorted['result_category'] == 'win'
            losses = data_sorted['result_category'] == 'loss'
            
            ax.scatter(np.array(x_pos)[wins], data_sorted['player_rating'][wins], 
                      color=self.colors['success'], alpha=0.6, s=20, label='Wins')
            ax.scatter(np.array(x_pos)[losses], data_sorted['player_rating'][losses], 
                      color=self.colors['danger'], alpha=0.6, s=20, label='Losses')
            
            # Plot moving averages
            ax.plot(x_pos, data_sorted['rating_ma_5'], 
                   color=self.colors['accent'], linewidth=2, alpha=0.8, label='5-Game Avg')
            ax.plot(x_pos, data_sorted['rating_ma_10'], 
                   color=self.colors['primary'], linewidth=3, label='10-Game Avg')
            
            ax.set_title('📈 RATING PROGRESSION', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            ax.set_xlabel('Game Number', color='white', fontsize=10)
            ax.set_ylabel('Rating', color='white', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Date Data\nAvailable', ha='center', va='center', 
                   transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title('📈 RATING PROGRESSION', fontweight='bold', 
                        color='white', fontsize=12, pad=15)

    def _plot_time_analysis_modern(self, ax) -> None:
        """Modern time analysis visualization."""
        if 'date' in self.data.columns and self.data['date'].notna().any():
            # Monthly analysis
            monthly_perf = self.data.groupby(self.data['date'].dt.month)['result_numeric'].mean()
            
            bars = ax.bar(range(len(monthly_perf)), monthly_perf.values, 
                         color=self.colors['secondary'], alpha=0.8)
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(len(monthly_perf)))
            ax.set_xticklabels([months[i-1] for i in monthly_perf.index], fontsize=9)
            ax.set_title('📅 MONTHLY PERFORMANCE', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            ax.set_ylabel('Win Rate', color='white', fontsize=10)
        else:
            # Time control analysis
            time_control_perf = self.data.groupby('time_control')['result_numeric'].mean().head(6)
            
            bars = ax.bar(range(len(time_control_perf)), time_control_perf.values, 
                         color=self.colors['warning'], alpha=0.8)
            
            ax.set_xticks(range(len(time_control_perf)))
            ax.set_xticklabels(time_control_perf.index, rotation=45, ha='right', fontsize=8)
            ax.set_title('⏲️ TIME CONTROL ANALYSIS', fontweight='bold', 
                        color='white', fontsize=12, pad=15)
            ax.set_ylabel('Win Rate', color='white', fontsize=10)

    def _plot_head_to_head_modern(self, ax) -> None:
        """Modern head-to-head visualization."""
        opponent_records = self.data.groupby('opponent_username').agg({
            'result_numeric': ['mean', 'count']
        })
        opponent_records.columns = ['win_rate', 'games']
        opponent_records = opponent_records[opponent_records['games'] >= 3].sort_values('win_rate', ascending=True).tail(8)
        
        if len(opponent_records) > 0:
            # Create horizontal bars with gradient
            bars = ax.barh(range(len(opponent_records)), opponent_records['win_rate'], 
                          color=self.colors['primary'], alpha=0.8)
            
            # Add glow effect
            ax.barh(range(len(opponent_records)), opponent_records['win_rate'], 
                   color=self.colors['primary'], alpha=0.3)
            
            # Add annotations
            for i, (_, row) in enumerate(opponent_records.iterrows()):
                ax.text(row['win_rate'] + 0.02, i, f"{row['win_rate']:.1%} ({int(row['games'])})", 
                       va='center', fontsize=9, color='white', fontweight='bold')
            
            ax.set_yticks(range(len(opponent_records)))
            ax.set_yticklabels(opponent_records.index, fontsize=9)
            ax.set_xlabel('Win Rate', color='white', fontsize=10)
            ax.set_xlim(0, 1.2)
        
        ax.set_title('🤝 HEAD-TO-HEAD RECORDS', fontweight='bold', 
                    color='white', fontsize=12, pad=15)

    def _plot_performance_radar(self, ax) -> None:
        """Create a performance radar chart."""
        # Calculate metrics for radar chart
        metrics = {
            'Win Rate': self.data['result_numeric'].mean(),
            'Rating Avg': min(1.0, self.data['player_rating'].mean() / 2500),  # Normalize to 0-1
            'vs Higher Rated': self.data[self.data['rating_difference'] < 0]['result_numeric'].mean() if len(self.data[self.data['rating_difference'] < 0]) > 0 else 0,
            'Consistency': max(0, 1 - self.data['result_numeric'].std()),  # Lower std = higher consistency
            'Activity': min(1.0, len(self.data) / 1000),  # Normalize games played
        }
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values for completing the circle
        values += values[:1]
        
        # Clear and set up polar plot
        ax.clear()
        ax = plt.subplot(2, 3, 5, projection='polar')
        self._style_polar_axis(ax)
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, color='white')
        ax.set_ylim(0, 1)
        ax.set_title('📊 PERFORMANCE RADAR', fontweight='bold', 
                    color='white', pad=20)

    def _style_polar_axis(self, ax):
        """Style polar axis for radar chart."""
        ax.set_facecolor('#2A2D3A')
        ax.grid(True, color='#404552', alpha=0.3)
        ax.tick_params(colors='white')

    def _plot_weekly_heatmap(self, ax) -> None:
        """Create a weekly performance heatmap."""
        if 'date' in self.data.columns and self.data['date'].notna().any():
            # Create week/day matrix
            self.data['week'] = self.data['date'].dt.isocalendar().week
            self.data['weekday'] = self.data['date'].dt.weekday
            
            # Group by week and weekday
            weekly_perf = self.data.groupby(['week', 'weekday'])['result_numeric'].mean().unstack(fill_value=0)
            
            if len(weekly_perf) > 1:
                # Create heatmap
                im = ax.imshow(weekly_perf.values[-10:], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                
                # Labels
                weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                ax.set_xticks(range(min(7, len(weekdays))))
                ax.set_xticklabels(weekdays[:min(7, len(weekdays))], fontsize=9)
                
                recent_weeks = weekly_perf.index[-10:]
                ax.set_yticks(range(len(recent_weeks)))
                ax.set_yticklabels([f'W{w}' for w in recent_weeks], fontsize=8)
                
                ax.set_title('📅 WEEKLY HEATMAP', fontweight='bold', 
                            color='white', fontsize=12, pad=15)
            else:
                ax.text(0.5, 0.5, 'Insufficient\nWeekly Data', ha='center', va='center', 
                       transform=ax.transAxes, color='white', fontsize=12)
                ax.set_title('📅 WEEKLY HEATMAP', fontweight='bold', 
                            color='white', fontsize=12, pad=15)
        else:
            ax.text(0.5, 0.5, 'No Date Data\nfor Heatmap', ha='center', va='center', 
                   transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title('📅 WEEKLY HEATMAP', fontweight='bold', 
                        color='white', fontsize=12, pad=15)

    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive text report of performance metrics.
        
        Returns:
            str: Formatted performance report
        """
        total_games = len(self.data)
        wins = len(self.data[self.data['result_category'] == 'win'])
        losses = len(self.data[self.data['result_category'] == 'loss'])
        draws = len(self.data[self.data['result_category'] == 'draw'])
        
        win_rate = wins / total_games if total_games > 0 else 0
        avg_rating = self.data['player_rating'].mean()
        avg_opponent_rating = self.data['opponent_rating'].mean()
        
        # Best and worst performing time controls
        time_performance = self.data.groupby('time_class')['result_numeric'].mean()
        best_time_control = time_performance.idxmax() if len(time_performance) > 0 else 'N/A'
        worst_time_control = time_performance.idxmin() if len(time_performance) > 0 else 'N/A'
        
        # Color preference
        color_performance = self.data.groupby('player_color')['result_numeric'].mean()
        
        # Most common results
        top_results = self.data['result'].value_counts().head(3)
        
        report = f"""
🏆 CHESS PERFORMANCE ANALYTICS REPORT
{'='*50}

📊 OVERALL STATISTICS
• Total Games Played: {total_games:,}
• Wins: {wins} ({wins/total_games:.1%})
• Losses: {losses} ({losses/total_games:.1%})
• Draws: {draws} ({draws/total_games:.1%})
• Overall Win Rate: {win_rate:.1%}

📈 RATING ANALYSIS
• Your Average Rating: {avg_rating:.0f}
• Average Opponent Rating: {avg_opponent_rating:.0f}
• Rating Advantage: {avg_rating - avg_opponent_rating:+.0f}

⏱️ TIME CONTROL PERFORMANCE
• Best Performance: {best_time_control} ({time_performance.get(best_time_control, 0):.1%} win rate)
• Worst Performance: {worst_time_control} ({time_performance.get(worst_time_control, 0):.1%} win rate)

♟️ COLOR PREFERENCE
• As White: {color_performance.get('white', 0):.1%} win rate
• As Black: {color_performance.get('black', 0):.1%} win rate

🎯 MOST COMMON RESULTS
"""
        
        for result, count in top_results.items():
            report += f"• {result}: {count} games ({count/total_games:.1%})\n"
        
        report += f"""
🌍 GEOGRAPHIC INSIGHTS
• Countries Faced: {self.data['opponent_country'].nunique()}
• Most Common Opponent Country: {self.data['opponent_country'].mode().iloc[0] if not self.data['opponent_country'].empty else 'N/A'}

💡 KEY INSIGHTS
• Rating correlation with results: {self.data[['rating_difference', 'result_numeric']].corr().iloc[0,1]:.3f}
• Most active time control: {self.data['time_class'].mode().iloc[0] if not self.data['time_class'].empty else 'N/A'}
• Average games per opponent: {self.data.groupby('opponent_username').size().mean():.1f}
"""
        
        if 'date' in self.data.columns and self.data['date'].notna().any():
            date_range = self.data['date'].max() - self.data['date'].min()
            games_per_month = total_games / max(1, date_range.days / 30.44)
            report += f"• Games per month: {games_per_month:.1f}\n"
        
        return report

    def export_results(self, filename: str = 'chess_performance_analysis.csv') -> None:
        """
        Export analyzed results to CSV for further analysis.
        
        Args:
            filename (str): Output filename for the CSV export
        """
        # Create summary statistics
        summary_stats = {
            'total_games': len(self.data),
            'win_rate': len(self.data[self.data['result_category'] == 'win']) / len(self.data),
            'avg_rating': self.data['player_rating'].mean(),
            'avg_opponent_rating': self.data['opponent_rating'].mean(),
        }
        
        # Export raw data with additional calculated columns
        export_data = self.data.copy()
        export_data.to_csv(filename, index=False)
        
        print(f"📁 Results exported to {filename}")
        print(f"📊 Summary: {summary_stats}")

    def print_data_diagnostics(self) -> None:
        """Print diagnostic information about the loaded data."""
        print("\n🔍 DATA DIAGNOSTICS")
        print("="*50)
        print(f"📊 Total records: {len(self.data)}")
        print(f"📅 Date range: {self.data['date'].min()} to {self.data['date'].max()}" if 'date' in self.data.columns else "📅 No date information available")
        print(f"🏆 Result types: {list(self.data['result'].unique())}")
        print(f"⏱️ Time classes: {list(self.data['time_class'].unique())}")
        print(f"🌍 Countries: {self.data['opponent_country'].nunique()} unique")
        print(f"👥 Opponents: {self.data['opponent_username'].nunique()} unique")
        
        # Country mapping diagnostic
        mapped_countries = self.data['opponent_country_mapped'].nunique()
        original_countries = self.data['opponent_country'].nunique()
        print(f"🗺️ Country mapping: {mapped_countries}/{original_countries} countries successfully mapped")

def main():
    """
    Main execution function demonstrating the chess analytics capabilities.
    """
    print("🚀 Starting Chess Performance Analysis...")
    
    # Initialize analyzer with output folder
    try:
        analyzer = ChessPerformanceAnalyzer('game_results.json', 'chess_analytics_output')
        
        # Print diagnostics
        analyzer.print_data_diagnostics()
        
        # Filter for specific player (modify as needed)
        analyzer.filter_data(['Yedfa'])  # Replace with your username
        
        # Generate comprehensive dashboard
        print("\n📈 Creating performance dashboard...")
        analyzer.create_performance_dashboard()
        
        # Create advanced analytics
        print("\n🔬 Creating advanced analytics...")
        analyzer.create_advanced_analytics()
        
        # Create interactive world map
        print("\n🌍 Creating interactive world map...")
        analyzer.create_interactive_world_map()
        
        # Generate and print performance report
        print("\n📋 Generating performance report...")
        report = analyzer.generate_performance_report()
        print(report)
        
        # Save report to file
        report_path = os.path.join(analyzer.output_folder, 'performance_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_path}")
        
        # Export results
        print("\n💾 Exporting results...")
        csv_path = os.path.join(analyzer.output_folder, 'chess_performance_data.csv')
        analyzer.export_results(csv_path)
        
        print(f"\n✅ Analysis complete! All files saved to: {analyzer.output_folder}")
        print("📁 Generated files:")
        print("   • chess_performance_dashboard.png - Main dashboard")
        print("   • chess_advanced_analytics.png - Advanced analysis")
        print("   • chess_world_map.html - Interactive world map")
        print("   • chess_world_map.png - World map image")
        print("   • performance_report.txt - Detailed text report")
        print("   • chess_performance_data.csv - Raw analyzed data")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Please ensure your game_results.json file is in the correct format and location.")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()        