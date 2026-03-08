"""
Real-time Training Dashboard for CompositeMotion
Monitors discriminator vs policy balance and training health
"""
import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class TrainingDashboard:
    """Real-time training monitoring dashboard"""
    
    def __init__(self, csv_path, window_size=None, frames_per_cycle=30, cycles_per_episode=1, fig_name=None):
        self.csv_path = csv_path
        self.window_size = window_size
        self.frames_per_cycle = frames_per_cycle
        self.cycles_per_episode = cycles_per_episode
        if fig_name == None:
            fig_name = os.path.basename(os.path.dirname(csv_path))
        
        # Data buffers
        if self.window_size is None:
            # No limit - store all data points
            self.epochs = deque()
            self.lifetimes = deque()
            self.policy_losses = deque()
            self.value_losses = deque()
            self.reward_means = deque()
            self.all_rewards = {}  # {reward_name: deque} for all reward columns
            
            # Discriminator data
            self.disc_scores_real = {}  # {name: deque}
            self.disc_scores_fake = {}  # {name: deque}
            self.disc_rewards = {}  # {name: deque}
        else:
            # Limited window
            self.epochs = deque(maxlen=window_size)
            self.lifetimes = deque(maxlen=window_size)
            self.policy_losses = deque(maxlen=window_size)
            self.value_losses = deque(maxlen=window_size)
            self.reward_means = deque(maxlen=window_size)
            self.all_rewards = {}  # {reward_name: deque}
            
            # Discriminator data
            self.disc_scores_real = {}  # {name: deque}
            self.disc_scores_fake = {}  # {name: deque}
            self.disc_rewards = {}  # {name: deque}
        
        # Setup plot
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 6), num=fig_name)
        self.fig.suptitle(f'{fig_name} Training Analysis', fontsize=14, fontweight='bold')
        
        # Initialize lines
        self._init_plots()
        
        # Last read position
        self.last_line = 0
        
        # Cache for disc lines to avoid recreation
        self.disc_lines = {}
        self.gap_lines = {}
        self.reward_lines = {}
        
        # Discriminator name mapping for clean legends
        self.disc_name_map = {}
    
    def _init_plots(self):
        """Initialize plot axes"""
        # Plot 1: Lifetime
        self.ax_lifetime = self.axes[0, 0]
        self.ax_lifetime.set_title('Episode Lifetime')
        self.ax_lifetime.set_xlabel('Epoch')
        self.ax_lifetime.set_ylabel('Steps')
        self.line_lifetime, = self.ax_lifetime.plot([], [], 'b-', linewidth=1.5, label='Lifetime')
        self.ax_lifetime.grid(True, alpha=0.2)
        
        # Add benchmark lines for lifetime based on frames_per_cycle and cycles_per_episode
        self.benchmark_texts = []  # Store text objects to avoid recreation
        for i in range(self.cycles_per_episode + 1):
            benchmark_y = i * self.frames_per_cycle
            line = self.ax_lifetime.axhline(y=benchmark_y, color='darkgray', linewidth=0.8, linestyle='--')
            txt = self.ax_lifetime.text(1.02, benchmark_y, f'cy[{i}]:{benchmark_y:3d}s', 
                                         transform=self.ax_lifetime.get_yaxis_transform(),
                                        fontsize=6, va='center', ha='left', alpha=0.7, color='magenta')
            self.benchmark_texts.append(txt)
        
        # Plot 2: Discriminator Scores
        self.ax_disc = self.axes[0, 1]
        self.ax_disc.set_title('Discriminator Scores')
        self.ax_disc.set_xlabel('Epoch')
        self.ax_disc.set_ylabel('Score')
        # Score Baselines
        self.ax_disc.axhline(y=0, color='k', linewidth=.8, linestyle='-', alpha=0.3)
        self.ax_disc.axhline(y=0.2, color='g', linewidth=.8, linestyle='--', alpha=0.3, label='Target Real (>0.2)')
        self.ax_disc.axhline(y=-0.2, color='r', linewidth=.8, linestyle='--', alpha=0.3, label='Target Fake (<-0.2)')
        self.ax_disc.legend(loc='upper right', fontsize='small')
        self.ax_disc.grid(True, alpha=0.3)
        
        # Cache for disc lines to avoid recreation
        self.disc_lines = {}
        self.gap_lines = {}
        
        # Plot 3: All Rewards
        self.ax_reward = self.axes[0, 2]
        self.ax_reward.set_title('All Reward Metrics')
        self.ax_reward.set_xlabel('Epoch')
        self.ax_reward.set_ylabel('Reward')
        self.reward_lines = {}
        self.ax_reward.grid(True, alpha=0.3)
        
        # Plot 4: Losses
        self.ax_loss = self.axes[1, 0]
        self.ax_loss.set_title('Policy & Value Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.line_policy_loss, = self.ax_loss.plot([], [], 'r-', linewidth=1.5, label='Policy')
        self.line_value_loss, = self.ax_loss.plot([], [], 'm-', linewidth=1.5, label='Value')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # Plot 5: Discriminator Gap
        self.ax_gap = self.axes[1, 1]
        self.ax_gap.set_title('Discriminator Gap (Real - Fake)')
        self.ax_gap.set_xlabel('Epoch')
        self.ax_gap.set_ylabel('Gap')
        # Gap Baselines
        self.ax_gap.axhline(y=0.25, color='k', linewidth=.8, linestyle='--', alpha=0.5, label='Warning (0.25)')
        self.ax_gap.axhline(y=0.5, color='k', linewidth=.8, linestyle='-', alpha=0.5, label='Critical (0.5)')
        self.ax_gap.legend()
        self.ax_gap.grid(True, alpha=0.3)
        
        # Plot 6: Health Status
        self.ax_health = self.axes[1, 2]
        self.ax_health.set_title('Training Health')
        self.ax_health.axis('off')
        self.health_text = self.ax_health.text(0.5, 0.5, '', transform=self.ax_health.transAxes,
                                                  fontsize=8, verticalalignment='center',
                                                horizontalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
    
    def _parse_disc_name(self, col_name, prefix):
        """Parse discriminator name from column header, handling special characters"""
        if col_name.startswith(prefix):
            name = col_name[len(prefix):]
            # Clean up the name for display (replace / with -, remove extra underscores)
            clean_name = name.replace('/', '-').replace('__', '_').strip('_')
            return clean_name
        return None
    
    def read_new_data(self):
        """Read new data from CSV file"""
        if not os.path.exists(self.csv_path):
            return False
        
        try:
            with open(self.csv_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= self.last_line:
                return False
            
            # Parse header if first read
            if self.last_line == 0:
                header = lines[0].strip().split(',')
                self.disc_names = []
                self.reward_columns = []
                
                for col in header:
                    col = col.strip()
                    # Discriminator columns
                    if col.startswith('score_real_'):
                        name = self._parse_disc_name(col, 'score_real_')
                        if name:
                            self.disc_names.append(name)
                            self.disc_name_map[col] = name
                            if self.window_size is None:
                                self.disc_scores_real[name] = deque()
                                self.disc_scores_fake[name] = deque()
                                self.disc_rewards[name] = deque()
                            else:
                                self.disc_scores_real[name] = deque(maxlen=self.window_size)
                                self.disc_scores_fake[name] = deque(maxlen=self.window_size)
                                self.disc_rewards[name] = deque(maxlen=self.window_size)
                    elif col.startswith('score_fake_'):
                        name = self._parse_disc_name(col, 'score_fake_')
                        if name and name not in self.disc_names:
                            # Map fake column to existing name
                            self.disc_name_map[col] = name
                    elif col.startswith('disc_reward_'):
                        name = self._parse_disc_name(col, 'disc_reward_')
                        if name:
                            self.disc_name_map[col] = name
                    # All reward columns (containing 'reward')
                    elif 'reward' in col.lower():
                        self.reward_columns.append(col)
                        if self.window_size is None:
                            self.all_rewards[col] = deque()
                        else:
                            self.all_rewards[col] = deque(maxlen=self.window_size)
                
                self.last_line = 1
            
            # Parse new rows
            header_list = lines[0].strip().split(',')
            reader = csv.DictReader(lines[self.last_line:], fieldnames=header_list)
            for row in reader:
                try:
                    self.epochs.append(int(row['epoch']))
                    self.lifetimes.append(float(row['lifetime']))
                    self.policy_losses.append(float(row['policy_loss']))
                    self.value_losses.append(float(row['value_loss']))
                    self.reward_means.append(float(row['reward_mean']))
                    
                    # All reward columns
                    for col in self.reward_columns:
                        if col in row and row[col]:
                            self.all_rewards[col].append(float(row[col]))
                    
                    # Discriminator data
                    for name in self.disc_names:
                        real_key = f'score_real_{name}'.replace('-', '/').replace('_', '__')
                        fake_key = f'score_fake_{name}'.replace('-', '/').replace('_', '__')
                        reward_key = f'disc_reward_{name}'.replace('-', '/').replace('_', '__')
                        
                        # Try to find matching columns in row
                        for key in row.keys():
                            key_stripped = key.strip()
                            if key_stripped.startswith('score_real_') and self._parse_disc_name(key_stripped, 'score_real_') == name:
                                if row[key_stripped]:
                                    self.disc_scores_real[name].append(float(row[key_stripped]))
                            elif key_stripped.startswith('score_fake_') and self._parse_disc_name(key_stripped, 'score_fake_') == name:
                                if row[key_stripped]:
                                    self.disc_scores_fake[name].append(float(row[key_stripped]))
                            elif key_stripped.startswith('disc_reward_') and self._parse_disc_name(key_stripped, 'disc_reward_') == name:
                                if row[key_stripped]:
                                    self.disc_rewards[name].append(float(row[key_stripped]))
                    
                    self.last_line += 1
                except (ValueError, KeyError) as e:
                    continue
            
            return True
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
    
    def _get_recent_values(self, data_deque, use_avg=False):
        """Get recent values - either last entry or average of last window_size/2 entries"""
        if len(data_deque) == 0:
            return 0
        
        if self.window_size is None or not use_avg:
            return data_deque[-1]
        else:
            # Take last window_size/2 entries and average
            n_samples = max(1, self.window_size // 2)
            recent = list(data_deque)[-n_samples:]
            return np.mean(recent)
    
    def _calculate_trend(self, data_deque, n_points=None):
        """Calculate trend using linear regression on recent points"""
        if len(data_deque) < 2:
            return 0
        
        if self.window_size is None:
            n_points = min(10, len(data_deque))
        else:
            n_points = min(self.window_size // 2, len(data_deque))
        
        if n_points < 2:
            return 0
        
        recent = list(data_deque)[-n_points:]
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Simple linear regression (poly deg = 1)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0] # return m & c from  y = mx + c
            return slope
        return 0
    
    def update_plots(self, frame):
        """Update all plots"""
        if not self.read_new_data():
            return
        
        if len(self.epochs) == 0:
            return
        
        epochs = list(self.epochs)
        
        # Calculate x-axis limits based on window_size
        if self.window_size is None:
            # Show all data points from epoch 1 to n
            x_min = min(epochs)
            x_max = max(epochs) + 1
        else:
            # Show only the last window_size epochs
            x_min = max(min(epochs), epochs[-1] - self.window_size + 1)
            x_max = epochs[-1] + 1
        
        # Update lifetime plot
        self.line_lifetime.set_data(epochs, list(self.lifetimes))
        self.ax_lifetime.set_xlim(x_min, x_max)
        
        # Dynamic ylim based on max lifetime and upper benchmark + offset
        max_lifetime = max(self.lifetimes) if self.lifetimes else 0
        no_of_cycles = np.ceil(max_lifetime / self.frames_per_cycle)
        upper_benchmark = no_of_cycles * self.frames_per_cycle + 5
        self.ax_lifetime.set_ylim(0, upper_benchmark)
        
        # Update discriminator scores - reuse lines instead of clearing
        self.ax_disc.set_xlim(x_min, x_max)
        
        # Clear existing disc lines only (not baselines)
        for line_list in self.disc_lines.values():
            for line in line_list:
                line.remove()
        self.disc_lines.clear()
        
        # Collect score extremes for dynamic y-axis
        min_score, max_score = -0.5, 0.5
        
        for name in self.disc_names:
            if name in self.disc_scores_real and len(self.disc_scores_real[name]) > 0:
                real_scores = list(self.disc_scores_real[name])
                fake_scores = list(self.disc_scores_fake[name])
                disc_epochs = epochs[-len(real_scores):]
                
                # Dynamic y-axis calculation
                min_score = min(min_score, min(real_scores), min(fake_scores))
                max_score = max(max_score, max(real_scores), max(fake_scores))
                
                # Use clean name for legend
                display_name = name.replace('_', ' ').title()
                real_line, = self.ax_disc.plot(disc_epochs, real_scores, 'g-', 
                                            linewidth=1.5, label=f'{display_name} (real)', alpha=0.7)
                fake_line, = self.ax_disc.plot(disc_epochs, fake_scores, 'r-', 
                                            linewidth=1.5, label=f'{display_name} (fake)', alpha=0.7)
                self.disc_lines[name] = [real_line, fake_line]
        
        self.ax_disc.set_ylim(min_score - 0.1, max_score + 0.1)
        
        # Update legend only once (baselines are static from _init_plots)
        if self.disc_names:
            self.ax_disc.legend(loc='upper right', fontsize='small')
        
        # Update ALL reward plots
        self.ax_reward.set_xlim(x_min, x_max)
        
        # Clear existing reward lines
        for line in self.reward_lines.values():
            line.remove()
        self.reward_lines.clear()
        
        # Color map for different reward lines
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.reward_columns))))
        
        for idx, col in enumerate(self.reward_columns):
            if col in self.all_rewards and len(self.all_rewards[col]) > 0:
                reward_data = list(self.all_rewards[col])
                reward_epochs = epochs[-len(reward_data):]
                
                # Clean column name for legend
                display_name = col.replace('reward_', '').replace('_', ' ').title()
                
                color = colors[idx % len(colors)]
                line, = self.ax_reward.plot(reward_epochs, reward_data, '-', 
                                           linewidth=1.2, label=display_name, 
                                           color=color, alpha=0.8)
                self.reward_lines[col] = line
        
        if len(self.reward_columns) > 0:
            # Update y-axis based on all reward data
            all_reward_values = []
            for col in self.reward_columns:
                if col in self.all_rewards and len(self.all_rewards[col]) > 0:
                    all_reward_values.extend(self.all_rewards[col])
            
            if all_reward_values:
                min_reward = min(all_reward_values)
                max_reward = max(all_reward_values)
                self.ax_reward.set_ylim(min_reward - 0.1, max_reward + 0.1)
            self.ax_reward.legend(loc='upper right', fontsize='x-small')
        
        # Update loss plot
        self.line_policy_loss.set_data(epochs, list(self.policy_losses))
        self.line_value_loss.set_data(epochs, list(self.value_losses))
        self.ax_loss.set_xlim(x_min, x_max)
        if len(self.policy_losses) > 0:
            self.ax_loss.set_ylim(min(min(self.policy_losses), min(self.value_losses)) - 0.1,
                                max(max(self.policy_losses), max(self.value_losses)) + 0.1)
        
        # Update gap plot - reuse lines instead of clearing
        self.ax_gap.set_xlim(x_min, x_max)
        
        # Clear existing gap lines only (not baselines)
        for line in self.gap_lines.values():
            line.remove()
        self.gap_lines.clear()
        
        for name in self.disc_names:
            if name in self.disc_scores_real and len(self.disc_scores_real[name]) > 0:
                real_scores = list(self.disc_scores_real[name])
                fake_scores = list(self.disc_scores_fake[name])
                gaps = [r - f for r, f in zip(real_scores, fake_scores)]
                gap_epochs = epochs[-len(gaps):]
                
                display_name = name.replace('_', ' ').title()
                gap_line, = self.ax_gap.plot(gap_epochs, gaps, linewidth=1.5, label=display_name)
                self.gap_lines[name] = gap_line
        
        if self.disc_names:
            self.ax_gap.legend(fontsize=8)
        
        # Update health status
        self._update_health_text()
        
        self.fig.canvas.draw_idle()
    
    def _update_health_text(self):
        """Update health status text"""
        if len(self.epochs) == 0:
            return
        
        # Use averaged values when window_size is set
        use_avg = self.window_size is not None
        
        latest_epoch = self._get_recent_values(self.epochs, use_avg=False)
        latest_lifetime = self._get_recent_values(self.lifetimes, use_avg)
        latest_policy_loss = self._get_recent_values(self.policy_losses, use_avg)
        latest_value_loss = self._get_recent_values(self.value_losses, use_avg)
        latest_reward_mean = self._get_recent_values(self.reward_means, use_avg)
        
        # Calculate reward trend
        reward_trend = self._calculate_trend(self.reward_means)
        
        # Calculate health metrics
        health_status = []
        
        # Lifetime health - UPDATED with frames_per_cycle thresholds
        if latest_lifetime < self.frames_per_cycle / 4:
            lifetime_status = "CRITICAL"
        elif latest_lifetime < self.frames_per_cycle / 2:
            lifetime_status = "WARNING"
        elif latest_lifetime < self.frames_per_cycle:
            lifetime_status = "PROGRESS"
        elif latest_lifetime < self.frames_per_cycle * 2:
            lifetime_status = "OK ❤"
        else:
            lifetime_status = "SWEET"
        health_status.append(f"Lifetime: {latest_lifetime:.1f} [{lifetime_status}]")
        
        # Reward Mean health - NEW with trend analysis
        health_status.append(f"Reward: {latest_reward_mean:.3f} (trend={reward_trend:.4f}) {"[✓]" if reward_trend > 0 else "[✗]"}")
        
        # Discriminator health
        for name in self.disc_names:
            if name in self.disc_scores_real and len(self.disc_scores_real[name]) > 0:
                real = self._get_recent_values(self.disc_scores_real[name], use_avg)
                fake = self._get_recent_values(self.disc_scores_fake[name], use_avg)
                gap = real - fake
                
                if gap > 0.5:
                    disc_status = "CRITICAL"
                elif gap > 0.3:
                    disc_status = "WARNING"
                elif gap > 0.25:
                    disc_status = "OK ✗"
                elif gap > 0.2:
                    disc_status = "OK ✓"
                else:
                    disc_status = "SWEET"
                health_status.append(f"Disc {name}: gap={gap:.3f} [{disc_status}]")
        
        # Value loss health
        if latest_value_loss > 1.0:
            value_status = "HIGH"
        elif latest_value_loss > 0.5:
            value_status = "ELEVATED"
        else:
            value_status = "OK"
        health_status.append(f"Value Loss: {latest_value_loss:.4f} [{value_status}]")
        
        # Policy loss health
        if latest_policy_loss > -0.001:
            policy_status = "STUCK"
        elif latest_policy_loss < -0.1:
            policy_status = "UNSTABLE"
        else:
            policy_status = "OK"
        health_status.append(f"Policy Loss: {latest_policy_loss:.4f} [{policy_status}]")
        
        # Overall assessment
        if any("CRITICAL" in s for s in health_status):
            overall = "CRITICAL\nCheck Physics / Actuator"
        elif any("WARNING" in s for s in health_status):
            overall = "WARNING\nMonitor Carefully!"
        else:
            overall = "HEALTHY\nGood Training Progress"
        
        avg_info = " (avg)" if use_avg else ""
        text = f"Epoch: {latest_epoch}{avg_info}\n"
        text += f"Overall: {overall}\n"
        text += "-" * 40 + "\n"
        text += "\n".join(health_status)
        
        self.health_text.set_text(text)
    
    def run(self):
        """Run the dashboard"""
        print(f"Monitoring: {self.csv_path} (Press Ctrl+C to exit)")
        
        # Optimized: increased interval(ms) and enabled blitting for smoother performance
        ani = FuncAnimation(self.fig, self.update_plots, interval=2000, blit=False, cache_frame_data=False)
        plt.show()


def resolve_csv_path(path):
    """
    Resolve CSV path - handles both direct file path and folder path
    Args:
        path: Path to CSV file OR folder containing training_metrics.csv
    Returns:
        Resolved path to training_metrics.csv
    """
    if os.path.isfile(path):
        # Direct file path provided
        if path.endswith('.csv'):
            return path
        else:
            print(f"Warning: {path} is a file but not a .csv file")
            return path
    elif os.path.isdir(path):
        # Folder path provided - append training_metrics.csv
        csv_path = os.path.join(path, "training_metrics.csv")
        if os.path.exists(csv_path):
            return csv_path
        else:
            print(f"Warning: training_metrics.csv not found in {path}")
            return csv_path
    else:
        # Path doesn't exist yet - assume it will be created
        if path.endswith('.csv'):
            return path
        else:
            return os.path.join(path, "training_metrics.csv")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Training Dashboard for CompositeMotion')
    parser.add_argument('path', type=str, help='Path to training_metrics.csv OR folder containing it')
    parser.add_argument('-w', type=int, default=None, help='Number of epochs to display (None for all)')
    parser.add_argument('-fpc', type=int, default=30,
                        help='Number of Frames-Per-Cycle for lifetime benchmarks (default: 30)')
    parser.add_argument('-cpe', type=int, default=1,
                        help='Number of Cycles-Per-Episode for lifetime benchmarks (default: 1)')
    args = parser.parse_args()
    
    # Resolve CSV path (handles both file and folder paths)
    csv_path = resolve_csv_path(args.path)
    
    dashboard = TrainingDashboard(csv_path, window_size=args.w,
            frames_per_cycle=args.fpc, cycles_per_episode=args.cpe)
    dashboard.run()


if __name__ == '__main__':
    main()