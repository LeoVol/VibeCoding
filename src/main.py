#!/usr/bin/env python3
"""
Lines (Color Lines) Game
========================
A classic puzzle game where you align 5 or more balls of the same color
to remove them and score points.

Requirements:
    pip install pygame

Run:
    python lines_game.py

Controls:
    - Click a ball to select it
    - Click an empty cell to move the selected ball (if a path exists)
    - Press 'R' to restart the game
    - Press 'Q' to quit

Author: Claude (Anthropic)
"""

import pygame
import random
import json
import os
from datetime import datetime
from collections import deque
from typing import Optional, List, Tuple, Set, Dict
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid settings
GRID_SIZE = 9
CELL_SIZE = 60
GRID_PADDING = 40

# Colors for balls (RGB)
BALL_COLORS = [
    (220, 50, 50),    # Red
    (50, 180, 50),    # Green
    (50, 100, 220),   # Blue
    (220, 180, 50),   # Yellow
    (180, 50, 180),   # Purple
    (50, 200, 200),   # Cyan
    (255, 130, 50),   # Orange
]

# UI Colors
BG_COLOR = (30, 32, 40)
GRID_COLOR = (60, 65, 80)
GRID_LINE_COLOR = (80, 85, 100)
CELL_HIGHLIGHT_COLOR = (100, 110, 140)
SELECTED_HIGHLIGHT = (255, 220, 100)
PATH_HIGHLIGHT = (80, 200, 120, 100)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR = (100, 180, 255)
BUTTON_COLOR = (70, 75, 95)
BUTTON_HOVER_COLOR = (90, 95, 115)

# Game settings
BALLS_PER_TURN = 3
MIN_LINE_LENGTH = 5
INITIAL_BALLS = 5

# Animation settings
ANIMATION_SPEED = 15  # cells per second
REMOVAL_ANIMATION_DURATION = 300  # milliseconds
SPAWN_ANIMATION_DURATION = 200  # milliseconds

# High score file
HIGH_SCORE_FILE = "highscores.json"
MAX_HIGH_SCORES = 10

# Window dimensions
SIDEBAR_WIDTH = 250
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + GRID_PADDING * 2 + SIDEBAR_WIDTH
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + GRID_PADDING * 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Ball:
    """Represents a ball on the game board."""
    color_index: int
    row: int
    col: int
    scale: float = 1.0  # For animation
    alpha: float = 1.0  # For fade animation


@dataclass
class HighScore:
    """Represents a high score entry."""
    name: str
    score: int
    date: str


@dataclass
class Animation:
    """Base animation class."""
    start_time: int
    duration: int
    
    def progress(self, current_time: int) -> float:
        """Returns animation progress from 0.0 to 1.0."""
        elapsed = current_time - self.start_time
        return min(1.0, elapsed / self.duration)
    
    def is_complete(self, current_time: int) -> bool:
        return self.progress(current_time) >= 1.0


@dataclass
class MoveAnimation(Animation):
    """Animation for ball movement."""
    ball: Ball
    path: List[Tuple[int, int]]
    current_pos: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class RemovalAnimation(Animation):
    """Animation for ball removal."""
    positions: List[Tuple[int, int]]
    color_index: int


@dataclass
class SpawnAnimation(Animation):
    """Animation for ball spawning."""
    ball: Ball


class GameState(Enum):
    """Game state enumeration."""
    PLAYING = "playing"
    ANIMATING = "animating"
    GAME_OVER = "game_over"
    ENTERING_NAME = "entering_name"


# =============================================================================
# PATHFINDER
# =============================================================================

class PathFinder:
    """Handles pathfinding using Breadth-First Search (BFS)."""
    
    @staticmethod
    def find_path(
        grid: List[List[Optional[Ball]]],
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to end using BFS.
        Movement is only allowed in 4 directions (no diagonals).
        Returns the path as a list of (row, col) tuples, or None if no path exists.
        """
        if start == end:
            return [start]
        
        rows = len(grid)
        cols = len(grid[0])
        
        # BFS setup
        queue = deque([(start, [start])])
        visited = {start}
        
        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            (row, col), path = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if not (0 <= new_row < rows and 0 <= new_col < cols):
                    continue
                
                # Check if already visited
                if (new_row, new_col) in visited:
                    continue
                
                # Check if cell is empty (or is the destination)
                if grid[new_row][new_col] is not None and (new_row, new_col) != end:
                    continue
                
                # Found the destination
                if (new_row, new_col) == end:
                    return path + [(new_row, new_col)]
                
                # Continue searching
                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))
        
        return None  # No path found


# =============================================================================
# LINE DETECTOR
# =============================================================================

class LineDetector:
    """Handles detection of lines of same-colored balls."""
    
    @staticmethod
    def find_lines(
        grid: List[List[Optional[Ball]]],
        min_length: int = MIN_LINE_LENGTH
    ) -> Set[Tuple[int, int]]:
        """
        Find all balls that are part of lines of min_length or more.
        Checks horizontal, vertical, and both diagonal directions.
        Returns a set of (row, col) positions to remove.
        """
        rows = len(grid)
        cols = len(grid[0])
        to_remove: Set[Tuple[int, int]] = set()
        
        # Direction vectors: horizontal, vertical, diagonal down-right, diagonal down-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(rows):
            for col in range(cols):
                ball = grid[row][col]
                if ball is None:
                    continue
                
                color = ball.color_index
                
                for dr, dc in directions:
                    # Count consecutive balls of same color
                    line_positions = [(row, col)]
                    r, c = row + dr, col + dc
                    
                    while (0 <= r < rows and 0 <= c < cols and
                           grid[r][c] is not None and
                           grid[r][c].color_index == color):
                        line_positions.append((r, c))
                        r += dr
                        c += dc
                    
                    # If line is long enough, mark for removal
                    if len(line_positions) >= min_length:
                        to_remove.update(line_positions)
        
        return to_remove
    
    @staticmethod
    def find_lines_from_position(
        grid: List[List[Optional[Ball]]],
        row: int,
        col: int,
        min_length: int = MIN_LINE_LENGTH
    ) -> Set[Tuple[int, int]]:
        """
        Find lines that include the specified position.
        More efficient than checking the entire grid.
        """
        if grid[row][col] is None:
            return set()
        
        rows = len(grid)
        cols = len(grid[0])
        color = grid[row][col].color_index
        to_remove: Set[Tuple[int, int]] = set()
        
        # Direction pairs for bidirectional checking
        direction_pairs = [
            ((0, -1), (0, 1)),   # Horizontal
            ((-1, 0), (1, 0)),   # Vertical
            ((-1, -1), (1, 1)),  # Diagonal \
            ((-1, 1), (1, -1)),  # Diagonal /
        ]
        
        for (dr1, dc1), (dr2, dc2) in direction_pairs:
            line_positions = [(row, col)]
            
            # Check in first direction
            r, c = row + dr1, col + dc1
            while (0 <= r < rows and 0 <= c < cols and
                   grid[r][c] is not None and
                   grid[r][c].color_index == color):
                line_positions.append((r, c))
                r += dr1
                c += dc1
            
            # Check in second direction
            r, c = row + dr2, col + dc2
            while (0 <= r < rows and 0 <= c < cols and
                   grid[r][c] is not None and
                   grid[r][c].color_index == color):
                line_positions.append((r, c))
                r += dr2
                c += dc2
            
            if len(line_positions) >= min_length:
                to_remove.update(line_positions)
        
        return to_remove


# =============================================================================
# SCORE MANAGER
# =============================================================================

class ScoreManager:
    """Manages high scores with file persistence."""
    
    def __init__(self, filename: str = HIGH_SCORE_FILE):
        self.filename = filename
        self.scores: List[HighScore] = []
        self.load_scores()
    
    def load_scores(self) -> None:
        """Load high scores from file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.scores = [
                        HighScore(
                            name=entry['name'],
                            score=entry['score'],
                            date=entry['date']
                        )
                        for entry in data
                    ]
            except (json.JSONDecodeError, KeyError, IOError):
                self.scores = []
        else:
            self.scores = []
    
    def save_scores(self) -> None:
        """Save high scores to file."""
        data = [
            {
                'name': score.name,
                'score': score.score,
                'date': score.date
            }
            for score in self.scores
        ]
        try:
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't write
    
    def is_high_score(self, score: int) -> bool:
        """Check if the score qualifies for the top 10."""
        if len(self.scores) < MAX_HIGH_SCORES:
            return True
        return score > self.scores[-1].score
    
    def add_score(self, name: str, score: int) -> int:
        """
        Add a new score to the list.
        Returns the rank (1-based index) of the new score.
        """
        new_entry = HighScore(
            name=name,
            score=score,
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        self.scores.append(new_entry)
        self.scores.sort(key=lambda x: x.score, reverse=True)
        self.scores = self.scores[:MAX_HIGH_SCORES]
        self.save_scores()
        
        # Find rank
        for i, entry in enumerate(self.scores):
            if entry.name == name and entry.score == score and entry.date == new_entry.date:
                return i + 1
        return -1
    
    def get_top_scores(self) -> List[HighScore]:
        """Get the list of top scores."""
        return self.scores.copy()


# =============================================================================
# GAME BOARD
# =============================================================================

class GameBoard:
    """Manages the game board state and logic."""
    
    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.grid: List[List[Optional[Ball]]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        self.score = 0
        self.selected_ball: Optional[Tuple[int, int]] = None
        self.next_colors: List[int] = []
        self.path_finder = PathFinder()
        self.line_detector = LineDetector()
        
        # Generate initial next colors and spawn initial balls
        self._generate_next_colors()
        self._spawn_balls(INITIAL_BALLS)
        self._generate_next_colors()
    
    def _generate_next_colors(self) -> None:
        """Generate the colors for the next balls to spawn."""
        self.next_colors = [
            random.randint(0, len(BALL_COLORS) - 1)
            for _ in range(BALLS_PER_TURN)
        ]
    
    def _get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get all empty cells on the board."""
        empty = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] is None:
                    empty.append((row, col))
        return empty
    
    def _spawn_balls(self, count: int = BALLS_PER_TURN) -> List[Ball]:
        """Spawn new balls in random empty cells. Returns the spawned balls."""
        empty_cells = self._get_empty_cells()
        spawned = []
        
        if not empty_cells:
            return spawned
        
        # Use next_colors if available, otherwise random
        colors_to_use = self.next_colors[:count] if self.next_colors else [
            random.randint(0, len(BALL_COLORS) - 1) for _ in range(count)
        ]
        
        cells_to_use = random.sample(empty_cells, min(count, len(empty_cells)))
        
        for i, (row, col) in enumerate(cells_to_use):
            color = colors_to_use[i] if i < len(colors_to_use) else random.randint(0, len(BALL_COLORS) - 1)
            ball = Ball(color_index=color, row=row, col=col, scale=0.0)
            self.grid[row][col] = ball
            spawned.append(ball)
        
        return spawned
    
    def spawn_turn_balls(self) -> Tuple[List[Ball], Set[Tuple[int, int]]]:
        """
        Spawn balls for the turn and check for lines.
        Returns (spawned_balls, positions_to_remove).
        """
        spawned = self._spawn_balls()
        self._generate_next_colors()
        
        # Check if any spawned balls create lines
        all_to_remove: Set[Tuple[int, int]] = set()
        for ball in spawned:
            lines = self.line_detector.find_lines_from_position(
                self.grid, ball.row, ball.col
            )
            all_to_remove.update(lines)
        
        return spawned, all_to_remove
    
    def select_cell(self, row: int, col: int) -> Optional[str]:
        """
        Handle cell selection.
        Returns: 'selected', 'moved', 'invalid', or None.
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return None
        
        cell = self.grid[row][col]
        
        if cell is not None:
            # Selecting a ball
            self.selected_ball = (row, col)
            return 'selected'
        elif self.selected_ball is not None:
            # Trying to move selected ball
            return 'move_attempt'
        
        return None
    
    def try_move(self, target_row: int, target_col: int) -> Optional[List[Tuple[int, int]]]:
        """
        Try to move the selected ball to the target cell.
        Returns the path if successful, None otherwise.
        """
        if self.selected_ball is None:
            return None
        
        src_row, src_col = self.selected_ball
        
        # Can't move to occupied cell
        if self.grid[target_row][target_col] is not None:
            return None
        
        # Find path
        path = self.path_finder.find_path(
            self.grid, (src_row, src_col), (target_row, target_col)
        )
        
        return path
    
    def execute_move(self, path: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Execute a move along the given path.
        Returns positions to remove (if a line was formed).
        """
        if not path or len(path) < 2:
            return set()
        
        start = path[0]
        end = path[-1]
        
        ball = self.grid[start[0]][start[1]]
        if ball is None:
            return set()
        
        # Move ball
        self.grid[start[0]][start[1]] = None
        ball.row, ball.col = end
        self.grid[end[0]][end[1]] = ball
        
        # Clear selection
        self.selected_ball = None
        
        # Check for lines at the destination
        lines = self.line_detector.find_lines_from_position(self.grid, end[0], end[1])
        
        return lines
    
    def remove_balls(self, positions: Set[Tuple[int, int]]) -> int:
        """Remove balls at the given positions. Returns points earned."""
        if not positions:
            return 0
        
        for row, col in positions:
            self.grid[row][col] = None
        
        # Score calculation: base points + bonus for larger lines
        count = len(positions)
        points = count * 10 + max(0, count - MIN_LINE_LENGTH) * 5
        self.score += points
        
        return points
    
    def is_game_over(self) -> bool:
        """Check if the game is over (board full)."""
        return len(self._get_empty_cells()) == 0
    
    def has_valid_moves(self) -> bool:
        """Check if any valid moves exist."""
        # Find all balls
        balls = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] is not None:
                    balls.append((row, col))
        
        # Find all empty cells
        empty = self._get_empty_cells()
        
        if not empty:
            return False
        
        # Check if any ball can reach any empty cell
        for ball_pos in balls:
            for empty_pos in empty:
                if self.path_finder.find_path(self.grid, ball_pos, empty_pos):
                    return True
        
        return False
    
    def reset(self) -> None:
        """Reset the game board."""
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        self.selected_ball = None
        self.next_colors = []
        self._generate_next_colors()
        self._spawn_balls(INITIAL_BALLS)
        self._generate_next_colors()


# =============================================================================
# RENDERER
# =============================================================================

class Renderer:
    """Handles all game rendering."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.fonts = self._load_fonts()
        self.clock = pygame.time.Clock()
    
    def _load_fonts(self) -> Dict[str, pygame.font.Font]:
        """Load fonts for rendering."""
        pygame.font.init()
        return {
            'title': pygame.font.Font(None, 48),
            'large': pygame.font.Font(None, 36),
            'medium': pygame.font.Font(None, 28),
            'small': pygame.font.Font(None, 22),
        }
    
    def _cell_to_pixel(self, row: int, col: int) -> Tuple[int, int]:
        """Convert grid coordinates to pixel coordinates."""
        x = GRID_PADDING + col * CELL_SIZE
        y = GRID_PADDING + row * CELL_SIZE
        return x, y
    
    def _draw_ball(
        self,
        row: int,
        col: int,
        color_index: int,
        scale: float = 1.0,
        alpha: float = 1.0,
        highlight: bool = False
    ) -> None:
        """Draw a ball at the specified grid position."""
        x, y = self._cell_to_pixel(row, col)
        center_x = x + CELL_SIZE // 2
        center_y = y + CELL_SIZE // 2
        
        base_radius = CELL_SIZE // 2 - 8
        radius = int(base_radius * scale)
        
        if radius <= 0:
            return
        
        color = BALL_COLORS[color_index]
        
        # Apply alpha by drawing to a temporary surface
        if alpha < 1.0:
            surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            surf_center = radius + 2
            
            # Draw gradient ball on surface
            for i in range(radius, 0, -1):
                ratio = i / radius
                r = min(255, int(color[0] + (255 - color[0]) * (1 - ratio) * 0.5))
                g = min(255, int(color[1] + (255 - color[1]) * (1 - ratio) * 0.5))
                b = min(255, int(color[2] + (255 - color[2]) * (1 - ratio) * 0.5))
                a = int(255 * alpha)
                pygame.draw.circle(surf, (r, g, b, a), (surf_center, surf_center), i)
            
            self.screen.blit(surf, (center_x - radius - 2, center_y - radius - 2))
        else:
            # Draw gradient ball (simple 3D effect)
            for i in range(radius, 0, -1):
                ratio = i / radius
                r = min(255, int(color[0] + (255 - color[0]) * (1 - ratio) * 0.5))
                g = min(255, int(color[1] + (255 - color[1]) * (1 - ratio) * 0.5))
                b = min(255, int(color[2] + (255 - color[2]) * (1 - ratio) * 0.5))
                pygame.draw.circle(self.screen, (r, g, b), (center_x, center_y), i)
            
            # Add highlight
            highlight_offset = radius // 3
            highlight_radius = radius // 4
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (center_x - highlight_offset, center_y - highlight_offset),
                highlight_radius
            )
        
        # Draw selection highlight
        if highlight:
            pygame.draw.circle(
                self.screen,
                SELECTED_HIGHLIGHT,
                (center_x, center_y),
                radius + 4,
                3
            )
    
    def draw_grid(self, board: GameBoard) -> None:
        """Draw the game grid."""
        # Draw grid background
        grid_rect = pygame.Rect(
            GRID_PADDING - 5,
            GRID_PADDING - 5,
            GRID_SIZE * CELL_SIZE + 10,
            GRID_SIZE * CELL_SIZE + 10
        )
        pygame.draw.rect(self.screen, GRID_COLOR, grid_rect, border_radius=8)
        
        # Draw cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x, y = self._cell_to_pixel(row, col)
                cell_rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                
                # Highlight selected cell
                if board.selected_ball == (row, col):
                    pygame.draw.rect(
                        self.screen,
                        CELL_HIGHLIGHT_COLOR,
                        cell_rect,
                        border_radius=4
                    )
                
                # Draw cell border
                pygame.draw.rect(
                    self.screen,
                    GRID_LINE_COLOR,
                    cell_rect,
                    1,
                    border_radius=4
                )
    
    def draw_balls(
        self,
        board: GameBoard,
        exclude_positions: Set[Tuple[int, int]] = None,
        removal_animation: Optional[RemovalAnimation] = None
    ) -> None:
        """Draw all balls on the board."""
        exclude = exclude_positions or set()
        current_time = pygame.time.get_ticks()
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row, col) in exclude:
                    continue
                
                ball = board.grid[row][col]
                if ball is not None:
                    is_selected = board.selected_ball == (row, col)
                    
                    # Check if this ball is being removed
                    if removal_animation and (row, col) in removal_animation.positions:
                        progress = removal_animation.progress(current_time)
                        scale = 1.0 - progress
                        alpha = 1.0 - progress
                        self._draw_ball(
                            row, col, ball.color_index,
                            scale=scale, alpha=alpha, highlight=is_selected
                        )
                    else:
                        self._draw_ball(
                            row, col, ball.color_index,
                            scale=ball.scale, highlight=is_selected
                        )
    
    def draw_moving_ball(
        self,
        animation: MoveAnimation,
        current_time: int
    ) -> None:
        """Draw a ball during movement animation."""
        progress = animation.progress(current_time)
        path = animation.path
        
        # Calculate position along path
        total_segments = len(path) - 1
        if total_segments <= 0:
            return
        
        segment_progress = progress * total_segments
        segment_index = min(int(segment_progress), total_segments - 1)
        segment_t = segment_progress - segment_index
        
        start = path[segment_index]
        end = path[min(segment_index + 1, len(path) - 1)]
        
        # Interpolate position
        current_row = start[0] + (end[0] - start[0]) * segment_t
        current_col = start[1] + (end[1] - start[1]) * segment_t
        
        # Draw the ball
        x = GRID_PADDING + current_col * CELL_SIZE + CELL_SIZE // 2
        y = GRID_PADDING + current_row * CELL_SIZE + CELL_SIZE // 2
        
        color = BALL_COLORS[animation.ball.color_index]
        radius = CELL_SIZE // 2 - 8
        
        # Draw gradient ball
        for i in range(radius, 0, -1):
            ratio = i / radius
            r = min(255, int(color[0] + (255 - color[0]) * (1 - ratio) * 0.5))
            g = min(255, int(color[1] + (255 - color[1]) * (1 - ratio) * 0.5))
            b = min(255, int(color[2] + (255 - color[2]) * (1 - ratio) * 0.5))
            pygame.draw.circle(self.screen, (r, g, b), (int(x), int(y)), i)
        
        # Add highlight
        highlight_offset = radius // 3
        highlight_radius = radius // 4
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(x) - highlight_offset, int(y) - highlight_offset),
            highlight_radius
        )
    
    def draw_sidebar(
        self,
        board: GameBoard,
        game_state: GameState,
        score_manager: ScoreManager
    ) -> None:
        """Draw the sidebar with score, next balls, and info."""
        sidebar_x = GRID_PADDING * 2 + GRID_SIZE * CELL_SIZE
        
        # Title
        title = self.fonts['title'].render("LINES", True, ACCENT_COLOR)
        self.screen.blit(title, (sidebar_x + 10, 20))
        
        # Score
        score_label = self.fonts['medium'].render("Score", True, TEXT_COLOR)
        self.screen.blit(score_label, (sidebar_x + 10, 80))
        
        score_value = self.fonts['large'].render(str(board.score), True, ACCENT_COLOR)
        self.screen.blit(score_value, (sidebar_x + 10, 105))
        
        # Next balls preview
        next_label = self.fonts['medium'].render("Next:", True, TEXT_COLOR)
        self.screen.blit(next_label, (sidebar_x + 10, 160))
        
        for i, color_idx in enumerate(board.next_colors):
            color = BALL_COLORS[color_idx]
            x = sidebar_x + 20 + i * 45
            y = 195
            radius = 15
            
            # Draw mini ball
            for j in range(radius, 0, -1):
                ratio = j / radius
                r = min(255, int(color[0] + (255 - color[0]) * (1 - ratio) * 0.5))
                g = min(255, int(color[1] + (255 - color[1]) * (1 - ratio) * 0.5))
                b = min(255, int(color[2] + (255 - color[2]) * (1 - ratio) * 0.5))
                pygame.draw.circle(self.screen, (r, g, b), (x, y), j)
        
        # High scores
        scores_label = self.fonts['medium'].render("High Scores", True, TEXT_COLOR)
        self.screen.blit(scores_label, (sidebar_x + 10, 240))
        
        top_scores = score_manager.get_top_scores()[:5]  # Show top 5 in sidebar
        for i, hs in enumerate(top_scores):
            y_pos = 270 + i * 25
            name_text = self.fonts['small'].render(
                f"{i+1}. {hs.name[:8]}", True, TEXT_COLOR
            )
            score_text = self.fonts['small'].render(
                str(hs.score), True, ACCENT_COLOR
            )
            self.screen.blit(name_text, (sidebar_x + 10, y_pos))
            self.screen.blit(score_text, (sidebar_x + 150, y_pos))
        
        # Controls
        controls_y = 420
        controls = [
            "Controls:",
            "Click: Select/Move",
            "R: Restart",
            "Q: Quit"
        ]
        for i, text in enumerate(controls):
            color = TEXT_COLOR if i > 0 else ACCENT_COLOR
            label = self.fonts['small'].render(text, True, color)
            self.screen.blit(label, (sidebar_x + 10, controls_y + i * 22))
    
    def draw_game_over(
        self,
        score: int,
        score_manager: ScoreManager,
        input_text: str = "",
        show_input: bool = False
    ) -> pygame.Rect:
        """Draw game over screen. Returns the restart button rect."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game over panel
        panel_width = 400
        panel_height = 450 if show_input else 400
        panel_x = (WINDOW_WIDTH - panel_width) // 2
        panel_y = (WINDOW_HEIGHT - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, GRID_COLOR, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, ACCENT_COLOR, panel_rect, 2, border_radius=12)
        
        # Game Over title
        title = self.fonts['title'].render("GAME OVER", True, ACCENT_COLOR)
        title_rect = title.get_rect(centerx=WINDOW_WIDTH // 2, y=panel_y + 20)
        self.screen.blit(title, title_rect)
        
        # Final score
        score_label = self.fonts['large'].render(
            f"Final Score: {score}", True, TEXT_COLOR
        )
        score_rect = score_label.get_rect(centerx=WINDOW_WIDTH // 2, y=panel_y + 75)
        self.screen.blit(score_label, score_rect)
        
        # Name input if needed
        if show_input:
            prompt = self.fonts['medium'].render(
                "New High Score! Enter name:", True, ACCENT_COLOR
            )
            prompt_rect = prompt.get_rect(centerx=WINDOW_WIDTH // 2, y=panel_y + 115)
            self.screen.blit(prompt, prompt_rect)
            
            # Input box
            input_rect = pygame.Rect(panel_x + 50, panel_y + 145, panel_width - 100, 35)
            pygame.draw.rect(self.screen, BG_COLOR, input_rect, border_radius=4)
            pygame.draw.rect(self.screen, ACCENT_COLOR, input_rect, 2, border_radius=4)
            
            input_surface = self.fonts['medium'].render(input_text + "_", True, TEXT_COLOR)
            self.screen.blit(input_surface, (input_rect.x + 10, input_rect.y + 7))
            
            top_scores_y = panel_y + 195
        else:
            top_scores_y = panel_y + 115
        
        # Top scores
        scores_title = self.fonts['medium'].render("Top 10 Scores", True, TEXT_COLOR)
        scores_title_rect = scores_title.get_rect(centerx=WINDOW_WIDTH // 2, y=top_scores_y)
        self.screen.blit(scores_title, scores_title_rect)
        
        top_scores = score_manager.get_top_scores()
        for i, hs in enumerate(top_scores):
            y_pos = top_scores_y + 30 + i * 20
            entry_text = f"{i+1}. {hs.name[:10]:10} {hs.score:>6}"
            entry = self.fonts['small'].render(entry_text, True, TEXT_COLOR)
            entry_rect = entry.get_rect(centerx=WINDOW_WIDTH // 2, y=y_pos)
            self.screen.blit(entry, entry_rect)
        
        # Restart button
        button_width = 150
        button_height = 40
        button_x = (WINDOW_WIDTH - button_width) // 2
        button_y = panel_y + panel_height - 55
        
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        # Check hover
        mouse_pos = pygame.mouse.get_pos()
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, BUTTON_HOVER_COLOR, button_rect, border_radius=6)
        else:
            pygame.draw.rect(self.screen, BUTTON_COLOR, button_rect, border_radius=6)
        
        pygame.draw.rect(self.screen, ACCENT_COLOR, button_rect, 2, border_radius=6)
        
        button_text = self.fonts['medium'].render("Restart", True, TEXT_COLOR)
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        return button_rect
    
    def clear(self) -> None:
        """Clear the screen."""
        self.screen.fill(BG_COLOR)


# =============================================================================
# INPUT HANDLER
# =============================================================================

class InputHandler:
    """Handles user input."""
    
    @staticmethod
    def get_cell_from_mouse(pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to grid cell. Returns None if outside grid."""
        x, y = pos
        
        # Check if within grid bounds
        grid_x = x - GRID_PADDING
        grid_y = y - GRID_PADDING
        
        if grid_x < 0 or grid_y < 0:
            return None
        
        col = grid_x // CELL_SIZE
        row = grid_y // CELL_SIZE
        
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return (row, col)
        
        return None


# =============================================================================
# GAME CONTROLLER
# =============================================================================

class Game:
    """Main game controller."""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Lines - Color Lines Game")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.board = GameBoard()
        self.renderer = Renderer(self.screen)
        self.score_manager = ScoreManager()
        self.input_handler = InputHandler()
        
        self.state = GameState.PLAYING
        self.running = True
        
        # Animation state
        self.move_animation: Optional[MoveAnimation] = None
        self.removal_animation: Optional[RemovalAnimation] = None
        self.spawn_animations: List[SpawnAnimation] = []
        self.pending_removals: Set[Tuple[int, int]] = set()
        self.pending_spawns = False
        
        # Name input
        self.input_text = ""
        self.restart_button_rect: Optional[pygame.Rect] = None
    
    def handle_events(self) -> None:
        """Process all pending events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if self.state == GameState.ENTERING_NAME:
                    self._handle_name_input(event)
                elif event.key == pygame.K_r:
                    self._restart_game()
                elif event.key == pygame.K_q:
                    self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_click(event.pos)
    
    def _handle_name_input(self, event: pygame.event.Event) -> None:
        """Handle keyboard input for name entry."""
        if event.key == pygame.K_RETURN:
            if self.input_text.strip():
                self.score_manager.add_score(
                    self.input_text.strip(),
                    self.board.score
                )
                self.state = GameState.GAME_OVER
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif len(self.input_text) < 15 and event.unicode.isprintable():
            self.input_text += event.unicode
    
    def _handle_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse click."""
        # Check for restart button click in game over state
        if self.state in (GameState.GAME_OVER, GameState.ENTERING_NAME):
            if self.restart_button_rect and self.restart_button_rect.collidepoint(pos):
                self._restart_game()
            return
        
        # Ignore clicks during animation
        if self.state == GameState.ANIMATING:
            return
        
        cell = self.input_handler.get_cell_from_mouse(pos)
        if cell is None:
            return
        
        row, col = cell
        
        if self.board.grid[row][col] is not None:
            # Clicking on a ball - select it
            self.board.selected_ball = (row, col)
        elif self.board.selected_ball is not None:
            # Clicking on empty cell with ball selected - try to move
            path = self.board.try_move(row, col)
            if path:
                self._start_move_animation(path)
    
    def _start_move_animation(self, path: List[Tuple[int, int]]) -> None:
        """Start ball movement animation."""
        start = path[0]
        ball = self.board.grid[start[0]][start[1]]
        
        if ball is None:
            return
        
        # Calculate animation duration based on path length
        duration = int((len(path) - 1) * (1000 / ANIMATION_SPEED))
        
        self.move_animation = MoveAnimation(
            start_time=pygame.time.get_ticks(),
            duration=max(duration, 100),
            ball=ball,
            path=path
        )
        self.state = GameState.ANIMATING
        
        # Remove ball from grid temporarily (it's being animated)
        self.board.grid[start[0]][start[1]] = None
    
    def _complete_move(self) -> None:
        """Complete the move after animation."""
        if self.move_animation is None:
            return
        
        path = self.move_animation.path
        ball = self.move_animation.ball
        end = path[-1]
        
        # Place ball at destination
        ball.row, ball.col = end
        self.board.grid[end[0]][end[1]] = ball
        self.board.selected_ball = None
        
        # Check for lines
        lines = LineDetector.find_lines_from_position(
            self.board.grid, end[0], end[1]
        )
        
        self.move_animation = None
        
        if lines:
            # Start removal animation
            color_idx = ball.color_index
            self._start_removal_animation(lines, color_idx)
        else:
            # Spawn new balls
            self.pending_spawns = True
    
    def _start_removal_animation(
        self,
        positions: Set[Tuple[int, int]],
        color_index: int
    ) -> None:
        """Start ball removal animation."""
        self.removal_animation = RemovalAnimation(
            start_time=pygame.time.get_ticks(),
            duration=REMOVAL_ANIMATION_DURATION,
            positions=list(positions),
            color_index=color_index
        )
        self.pending_removals = positions
    
    def _complete_removal(self) -> None:
        """Complete ball removal after animation."""
        if not self.pending_removals:
            return
        
        self.board.remove_balls(self.pending_removals)
        self.pending_removals = set()
        self.removal_animation = None
        self.state = GameState.PLAYING
    
    def _spawn_new_balls(self) -> None:
        """Spawn new balls and handle any resulting lines."""
        spawned, lines_from_spawn = self.board.spawn_turn_balls()
        
        # Start spawn animations
        current_time = pygame.time.get_ticks()
        for ball in spawned:
            ball.scale = 0.0
            self.spawn_animations.append(SpawnAnimation(
                start_time=current_time,
                duration=SPAWN_ANIMATION_DURATION,
                ball=ball
            ))
        
        if lines_from_spawn:
            # Will handle removal after spawn animation
            self.pending_removals = lines_from_spawn
        
        self.pending_spawns = False
        
        # Check for game over
        if self.board.is_game_over():
            self._trigger_game_over()
    
    def _trigger_game_over(self) -> None:
        """Handle game over."""
        if self.score_manager.is_high_score(self.board.score):
            self.state = GameState.ENTERING_NAME
            self.input_text = ""
        else:
            self.state = GameState.GAME_OVER
    
    def _restart_game(self) -> None:
        """Restart the game."""
        self.board.reset()
        self.state = GameState.PLAYING
        self.move_animation = None
        self.removal_animation = None
        self.spawn_animations = []
        self.pending_removals = set()
        self.pending_spawns = False
        self.input_text = ""
    
    def update(self) -> None:
        """Update game state and animations."""
        current_time = pygame.time.get_ticks()
        
        # Update move animation
        if self.move_animation and self.move_animation.is_complete(current_time):
            self._complete_move()
        
        # Update removal animation
        if self.removal_animation and self.removal_animation.is_complete(current_time):
            self._complete_removal()
        
        # Update spawn animations
        completed_spawns = []
        for anim in self.spawn_animations:
            if anim.is_complete(current_time):
                anim.ball.scale = 1.0
                completed_spawns.append(anim)
            else:
                progress = anim.progress(current_time)
                # Ease-out animation
                anim.ball.scale = 1.0 - (1.0 - progress) ** 2
        
        for anim in completed_spawns:
            self.spawn_animations.remove(anim)
        
        # If spawn animations complete and there are pending removals
        if not self.spawn_animations and self.pending_removals:
            color_idx = 0
            if self.pending_removals:
                pos = next(iter(self.pending_removals))
                ball = self.board.grid[pos[0]][pos[1]]
                if ball:
                    color_idx = ball.color_index
            self._start_removal_animation(self.pending_removals, color_idx)
            self.pending_removals = set()
        
        # Handle pending spawns
        if self.pending_spawns and not self.move_animation and not self.removal_animation:
            self._spawn_new_balls()
        
        # Update state based on animations
        if (self.state == GameState.ANIMATING and
            not self.move_animation and
            not self.removal_animation and
            not self.spawn_animations):
            self.state = GameState.PLAYING
    
    def render(self) -> None:
        """Render the current game state."""
        self.renderer.clear()
        
        # Draw grid
        self.renderer.draw_grid(self.board)
        
        # Determine which balls to exclude from normal rendering
        exclude = set()
        if self.move_animation:
            exclude.add(self.move_animation.path[0])
        
        # Draw balls
        self.renderer.draw_balls(
            self.board,
            exclude_positions=exclude,
            removal_animation=self.removal_animation
        )
        
        # Draw moving ball
        if self.move_animation:
            self.renderer.draw_moving_ball(
                self.move_animation,
                pygame.time.get_ticks()
            )
        
        # Draw sidebar
        self.renderer.draw_sidebar(self.board, self.state, self.score_manager)
        
        # Draw game over screen
        if self.state in (GameState.GAME_OVER, GameState.ENTERING_NAME):
            self.restart_button_rect = self.renderer.draw_game_over(
                self.board.score,
                self.score_manager,
                self.input_text,
                self.state == GameState.ENTERING_NAME
            )
        
        pygame.display.flip()
    
    def run(self) -> None:
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("="*50)
    print("LINES (Color Lines) Game")
    print("="*50)
    print("\nControls:")
    print("  - Click a ball to select it")
    print("  - Click an empty cell to move (if path exists)")
    print("  - Press 'R' to restart")
    print("  - Press 'Q' to quit")
    print("\nGoal: Align 5+ balls of the same color to remove them!")
    print("="*50)
    
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
