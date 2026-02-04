#!/usr/bin/env python3
"""
Lines (Color Lines) Game
========================

A classic puzzle game where you remove balls by forming lines of 5 or more
balls of the same color.

Requirements:
    pip install pygame

Run:
    python lines_game.py

Controls:
    - Click on a ball to select it (highlighted with a ring)
    - Click on an empty cell to move the ball there
    - Form lines of 5+ same-colored balls to score points
    - Game ends when the board is full

Author: Claude (Anthropic)
"""

import pygame
import random
import json
import os
from datetime import datetime
from collections import deque
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONSTANTS
# =============================================================================

# Board configuration
BOARD_SIZE = 9
CELL_SIZE = 60
BOARD_PADDING = 40
MIN_LINE_LENGTH = 5
INITIAL_BALLS = 5
BALLS_PER_TURN = 3

# Window configuration
SIDEBAR_WIDTH = 250
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + BOARD_PADDING * 2 + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + BOARD_PADDING * 2 + 60  # Extra for title

# Colors (RGB)
COLORS = {
    'background': (40, 44, 52),
    'board_bg': (60, 64, 72),
    'grid_line': (80, 84, 92),
    'cell_hover': (70, 74, 82),
    'selection': (255, 215, 0),
    'text': (220, 220, 220),
    'text_dim': (150, 150, 150),
    'button': (70, 130, 180),
    'button_hover': (90, 150, 200),
    'game_over_bg': (0, 0, 0, 200),
    'path_highlight': (100, 255, 100, 100),
}

# Ball colors - vibrant and distinct
BALL_COLORS = [
    (255, 87, 87),    # Red
    (87, 255, 87),    # Green
    (87, 87, 255),    # Blue
    (255, 255, 87),   # Yellow
    (255, 87, 255),   # Magenta
    (87, 255, 255),   # Cyan
    (255, 165, 0),    # Orange
]

# Animation settings
ANIMATION_SPEED = 15  # cells per second for movement
BOUNCE_SPEED = 4      # bounce animation speed
BOUNCE_HEIGHT = 5     # pixels

# High scores
HIGHSCORE_FILE = "lines_highscores.json"
MAX_HIGHSCORES = 10


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    """Represents a position on the board."""
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
        return False
    
    def as_tuple(self) -> Tuple[int, int]:
        return (self.row, self.col)


@dataclass
class Ball:
    """Represents a ball on the board."""
    color_index: int
    
    @property
    def color(self) -> Tuple[int, int, int]:
        return BALL_COLORS[self.color_index]


@dataclass
class HighScore:
    """Represents a high score entry."""
    name: str
    score: int
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'score': self.score,
            'timestamp': self.timestamp
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'HighScore':
        return HighScore(
            name=data['name'],
            score=data['score'],
            timestamp=data['timestamp']
        )


class GameState(Enum):
    """Game state enumeration."""
    PLAYING = "playing"
    GAME_OVER = "game_over"
    ENTERING_NAME = "entering_name"


# =============================================================================
# PATHFINDER
# =============================================================================

class PathFinder:
    """Handles pathfinding using BFS algorithm."""
    
    # Movement directions: up, down, left, right (no diagonals)
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    @staticmethod
    def find_path(board: List[List[Optional[Ball]]], 
                  start: Position, 
                  end: Position) -> Optional[List[Position]]:
        """
        Find a path from start to end using BFS.
        
        Args:
            board: The game board (2D list)
            start: Starting position
            end: Target position
            
        Returns:
            List of positions forming the path, or None if no path exists.
        """
        if start == end:
            return [start]
        
        rows = len(board)
        cols = len(board[0])
        
        # BFS initialization
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for dr, dc in PathFinder.DIRECTIONS:
                new_row = current.row + dr
                new_col = current.col + dc
                new_pos = Position(new_row, new_col)
                
                # Check bounds
                if not (0 <= new_row < rows and 0 <= new_col < cols):
                    continue
                
                # Check if already visited
                if new_pos in visited:
                    continue
                
                # Check if we reached the destination
                if new_pos == end:
                    return path + [new_pos]
                
                # Check if cell is empty (can pass through)
                if board[new_row][new_col] is None:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return None  # No path found


# =============================================================================
# LINE DETECTOR
# =============================================================================

class LineDetector:
    """Handles detection of lines of same-colored balls."""
    
    # All 8 directions, but we only need 4 unique axes
    DIRECTIONS = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonal down-right
        (1, -1),  # Diagonal down-left
    ]
    
    @staticmethod
    def find_lines(board: List[List[Optional[Ball]]], 
                   min_length: int = MIN_LINE_LENGTH) -> Set[Position]:
        """
        Find all positions that are part of lines of min_length or more.
        
        Args:
            board: The game board
            min_length: Minimum line length to detect
            
        Returns:
            Set of positions to remove.
        """
        rows = len(board)
        cols = len(board[0])
        to_remove = set()
        
        for row in range(rows):
            for col in range(cols):
                ball = board[row][col]
                if ball is None:
                    continue
                
                # Check each direction from this position
                for dr, dc in LineDetector.DIRECTIONS:
                    line = LineDetector._get_line(board, row, col, dr, dc, ball.color_index)
                    if len(line) >= min_length:
                        to_remove.update(line)
        
        return to_remove
    
    @staticmethod
    def _get_line(board: List[List[Optional[Ball]]], 
                  start_row: int, 
                  start_col: int, 
                  dr: int, 
                  dc: int, 
                  color_index: int) -> Set[Position]:
        """
        Get all positions in a line of the same color.
        
        Extends in both directions from the starting point.
        """
        rows = len(board)
        cols = len(board[0])
        line = {Position(start_row, start_col)}
        
        # Extend in positive direction
        r, c = start_row + dr, start_col + dc
        while 0 <= r < rows and 0 <= c < cols:
            ball = board[r][c]
            if ball is None or ball.color_index != color_index:
                break
            line.add(Position(r, c))
            r += dr
            c += dc
        
        # Extend in negative direction
        r, c = start_row - dr, start_col - dc
        while 0 <= r < rows and 0 <= c < cols:
            ball = board[r][c]
            if ball is None or ball.color_index != color_index:
                break
            line.add(Position(r, c))
            r -= dr
            c -= dc
        
        return line


# =============================================================================
# SCORE MANAGER
# =============================================================================

class ScoreManager:
    """Handles score calculation and high score persistence."""
    
    def __init__(self, filepath: str = HIGHSCORE_FILE):
        self.filepath = filepath
        self.highscores: List[HighScore] = []
        self.load_highscores()
    
    def load_highscores(self) -> None:
        """Load high scores from file."""
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.highscores = [HighScore.from_dict(entry) for entry in data]
                    self.highscores.sort(key=lambda x: x.score, reverse=True)
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: Could not load high scores: {e}")
            self.highscores = []
    
    def save_highscores(self) -> None:
        """Save high scores to file."""
        try:
            with open(self.filepath, 'w') as f:
                data = [hs.to_dict() for hs in self.highscores]
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save high scores: {e}")
    
    def is_highscore(self, score: int) -> bool:
        """Check if a score qualifies for the top 10."""
        if score <= 0:
            return False
        if len(self.highscores) < MAX_HIGHSCORES:
            return True
        return score > self.highscores[-1].score
    
    def add_highscore(self, name: str, score: int) -> int:
        """
        Add a new high score.
        
        Returns:
            The rank (1-indexed) of the new score, or -1 if not added.
        """
        if not self.is_highscore(score):
            return -1
        
        entry = HighScore(
            name=name[:20],  # Limit name length
            score=score,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        
        self.highscores.append(entry)
        self.highscores.sort(key=lambda x: x.score, reverse=True)
        self.highscores = self.highscores[:MAX_HIGHSCORES]
        self.save_highscores()
        
        # Find rank
        for i, hs in enumerate(self.highscores):
            if hs.name == entry.name and hs.score == entry.score and hs.timestamp == entry.timestamp:
                return i + 1
        return -1
    
    @staticmethod
    def calculate_score(balls_removed: int) -> int:
        """
        Calculate score based on balls removed.
        
        Bonus for removing more than minimum required.
        """
        if balls_removed < MIN_LINE_LENGTH:
            return 0
        
        # Base score + bonus for extra balls
        base = balls_removed * 10
        bonus = (balls_removed - MIN_LINE_LENGTH) * 5
        return base + bonus


# =============================================================================
# GAME BOARD
# =============================================================================

class GameBoard:
    """Manages the game board state and logic."""
    
    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.board: List[List[Optional[Ball]]] = [[None] * size for _ in range(size)]
        self.score = 0
        self.selected: Optional[Position] = None
        self.next_balls: List[Tuple[int, int]] = []  # (color_index, preview_position)
        self.moving_ball: Optional[dict] = None  # Animation state
        self.removing_balls: Set[Position] = set()  # Balls being removed (animation)
        
        # Generate next balls preview
        self._generate_next_balls()
        
        # Place initial balls
        self._place_initial_balls()
    
    def _place_initial_balls(self) -> None:
        """Place initial balls on the board."""
        empty_cells = self._get_empty_cells()
        positions = random.sample(empty_cells, min(INITIAL_BALLS, len(empty_cells)))
        
        for pos in positions:
            color_index = random.randint(0, len(BALL_COLORS) - 1)
            self.board[pos.row][pos.col] = Ball(color_index)
    
    def _generate_next_balls(self) -> None:
        """Generate the next balls that will be spawned."""
        self.next_balls = [
            random.randint(0, len(BALL_COLORS) - 1)
            for _ in range(BALLS_PER_TURN)
        ]
    
    def _get_empty_cells(self) -> List[Position]:
        """Get all empty cells on the board."""
        empty = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] is None:
                    empty.append(Position(row, col))
        return empty
    
    def get_ball_count(self) -> int:
        """Get the number of balls on the board."""
        count = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] is not None:
                    count += 1
        return count
    
    def is_full(self) -> bool:
        """Check if the board is full."""
        return len(self._get_empty_cells()) == 0
    
    def select_cell(self, pos: Position) -> bool:
        """
        Handle cell selection.
        
        Returns True if the selection state changed.
        """
        if not (0 <= pos.row < self.size and 0 <= pos.col < self.size):
            return False
        
        ball = self.board[pos.row][pos.col]
        
        if ball is not None:
            # Select this ball
            self.selected = pos
            return True
        elif self.selected is not None:
            # Try to move selected ball to this position
            return self._try_move(self.selected, pos)
        
        return False
    
    def _try_move(self, start: Position, end: Position) -> bool:
        """
        Try to move a ball from start to end.
        
        Returns True if the move was successful.
        """
        # Find path
        path = PathFinder.find_path(self.board, start, end)
        
        if path is None:
            return False
        
        # Start move animation
        ball = self.board[start.row][start.col]
        self.board[start.row][start.col] = None
        self.board[end.row][end.col] = ball
        self.selected = None
        
        # Check for lines at the destination
        lines_to_remove = LineDetector.find_lines(self.board)
        
        if lines_to_remove:
            # Remove lines and update score
            self._remove_balls(lines_to_remove)
        else:
            # No lines formed, spawn new balls
            self._spawn_new_balls()
            
            # Check for lines after spawning
            lines_to_remove = LineDetector.find_lines(self.board)
            if lines_to_remove:
                self._remove_balls(lines_to_remove)
        
        return True
    
    def _remove_balls(self, positions: Set[Position]) -> None:
        """Remove balls at the given positions and update score."""
        for pos in positions:
            self.board[pos.row][pos.col] = None
        
        self.score += ScoreManager.calculate_score(len(positions))
    
    def _spawn_new_balls(self) -> None:
        """Spawn new balls on the board."""
        empty_cells = self._get_empty_cells()
        
        if not empty_cells:
            return
        
        # Spawn balls using the preview colors
        num_to_spawn = min(len(self.next_balls), len(empty_cells))
        positions = random.sample(empty_cells, num_to_spawn)
        
        for i, pos in enumerate(positions):
            color_index = self.next_balls[i]
            self.board[pos.row][pos.col] = Ball(color_index)
        
        # Generate next preview
        self._generate_next_balls()
    
    def has_valid_moves(self) -> bool:
        """Check if any valid moves exist."""
        # Find all balls
        ball_positions = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] is not None:
                    ball_positions.append(Position(row, col))
        
        # Find all empty cells
        empty_cells = self._get_empty_cells()
        
        # Check if any ball can reach any empty cell
        for ball_pos in ball_positions:
            for empty_pos in empty_cells:
                if PathFinder.find_path(self.board, ball_pos, empty_pos) is not None:
                    return True
        
        return False
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.is_full() or (not self.has_valid_moves() and self.get_ball_count() > 0)
    
    def reset(self) -> None:
        """Reset the board for a new game."""
        self.board = [[None] * self.size for _ in range(self.size)]
        self.score = 0
        self.selected = None
        self.next_balls = []
        self._generate_next_balls()
        self._place_initial_balls()


# =============================================================================
# RENDERER
# =============================================================================

class Renderer:
    """Handles all rendering operations."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.fonts = {}
        self._init_fonts()
        self.bounce_offset = 0
        self.bounce_direction = 1
        self.animation_tick = 0
    
    def _init_fonts(self) -> None:
        """Initialize fonts."""
        pygame.font.init()
        self.fonts['title'] = pygame.font.SysFont('Arial', 36, bold=True)
        self.fonts['large'] = pygame.font.SysFont('Arial', 28, bold=True)
        self.fonts['medium'] = pygame.font.SysFont('Arial', 22)
        self.fonts['small'] = pygame.font.SysFont('Arial', 18)
        self.fonts['tiny'] = pygame.font.SysFont('Arial', 14)
    
    def update_animations(self) -> None:
        """Update animation states."""
        self.animation_tick += 1
        
        # Bounce animation for selected ball
        if self.animation_tick % BOUNCE_SPEED == 0:
            self.bounce_offset += self.bounce_direction
            if abs(self.bounce_offset) >= BOUNCE_HEIGHT:
                self.bounce_direction *= -1
    
    def render(self, game: 'GameBoard', state: GameState, 
               score_manager: ScoreManager, hover_pos: Optional[Position] = None,
               input_text: str = "", new_rank: int = -1) -> None:
        """Render the entire game screen."""
        self.screen.fill(COLORS['background'])
        
        # Draw title
        self._draw_title()
        
        # Draw board
        self._draw_board(game, hover_pos)
        
        # Draw sidebar
        self._draw_sidebar(game, score_manager)
        
        # Draw game over overlay if needed
        if state == GameState.GAME_OVER:
            self._draw_game_over(game, score_manager)
        elif state == GameState.ENTERING_NAME:
            self._draw_name_input(game, input_text, new_rank)
    
    def _draw_title(self) -> None:
        """Draw the game title."""
        title = self.fonts['title'].render("LINES", True, COLORS['text'])
        title_rect = title.get_rect(
            centerx=BOARD_PADDING + (BOARD_SIZE * CELL_SIZE) // 2,
            top=10
        )
        self.screen.blit(title, title_rect)
    
    def _draw_board(self, game: 'GameBoard', hover_pos: Optional[Position]) -> None:
        """Draw the game board."""
        board_x = BOARD_PADDING
        board_y = BOARD_PADDING + 40
        board_width = BOARD_SIZE * CELL_SIZE
        board_height = BOARD_SIZE * CELL_SIZE
        
        # Draw board background
        pygame.draw.rect(
            self.screen, 
            COLORS['board_bg'],
            (board_x - 2, board_y - 2, board_width + 4, board_height + 4),
            border_radius=5
        )
        
        # Draw cells and grid
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell_x = board_x + col * CELL_SIZE
                cell_y = board_y + row * CELL_SIZE
                
                # Highlight hovered cell
                if hover_pos and hover_pos.row == row and hover_pos.col == col:
                    if game.board[row][col] is None and game.selected is not None:
                        pygame.draw.rect(
                            self.screen,
                            COLORS['cell_hover'],
                            (cell_x + 1, cell_y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                        )
                
                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    COLORS['grid_line'],
                    (cell_x, cell_y, CELL_SIZE, CELL_SIZE),
                    1
                )
                
                # Draw ball if present
                ball = game.board[row][col]
                if ball is not None:
                    is_selected = game.selected and game.selected.row == row and game.selected.col == col
                    self._draw_ball(cell_x, cell_y, ball.color, is_selected)
    
    def _draw_ball(self, cell_x: int, cell_y: int, color: Tuple[int, int, int], 
                   is_selected: bool = False) -> None:
        """Draw a ball in a cell."""
        center_x = cell_x + CELL_SIZE // 2
        center_y = cell_y + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 6
        
        # Apply bounce offset for selected ball
        if is_selected:
            center_y -= self.bounce_offset
        
        # Draw shadow
        shadow_color = (30, 30, 30)
        pygame.draw.circle(self.screen, shadow_color, (center_x + 2, center_y + 3), radius)
        
        # Draw main ball
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
        
        # Draw highlight (3D effect)
        highlight_radius = radius // 3
        highlight_x = center_x - radius // 3
        highlight_y = center_y - radius // 3
        highlight_color = tuple(min(255, c + 80) for c in color)
        pygame.draw.circle(self.screen, highlight_color, (highlight_x, highlight_y), highlight_radius)
        
        # Draw selection ring
        if is_selected:
            pygame.draw.circle(self.screen, COLORS['selection'], (center_x, center_y), radius + 3, 3)
    
    def _draw_sidebar(self, game: 'GameBoard', score_manager: ScoreManager) -> None:
        """Draw the sidebar with score and info."""
        sidebar_x = BOARD_PADDING * 2 + BOARD_SIZE * CELL_SIZE
        sidebar_y = BOARD_PADDING + 40
        
        # Current score
        score_label = self.fonts['medium'].render("Score", True, COLORS['text_dim'])
        self.screen.blit(score_label, (sidebar_x, sidebar_y))
        
        score_text = self.fonts['large'].render(str(game.score), True, COLORS['text'])
        self.screen.blit(score_text, (sidebar_x, sidebar_y + 25))
        
        # Next balls preview
        next_y = sidebar_y + 80
        next_label = self.fonts['medium'].render("Next Balls", True, COLORS['text_dim'])
        self.screen.blit(next_label, (sidebar_x, next_y))
        
        preview_y = next_y + 30
        for i, color_index in enumerate(game.next_balls):
            preview_x = sidebar_x + i * 45
            color = BALL_COLORS[color_index]
            
            # Draw mini ball
            pygame.draw.circle(self.screen, (30, 30, 30), (preview_x + 17, preview_y + 18), 15)
            pygame.draw.circle(self.screen, color, (preview_x + 15, preview_y + 15), 15)
            # Highlight
            highlight = tuple(min(255, c + 80) for c in color)
            pygame.draw.circle(self.screen, highlight, (preview_x + 10, preview_y + 10), 5)
        
        # Instructions
        inst_y = preview_y + 60
        inst_label = self.fonts['medium'].render("How to Play", True, COLORS['text_dim'])
        self.screen.blit(inst_label, (sidebar_x, inst_y))
        
        instructions = [
            "• Click a ball to select",
            "• Click empty cell to move",
            "• Make lines of 5+ balls",
            "• Same color to score",
            "",
            "Press R to restart"
        ]
        
        for i, line in enumerate(instructions):
            text = self.fonts['small'].render(line, True, COLORS['text_dim'])
            self.screen.blit(text, (sidebar_x, inst_y + 25 + i * 22))
        
        # High scores preview
        hs_y = inst_y + 180
        hs_label = self.fonts['medium'].render("High Scores", True, COLORS['text_dim'])
        self.screen.blit(hs_label, (sidebar_x, hs_y))
        
        if score_manager.highscores:
            for i, hs in enumerate(score_manager.highscores[:5]):
                rank_text = f"{i+1}. {hs.name[:10]}: {hs.score}"
                text = self.fonts['small'].render(rank_text, True, COLORS['text_dim'])
                self.screen.blit(text, (sidebar_x, hs_y + 25 + i * 20))
        else:
            no_scores = self.fonts['small'].render("No scores yet", True, COLORS['text_dim'])
            self.screen.blit(no_scores, (sidebar_x, hs_y + 25))
    
    def _draw_game_over(self, game: 'GameBoard', score_manager: ScoreManager) -> None:
        """Draw the game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2 - 50
        
        # Game Over text
        go_text = self.fonts['title'].render("GAME OVER", True, (255, 100, 100))
        go_rect = go_text.get_rect(center=(center_x, center_y - 80))
        self.screen.blit(go_text, go_rect)
        
        # Final score
        score_text = self.fonts['large'].render(f"Final Score: {game.score}", True, COLORS['text'])
        score_rect = score_text.get_rect(center=(center_x, center_y - 30))
        self.screen.blit(score_text, score_rect)
        
        # High scores table
        hs_title = self.fonts['medium'].render("Top 10 High Scores", True, COLORS['selection'])
        hs_rect = hs_title.get_rect(center=(center_x, center_y + 20))
        self.screen.blit(hs_title, hs_rect)
        
        if score_manager.highscores:
            for i, hs in enumerate(score_manager.highscores):
                color = COLORS['selection'] if hs.score == game.score else COLORS['text']
                entry = f"{i+1}. {hs.name}: {hs.score} ({hs.timestamp})"
                text = self.fonts['small'].render(entry, True, color)
                text_rect = text.get_rect(center=(center_x, center_y + 55 + i * 22))
                self.screen.blit(text, text_rect)
        else:
            no_scores = self.fonts['small'].render("No high scores yet", True, COLORS['text_dim'])
            no_rect = no_scores.get_rect(center=(center_x, center_y + 55))
            self.screen.blit(no_scores, no_rect)
        
        # Restart instruction
        restart_text = self.fonts['medium'].render("Press R to play again", True, COLORS['text_dim'])
        restart_rect = restart_text.get_rect(center=(center_x, WINDOW_HEIGHT - 50))
        self.screen.blit(restart_text, restart_rect)
    
    def _draw_name_input(self, game: 'GameBoard', input_text: str, new_rank: int) -> None:
        """Draw the name input overlay for high score."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        self.screen.blit(overlay, (0, 0))
        
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        
        # Congratulations text
        congrats = self.fonts['title'].render("NEW HIGH SCORE!", True, COLORS['selection'])
        congrats_rect = congrats.get_rect(center=(center_x, center_y - 100))
        self.screen.blit(congrats, congrats_rect)
        
        # Score display
        score_text = self.fonts['large'].render(f"Score: {game.score}", True, COLORS['text'])
        score_rect = score_text.get_rect(center=(center_x, center_y - 50))
        self.screen.blit(score_text, score_rect)
        
        # Rank display
        rank_text = self.fonts['medium'].render(f"Rank: #{new_rank}", True, COLORS['text'])
        rank_rect = rank_text.get_rect(center=(center_x, center_y - 15))
        self.screen.blit(rank_text, rank_rect)
        
        # Input prompt
        prompt = self.fonts['medium'].render("Enter your name:", True, COLORS['text'])
        prompt_rect = prompt.get_rect(center=(center_x, center_y + 30))
        self.screen.blit(prompt, prompt_rect)
        
        # Input box
        box_width = 300
        box_height = 40
        box_x = center_x - box_width // 2
        box_y = center_y + 55
        
        pygame.draw.rect(self.screen, COLORS['board_bg'], 
                        (box_x, box_y, box_width, box_height), border_radius=5)
        pygame.draw.rect(self.screen, COLORS['selection'], 
                        (box_x, box_y, box_width, box_height), 2, border_radius=5)
        
        # Input text with cursor
        display_text = input_text + ("_" if self.animation_tick % 30 < 15 else " ")
        text_surface = self.fonts['medium'].render(display_text, True, COLORS['text'])
        text_rect = text_surface.get_rect(midleft=(box_x + 10, box_y + box_height // 2))
        self.screen.blit(text_surface, text_rect)
        
        # Instructions
        inst = self.fonts['small'].render("Press ENTER to confirm", True, COLORS['text_dim'])
        inst_rect = inst.get_rect(center=(center_x, center_y + 115))
        self.screen.blit(inst, inst_rect)


# =============================================================================
# INPUT HANDLER
# =============================================================================

class InputHandler:
    """Handles user input."""
    
    def __init__(self):
        self.hover_pos: Optional[Position] = None
    
    def get_board_position(self, mouse_pos: Tuple[int, int]) -> Optional[Position]:
        """Convert mouse position to board position."""
        x, y = mouse_pos
        board_x = BOARD_PADDING
        board_y = BOARD_PADDING + 40
        
        # Check if within board bounds
        if not (board_x <= x < board_x + BOARD_SIZE * CELL_SIZE and
                board_y <= y < board_y + BOARD_SIZE * CELL_SIZE):
            return None
        
        col = (x - board_x) // CELL_SIZE
        row = (y - board_y) // CELL_SIZE
        
        return Position(row, col)
    
    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        """Update the hover position."""
        self.hover_pos = self.get_board_position(mouse_pos)


# =============================================================================
# MAIN GAME CLASS
# =============================================================================

class LinesGame:
    """Main game class that ties everything together."""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Lines - Match 5+ Balls")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.game_board = GameBoard()
        self.renderer = Renderer(self.screen)
        self.input_handler = InputHandler()
        self.score_manager = ScoreManager()
        
        self.state = GameState.PLAYING
        self.input_text = ""
        self.pending_rank = -1
        self.running = True
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEMOTION:
                self.input_handler.update_hover(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_click(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event)
    
    def _handle_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse click."""
        if self.state == GameState.PLAYING:
            board_pos = self.input_handler.get_board_position(pos)
            if board_pos:
                self.game_board.select_cell(board_pos)
                
                # Check for game over after the move
                if self.game_board.is_game_over():
                    self._trigger_game_over()
    
    def _handle_key(self, event: pygame.event.Event) -> None:
        """Handle keyboard input."""
        if self.state == GameState.ENTERING_NAME:
            if event.key == pygame.K_RETURN:
                # Submit the name
                name = self.input_text.strip() or "Anonymous"
                self.score_manager.add_highscore(name, self.game_board.score)
                self.input_text = ""
                self.state = GameState.GAME_OVER
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.unicode.isprintable() and len(self.input_text) < 20:
                self.input_text += event.unicode
        
        elif event.key == pygame.K_r:
            # Restart game
            self.game_board.reset()
            self.state = GameState.PLAYING
            self.input_text = ""
            self.pending_rank = -1
        
        elif event.key == pygame.K_ESCAPE:
            if self.state == GameState.GAME_OVER:
                self.running = False
    
    def _trigger_game_over(self) -> None:
        """Handle game over state transition."""
        if self.score_manager.is_highscore(self.game_board.score):
            # Calculate what rank this score would be
            temp_scores = self.score_manager.highscores + [
                HighScore("", self.game_board.score, "")
            ]
            temp_scores.sort(key=lambda x: x.score, reverse=True)
            for i, hs in enumerate(temp_scores):
                if hs.score == self.game_board.score and hs.name == "":
                    self.pending_rank = i + 1
                    break
            
            self.state = GameState.ENTERING_NAME
        else:
            self.state = GameState.GAME_OVER
    
    def update(self) -> None:
        """Update game state."""
        self.renderer.update_animations()
    
    def render(self) -> None:
        """Render the game."""
        self.renderer.render(
            self.game_board,
            self.state,
            self.score_manager,
            self.input_handler.hover_pos,
            self.input_text,
            self.pending_rank
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
# ENTRY POINT
# =============================================================================

def main():
    """Entry point for the game."""
    print("=" * 50)
    print("       LINES - Color Lines Puzzle Game")
    print("=" * 50)
    print("\nControls:")
    print("  • Click on a ball to select it")
    print("  • Click on an empty cell to move")
    print("  • Form lines of 5+ same-colored balls")
    print("  • Press R to restart at any time")
    print("  • Press ESC on game over to quit")
    print("\nStarting game...\n")
    
    game = LinesGame()
    game.run()


if __name__ == "__main__":
    main()
