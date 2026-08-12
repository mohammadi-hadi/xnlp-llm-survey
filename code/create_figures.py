#!/usr/bin/env python3
"""
Generate figures for XAI Survey Paper
- Taxonomy of Explainable NLP Methods (hierarchical tree with elbow connectors)
- Method Selection Decision Tree (clean flowchart)
- Explainability approaches overview and accuracy-interpretability schematic
- Vector redraw of the traditional AI vs XAI comparison (ai_vs_xai_vector)

Figure numbering and captions are handled by LaTeX; figures contain no baked-in captions.

Print-legibility contract: each figure is placed in the paper at ~6.3in
(\\textwidth) or a fraction of it. Printed text size = fontsize * placed_width /
canvas_width, and every label must come out >= 7pt printed. Canvas sizes and
font sizes below are chosen together to satisfy that; if you resize a canvas,
rescale the fonts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

# Set publication-quality defaults
plt.rcParams['pdf.fonttype'] = 42   # embed TrueType (Type 42), not Type 3: journal font requirement
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Grayscale color scheme for black and white printing
COLORS = {
    # Root
    'root': '#1A1A1A',           # Near black

    # Local Methods - Dark gray with diagonal pattern
    'local': '#4D4D4D',          # Dark gray
    'local_bg': '#F0F0F0',       # Very light gray
    'local_border': '#4D4D4D',

    # Global Methods - Medium gray with dot pattern
    'global': '#666666',         # Medium gray
    'global_bg': '#E8E8E8',
    'global_border': '#666666',

    # LLM-Era Methods - Light-medium gray with crosshatch
    'llm': '#808080',            # Light-medium gray
    'llm_bg': '#E0E0E0',
    'llm_border': '#808080',

    # Neutrals
    'method': '#FAFAFA',         # Near white for method backgrounds
    'text_primary': '#1A1A1A',   # Near black
    'connector': '#4D4D4D',      # Dark gray
    'arrow': '#4D4D4D',

    # Decision tree colors
    'decision': '#D0D0D0',       # Light gray for decision diamonds
    'recommend': '#4D4D4D',      # Dark gray for recommendations
}

# Pattern definitions for grayscale printing (distinguishes categories)
PATTERNS = {
    'local': '///',      # Diagonal lines
    'global': '...',     # Dots
    'llm': 'xxx',        # Crosshatch
    'none': '',          # No pattern
}


def create_taxonomy_diagram():
    """Create flattened 2-level taxonomy diagram.

    Placed at \\textwidth (~6.3in). Canvas 11in wide -> scale ~0.57;
    smallest font (12.5pt leaves/legend) prints at ~7.2pt.
    """

    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    def draw_background_region(x, width, color, pattern):
        """Draw subtle background region with pattern for category grouping.

        Two stacked rectangles: a low-alpha tint underneath, and a hatch-only
        overlay on top. Matplotlib renders hatches in the patch edge color, so
        the overlay needs a visible edgecolor (alpha would also fade the hatch,
        hence full-alpha overlay with facecolor='none')."""
        ax.add_patch(Rectangle(
            (x, 0.55), width, 5.15,
            facecolor=color, alpha=0.12, zorder=-2, edgecolor='none'))
        ax.add_patch(Rectangle(
            (x, 0.55), width, 5.15,
            facecolor='none', zorder=-1, edgecolor='#C8C8C8',
            linewidth=0.0, hatch=pattern))

    def draw_root(x, y, text):
        """Draw root node with maximum prominence."""
        box = FancyBboxPatch(
            (x - 1.7, y - 0.33), 3.4, 0.66,
            boxstyle="round,pad=0.15",
            facecolor=COLORS['root'],
            edgecolor='#37474F',
            linewidth=2.0,
            zorder=10
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=15, fontweight='bold', color='white', zorder=11)
        return {'center': (x, y), 'bottom': (x, y - 0.42), 'top': (x, y + 0.42)}

    def draw_category(x, y, text, category):
        """Draw category box with colored background and near-black text
        (all headers use dark text for contrast on the light fills)."""
        box = FancyBboxPatch(
            (x - 1.55, y - 0.38), 3.1, 0.76,
            boxstyle="round,pad=0.12",
            facecolor=COLORS[f'{category}_bg'],
            edgecolor=COLORS[f'{category}_border'],
            linewidth=2.0,
            zorder=8
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=13, fontweight='bold',
                color=COLORS['text_primary'], zorder=9)
        return {'center': (x, y), 'bottom': (x, y - 0.46), 'top': (x, y + 0.46)}

    def draw_method(x, y, text):
        """Draw method as text with an OPAQUE white background box so column
        spines can pass behind labels without striking the text."""
        ax.text(x, y, text, ha='center', va='center',
                fontsize=12.5, fontweight='normal',
                color=COLORS['text_primary'],
                bbox=dict(boxstyle='round,pad=0.22',
                         facecolor='white',
                         edgecolor='#BDBDBD',
                         linewidth=0.8,
                         alpha=1.0),
                zorder=4)
        return {'center': (x, y), 'top': (x, y + 0.2)}

    def draw_column_tree(parent_bottom, columns, color, lw=1.6):
        """Connect a category to columns of stacked leaves.

        One vertical from the parent to a horizontal bar, then ONE vertical
        spine per column drawn from the bar down to the lowest leaf, BEHIND
        the opaque leaf label boxes (zorder 1 < 4). No connector ever crosses
        visible text.
        columns: list of (x, y_top_leaf, y_bottom_leaf).
        """
        bar_y = parent_bottom[1] - 0.28
        ax.plot([parent_bottom[0], parent_bottom[0]], [parent_bottom[1], bar_y],
                color=color, linewidth=lw, zorder=1)
        xs = [c[0] for c in columns]
        if len(xs) > 1:
            ax.plot([min(xs), max(xs)], [bar_y, bar_y], color=color,
                    linewidth=lw, zorder=1)
        for x, y_top, y_bottom in columns:
            ax.plot([x, x], [bar_y, y_bottom], color=color, linewidth=lw,
                    zorder=1)

    # === Background regions for each category ===
    draw_background_region(0.20, 3.45, COLORS['local_bg'], PATTERNS['local'])
    draw_background_region(3.80, 3.40, COLORS['global_bg'], PATTERNS['global'])
    draw_background_region(7.35, 3.45, COLORS['llm_bg'], PATTERNS['llm'])

    # === Root node ===
    root = draw_root(5.5, 6.6, 'Explainable NLP Methods')

    # === Level 1: Categories ===
    cat_y = 5.15
    local_cat = draw_category(1.93, cat_y, 'Local Explanations\n(Single Prediction)', 'local')
    global_cat = draw_category(5.5, cat_y, 'Global Explanations\n(Model-Wide)', 'global')
    llm_cat = draw_category(9.07, cat_y, 'LLM-Era Methods', 'llm')

    # Connect root to categories
    bar_y = root['bottom'][1] - 0.28
    ax.plot([5.5, 5.5], [root['bottom'][1], bar_y], color=COLORS['connector'],
            linewidth=2.0, zorder=1)
    ax.plot([1.93, 9.07], [bar_y, bar_y], color=COLORS['connector'],
            linewidth=2.0, zorder=1)
    for cx, cat in [(1.93, local_cat), (5.5, global_cat), (9.07, llm_cat)]:
        ax.plot([cx, cx], [bar_y, cat['top'][1]], color=COLORS['connector'],
                linewidth=2.0, zorder=1)

    # === Level 2: Methods (stacked columns; spine drawn behind labels) ===
    # Local: two columns of six
    local_col1_x, local_col2_x = 1.1, 2.85
    col1_rows = [3.95, 3.35, 2.75, 2.15, 1.55, 0.95]
    col1_labels = ['LIME', 'SHAP', 'Anchors', 'LRP', 'DeepLIFT', 'Counterfactual']
    col2_rows = [3.95, 3.30, 2.65, 2.00, 1.42, 0.95]
    col2_labels = ['Integrated\nGradients', 'Attention\nWeights',
                   'Attention\nRollout', 'Influence\nFunctions',
                   'Prototypes', 'Contrastive']
    for y, lab in zip(col1_rows, col1_labels):
        draw_method(local_col1_x, y, lab)
    for y, lab in zip(col2_rows, col2_labels):
        draw_method(local_col2_x, y, lab)
    draw_column_tree(local_cat['bottom'],
                     [(local_col1_x, col1_rows[0], col1_rows[-1]),
                      (local_col2_x, col2_rows[0], col2_rows[-1])],
                     color=COLORS['local'], lw=1.6)

    # Global: one column of four
    global_x = 5.5
    global_rows = [3.85, 3.1, 2.35, 1.6]
    global_labels = ['Rule Extraction', 'SHAP (Global)', 'TCAV',
                     'Probing Classifiers']
    for y, lab in zip(global_rows, global_labels):
        draw_method(global_x, y, lab)
    draw_column_tree(global_cat['bottom'],
                     [(global_x, global_rows[0], global_rows[-1])],
                     color=COLORS['global'], lw=1.6)

    # LLM-Era: two columns of three
    llm_col1_x, llm_col2_x = 8.2, 9.95
    llm_rows = [3.85, 3.1, 2.35]
    llm_col1_labels = ['Chain-of-Thought\n(Zero- / Few-shot)', 'Self-Critique',
                       'Rationale\nGeneration']
    llm_col2_labels = ['Activation\nPatching', 'SAE Features',
                       'Circuit\nTracing']
    for y, lab in zip(llm_rows, llm_col1_labels):
        draw_method(llm_col1_x, y, lab)
    for y, lab in zip(llm_rows, llm_col2_labels):
        draw_method(llm_col2_x, y, lab)
    draw_column_tree(llm_cat['bottom'],
                     [(llm_col1_x, llm_rows[0], llm_rows[-1]),
                      (llm_col2_x, llm_rows[0], llm_rows[-1])],
                     color=COLORS['llm'], lw=1.6)

    # === Legend with patterns (top-left corner is empty) ===
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['local_bg'],
                      edgecolor=COLORS['local_border'],
                      hatch=PATTERNS['local'],
                      label='Local Methods'),
        mpatches.Patch(facecolor=COLORS['global_bg'],
                      edgecolor=COLORS['global_border'],
                      hatch=PATTERNS['global'],
                      label='Global Methods'),
        mpatches.Patch(facecolor=COLORS['llm_bg'],
                      edgecolor=COLORS['llm_border'],
                      hatch=PATTERNS['llm'],
                      label='LLM-Era Methods'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=12.5, framealpha=0.95, edgecolor='#2C3E50')

    # No baked-in caption: the LaTeX \caption provides figure numbering and text.

    plt.tight_layout()
    plt.savefig('taxonomy_diagram.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('taxonomy_diagram.png', format='png', bbox_inches='tight')
    print("Created: taxonomy_diagram.pdf and .png")
    plt.close()


def create_decision_tree():
    """Create decision tree flowchart with clean vertical flow and no crossed lines.

    Placed at \\textwidth (~6.3in). Canvas 11.2in wide -> scale ~0.57;
    smallest font (12.5pt) prints at ~7.1pt.
    """

    fig, ax = plt.subplots(figsize=(11.2, 8.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 8.4)
    ax.axis('off')

    def draw_diamond(x, y, text, color='#FFFFFF', size=0.88):
        """Draw a diamond decision node with white fill and black border."""
        diamond = plt.Polygon(
            [(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
            facecolor=color, edgecolor='#1A1A1A', linewidth=2.2, zorder=10
        )
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=12.5,
                fontweight='bold', wrap=True, color='#1A1A1A', zorder=11)
        return {'top': (x, y + size), 'bottom': (x, y - size),
                'left': (x - size, y), 'right': (x + size, y)}

    def draw_rect(x, y, text, color=COLORS['recommend'], width=2.0, height=0.65,
                  linestyle='-', linewidth=2.0):
        """Draw a rectangular recommendation node with customizable border style."""
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor='#1A1A1A', linewidth=linewidth,
            linestyle=linestyle, zorder=10
        )
        ax.add_patch(rect)
        # Use dark text on light backgrounds, white on dark
        light_colors = ['#C0C0C0', '#D0D0D0', '#E0E0E0', '#F0F0F0', '#FFFFFF', '#E8E8E8', '#F5F5F5']
        text_color = '#1A1A1A' if color in light_colors else 'white'
        # Normal weight: bold serif at 12.5pt overflows the box widths
        ax.text(x, y, text, ha='center', va='center', fontsize=12.5,
                color=text_color, fontweight='normal', wrap=True, zorder=11)
        return {'top': (x, y + height/2), 'bottom': (x, y - height/2),
                'left': (x - width/2, y), 'right': (x + width/2, y)}

    def draw_start(x, y, text, color='#1A1A1A'):
        """Draw start/end node (rounded rectangle)."""
        rect = FancyBboxPatch(
            (x - 1.45, y - 0.33), 2.9, 0.66,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor=color, edgecolor='#1A1A1A', linewidth=2, zorder=10
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=14,
                color='white', fontweight='bold', zorder=11)
        return {'bottom': (x, y - 0.33), 'top': (x, y + 0.33)}

    def draw_arrow(start, end, label='', label_offset=(0, 0.2)):
        """Draw an arrow with optional label offset AWAY from the line
        (default above; pass a horizontal offset for vertical arrows)."""
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.8),
                    zorder=5)
        if label:
            mid_x = (start[0] + end[0]) / 2 + label_offset[0]
            mid_y = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, fontsize=12.5, ha='center',
                    va='center', color='#1A1A1A', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='none', alpha=0.9), zorder=6)

    def draw_elbow_arrow(start, mid_x, end, label='', label_pos='mid'):
        """Draw an elbow-shaped arrow (horizontal then vertical)."""
        # Horizontal line
        ax.plot([start[0], mid_x], [start[1], start[1]], color=COLORS['arrow'], lw=1.8, zorder=5)
        # Vertical line with arrow
        ax.annotate('', xy=end, xytext=(mid_x, start[1]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.8), zorder=5)
        if label:
            if label_pos == 'start':
                lx, ly = (start[0] + mid_x) / 2, start[1] + 0.25
            else:
                lx, ly = mid_x, (start[1] + end[1]) / 2
            ax.text(lx, ly, label, fontsize=12.5, ha='center', va='center',
                    color='#1A1A1A', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='none', alpha=0.9), zorder=6)

    # Category styling contract (must match the legend):
    #   Local methods  -> white fill,   solid border, lw 3.0
    #   Global methods -> #F0F0F0 fill, dashed border, lw 2.5
    #   LLM methods    -> #E8E8E8 fill, dotted border, lw 2.5

    # === Start node ===
    start = draw_start(6.5, 8.0, 'Select XAI Method')

    # === First decision: Model Access ===
    d1 = draw_diamond(6.5, 6.85, 'Model\nAccess?', size=0.78)
    draw_arrow(start['bottom'], d1['top'])

    # === Three branches from Model Access (well-separated) ===

    # LEFT: Full Access path
    d2_full = draw_diamond(2.29, 4.9, 'Explanation\nScope?')
    draw_elbow_arrow(d1['left'], 2.29, d2_full['top'], 'Full Access', 'start')

    # Local methods (Full Access) - white fill, solid thick border
    r_local = draw_rect(1.15, 3.1, 'Integrated Gradients\nAttention\nSHAP',
                        color='#FFFFFF', width=2.1, height=1.05, linestyle='-', linewidth=3.0)
    draw_elbow_arrow(d2_full['left'], 1.15, r_local['top'], 'Local', 'start')

    # Global methods (Full Access) - light gray fill, dashed border
    r_global = draw_rect(3.42, 3.1, 'Probing Classifiers\nTCAV\nDistillation',
                         color='#F0F0F0', width=1.85, height=1.05, linestyle='--', linewidth=2.5)
    draw_elbow_arrow(d2_full['right'], 3.42, r_global['top'], 'Global', 'start')

    # MIDDLE: API Access path (label offset to the right of the vertical arrow)
    d2_api = draw_diamond(6.5, 4.9, 'Is it\nan LLM?')
    draw_arrow(d1['bottom'], d2_api['top'], 'API Only', label_offset=(0.78, 0.02))

    # LLM Yes - medium gray fill, dotted border
    r_llm_yes = draw_rect(5.42, 3.1, 'Chain-of-Thought\nSelf-Explanation',
                          color='#E8E8E8', width=1.75, height=0.85, linestyle=':', linewidth=2.5)
    draw_elbow_arrow(d2_api['left'], 5.42, r_llm_yes['top'], 'Yes', 'start')

    # Caveat note for reasoning models: CoT traces are not guaranteed faithful
    ax.plot([5.42, 5.42], [r_llm_yes['bottom'][1], 2.35], color=COLORS['arrow'],
            lw=1.2, linestyle=':', zorder=5)
    ax.text(5.42, 2.0, 'Reasoning model? Verify CoT\nfaithfulness before use',
            ha='center', va='center', fontsize=12.5, fontstyle='italic',
            color='#1A1A1A',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF',
                      edgecolor='#1A1A1A', linewidth=1.2, linestyle='--'),
            zorder=11)

    # LLM No - local model-agnostic methods: white fill, solid thick border
    r_llm_no = draw_rect(7.7, 3.1, 'LIME\nAnchors',
                         color='#FFFFFF', width=1.1, height=0.85, linestyle='-', linewidth=3.0)
    draw_elbow_arrow(d2_api['right'], 7.7, r_llm_no['top'], 'No', 'start')

    # RIGHT: Black-box path - local perturbation methods: white fill, solid thick border
    # (edge label drawn manually so it stays clear of the legend)
    r_blackbox = draw_rect(9.9, 4.9, 'LIME\nCounterfactuals\nAnchors',
                           color='#FFFFFF', width=1.7, height=1.05, linestyle='-', linewidth=3.0)
    draw_elbow_arrow(d1['right'], 9.9, r_blackbox['top'])
    ax.text(7.95, 7.12, 'Black-box', fontsize=12.5, ha='center', va='center',
            color='#1A1A1A', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.9), zorder=6)

    # === Audience section (as informational box, not decision) ===
    audience_bg = FancyBboxPatch(
        (0.15, 0.1), 10.9, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#F5F5F5', edgecolor='#909090', linewidth=1.5, zorder=1
    )
    ax.add_patch(audience_bg)

    # Header for audience section
    ax.text(5.6, 1.36, 'Tailor Explanations to Target Audience',
            ha='center', fontsize=13, fontweight='bold', color='#1A1A1A', zorder=10)

    # Audience boxes with even spacing - grayscale progression
    audiences = [
        (1.55, 'End Users', 'Simple highlights\nNatural language', '#D0D0D0'),
        (4.3, 'Domain Experts', 'Feature importance\nDomain terms', '#A0A0A0'),
        (7.05, 'ML Practitioners', 'Gradients\nAttention maps', '#707070'),
        (9.8, 'Regulators', 'Auditable\nMethodology', '#505050'),
    ]

    for x, title, desc, color in audiences:
        # Title box
        title_rect = FancyBboxPatch(
            (x - 0.95, 0.82), 1.9, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='#1A1A1A', linewidth=1.2, zorder=10
        )
        ax.add_patch(title_rect)
        # Use white text on darker backgrounds, black on lighter
        text_color = 'white' if color in ['#707070', '#505050'] else '#1A1A1A'
        ax.text(x, 1.03, title, ha='center', va='center', fontsize=12.5,
                fontweight='bold', color=text_color, zorder=11)
        # Description
        ax.text(x, 0.47, desc, ha='center', va='center', fontsize=12.5,
                color='#1A1A1A', zorder=10)

    # === Legend - border styles match the node styling contract above ===
    legend_elements = [
        Line2D([0], [0], marker='D', color='none', markerfacecolor='#FFFFFF',
               markeredgecolor='#1A1A1A', markeredgewidth=2, markersize=13,
               label='Decision Point'),
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='#1A1A1A', linewidth=3,
                       linestyle='-', label='Local Methods'),
        mpatches.Patch(facecolor='#F0F0F0', edgecolor='#1A1A1A', linewidth=2.5,
                       linestyle='--', label='Global Methods'),
        mpatches.Patch(facecolor='#E8E8E8', edgecolor='#1A1A1A', linewidth=2.5,
                       linestyle=':', label='LLM Methods'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12.5,
              framealpha=0.95, edgecolor='#1A1A1A', bbox_to_anchor=(0.995, 0.995))

    # No baked-in caption: the LaTeX \caption provides figure numbering and text.

    plt.tight_layout()
    plt.savefig('decision_tree.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('decision_tree.png', format='png', bbox_inches='tight')
    print("Created: decision_tree.pdf and .png")
    plt.close()


def create_explainability_approaches():
    """Create explainability approaches categorization diagram.

    Placed at \\textwidth (~6.3in). Canvas 10.5in wide -> scale ~0.6;
    smallest font (12.5pt) prints at ~7.5pt.
    """

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.4, 5.6)
    ax.axis('off')

    def draw_box(x, y, text, color, width=2.2, height=0.55, fontsize=12.5, bold=False,
                 linestyle='-', linewidth=1.5):
        """Draw a rounded box with text and customizable border style."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=color, edgecolor='#1A1A1A', linewidth=linewidth,
            linestyle=linestyle, zorder=10
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        # Dark text on light backgrounds, white on dark
        dark_colors = [COLORS['root'], '#505050', '#606060', '#707070', '#404040', '#303030']
        text_color = 'white' if color in dark_colors else '#1A1A1A'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color=text_color, wrap=True, zorder=11)
        return {'center': (x, y), 'bottom': (x, y - height/2), 'top': (x, y + height/2)}

    def draw_elbow_connection(start, end, color='#606060', lw=1.2):
        """Draw an elbow connector."""
        mid_y = (start[1] + end[1]) / 2
        ax.plot([start[0], start[0]], [start[1], mid_y], color=color, linewidth=lw, zorder=1)
        ax.plot([start[0], end[0]], [mid_y, mid_y], color=color, linewidth=lw, zorder=1)
        ax.plot([end[0], end[0]], [mid_y, end[1]], color=color, linewidth=lw, zorder=1)

    def draw_vertical_connection(start, end, color='#606060', lw=1.2):
        """Draw a simple vertical connection."""
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, zorder=1)

    # === Root node ===
    root = draw_box(5.25, 5.15, 'Explainability Approaches', COLORS['root'],
                    width=3.5, height=0.62, fontsize=15, bold=True)

    # === Level 1: Three main categorization dimensions (grayscale with border styles) ===
    timing_color = '#404040'    # Dark gray
    scope_color = '#505050'     # Medium-dark gray
    access_color = '#606060'    # Medium gray

    timing = draw_box(1.85, 4.1, 'By Timing', timing_color, width=1.7, height=0.55, fontsize=13, bold=True,
                      linestyle='-', linewidth=3.0)
    scope = draw_box(5.25, 4.1, 'By Scope', scope_color, width=1.7, height=0.55, fontsize=13, bold=True,
                     linestyle='--', linewidth=2.5)
    access = draw_box(8.8, 4.1, 'By Model Access', access_color, width=2.1, height=0.55, fontsize=13, bold=True,
                      linestyle=':', linewidth=2.5)

    # Connect root to level 1
    draw_elbow_connection(root['bottom'], timing['top'], color='#4D4D4D', lw=1.5)
    draw_vertical_connection(root['bottom'], scope['top'], color='#4D4D4D', lw=1.5)
    draw_elbow_connection(root['bottom'], access['top'], color='#4D4D4D', lw=1.5)

    # === Level 2: Subcategories (light grayscale) ===
    sub_timing_color = '#D0D0D0'
    sub_scope_color = '#D8D8D8'
    sub_access_color = '#E0E0E0'

    # Under Timing
    direct = draw_box(1.0, 2.95, 'Direct\nInterpretability', sub_timing_color, width=1.55, height=0.72, fontsize=12.5)
    posthoc = draw_box(2.7, 2.95, 'Post-hoc\nExplanation', sub_timing_color, width=1.4, height=0.72, fontsize=12.5)
    draw_elbow_connection(timing['bottom'], direct['top'], color=timing_color)
    draw_elbow_connection(timing['bottom'], posthoc['top'], color=timing_color)

    # Under Scope
    local = draw_box(4.5, 2.95, 'Local\n(Per Instance)', sub_scope_color, width=1.55, height=0.72, fontsize=12.5)
    global_exp = draw_box(6.25, 2.95, 'Global\n(Model-Wide)', sub_scope_color, width=1.5, height=0.72, fontsize=12.5)
    draw_elbow_connection(scope['bottom'], local['top'], color=scope_color)
    draw_elbow_connection(scope['bottom'], global_exp['top'], color=scope_color)

    # Under Model Access
    specific = draw_box(8.0, 2.95, 'Model-\nSpecific', sub_access_color, width=1.15, height=0.72, fontsize=12.5)
    agnostic = draw_box(9.7, 2.95, 'Model-\nAgnostic', sub_access_color, width=1.15, height=0.72, fontsize=12.5)
    draw_elbow_connection(access['bottom'], specific['top'], color=access_color)
    draw_elbow_connection(access['bottom'], agnostic['top'], color=access_color)

    # === Level 3: Examples ===
    example_color = '#F5F5F5'
    example_fontsize = 12.5

    # Examples under Direct Interpretability
    direct_ex = draw_box(1.0, 1.45, 'Decision Trees\nLinear Models\nRule Lists', example_color,
                         width=1.45, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(direct['bottom'], direct_ex['top'], color='#808080')

    # Examples under Post-hoc
    posthoc_ex = draw_box(2.7, 1.45, 'LIME\nSHAP\nAnchors', example_color,
                          width=1.0, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(posthoc['bottom'], posthoc_ex['top'], color='#808080')

    # Examples under Local
    local_ex = draw_box(4.45, 1.45, 'Feature Attribution\nCounterfactuals\nInfluence Functions', example_color,
                        width=1.85, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(local['bottom'], local_ex['top'], color='#909090')

    # Examples under Global
    global_ex = draw_box(6.55, 1.45, 'Model Distillation\nProbing Classifiers\nTCAV', example_color,
                         width=1.85, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(global_exp['bottom'], global_ex['top'], color='#909090')

    # Examples under Model-Specific
    specific_ex = draw_box(8.35, 1.45, 'Attention Viz\nLRP\nDeepLIFT', example_color,
                           width=1.35, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(specific['bottom'], specific_ex['top'], color='#A0A0A0')

    # Examples under Model-Agnostic
    agnostic_ex = draw_box(9.85, 1.45, 'LIME\nSHAP\nAnchors', example_color,
                           width=1.0, height=1.0, fontsize=example_fontsize)
    draw_vertical_connection(agnostic['bottom'], agnostic_ex['top'], color='#A0A0A0')

    # No legend: the three dimension headers are labeled directly on the boxes,
    # and the tightened ylim removes the former dead whitespace band.

    plt.tight_layout()
    plt.savefig('explainability_approaches.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('explainability_approaches.png', format='png', bbox_inches='tight')
    print("Created: explainability_approaches.pdf and .png")
    plt.close()


def create_accuracy_interpretability():
    """Create accuracy vs interpretability trade-off scatter plot (schematic).

    Placed at 0.85\\textwidth (~5.36in). Canvas 8in wide -> scale ~0.68;
    smallest font (11.5pt labels) prints at ~7.8pt.
    """
    fig, ax = plt.subplots(figsize=(8, 6.4))

    # Model data: (x_accuracy, y_interpretability, name, label side)
    # Schematic positions along the accuracy-interpretability diagonal
    models = [
        (1.5, 9.0, 'Linear Regression', 'right'),
        (2.2, 8.0, 'Decision Trees', 'right'),
        (4.0, 6.0, 'K-Nearest Neighbors', 'right'),
        (5.0, 5.0, 'Random Forests', 'right'),
        (6.5, 3.8, 'Support Vector Machines', 'right'),
        (8.0, 2.5, 'Deep Neural Networks', 'left'),
    ]

    # Plot models as gray filled circles for grayscale printing
    marker_color = '#808080'
    for x, y, name, side in models:
        ax.scatter(x, y, s=350, c=marker_color, edgecolors='#1A1A1A',
                   linewidth=1.5, zorder=10)
        # Label beside each marker (left for points near the right edge)
        if side == 'right':
            ax.text(x + 0.45, y, name, fontsize=11.5, ha='left', va='center',
                    fontweight='bold', color='#1A1A1A', zorder=11)
        else:
            ax.text(x - 0.45, y, name, fontsize=11.5, ha='right', va='center',
                    fontweight='bold', color='#1A1A1A', zorder=11)

    # Axis configuration - simple L-shaped axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Remove top and right spines for L-shaped axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Style remaining spines
    ax.spines['left'].set_color('#1A1A1A')
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_color('#1A1A1A')
    ax.spines['bottom'].set_linewidth(2.5)

    # Axis labels - bold, centered
    ax.set_xlabel('Accuracy', fontsize=14, fontweight='bold', color='#1A1A1A')
    ax.set_ylabel('Interpretability', fontsize=14, fontweight='bold', color='#1A1A1A')

    # Remove tick marks and labels; axes are schematic (no quantitative scale)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('accuracy_interpretability.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('accuracy_interpretability.png', format='png', bbox_inches='tight')
    print("Created: accuracy_interpretability.pdf and .png")
    plt.close()


def create_ai_vs_xai():
    """Vector redraw of the traditional AI vs XAI comparison.

    Replaces the low-resolution raster figures/ai_vs_xai.pdf (which is kept
    untouched); saved as ai_vs_xai_vector.pdf/.png. Placed at 0.9\\textwidth
    (~5.67in). Canvas 10in wide -> scale ~0.57; smallest font (12.5pt) prints
    at ~7.2pt.
    """
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    def draw_panel(y0, title):
        panel = FancyBboxPatch(
            (0.15, y0), 9.7, 2.55,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor='#FAFAFA', edgecolor='#909090', linewidth=1.5, zorder=0
        )
        ax.add_patch(panel)
        ax.text(0.45, y0 + 2.22, title, ha='left', va='center', fontsize=14.5,
                fontweight='bold', color='#1A1A1A', zorder=10)

    def draw_node(x, y, text, width, height, facecolor='#FFFFFF',
                  hatch='', bold=True, text_bbox=False):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=facecolor, edgecolor='#1A1A1A', linewidth=2.0,
            hatch=hatch, zorder=5
        )
        ax.add_patch(box)
        dark = facecolor in ('#1A1A1A', '#333333', '#4D4D4D')
        kwargs = {}
        if text_bbox:
            # Opaque backing so a hatched fill never strikes the text
            kwargs['bbox'] = dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=1.0)
        ax.text(x, y, text, ha='center', va='center', fontsize=13,
                fontweight='bold' if bold else 'normal',
                color='white' if dark else '#1A1A1A', zorder=6, **kwargs)
        return {'left': (x - width/2, y), 'right': (x + width/2, y)}

    def draw_flow_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#4D4D4D', lw=2.2),
                    zorder=4)

    def draw_user(x, y, quote):
        """Simple user glyph (head + shoulders) with an italic quote below."""
        ax.add_patch(Circle((x, y + 0.42), 0.16, facecolor='#B0B0B0',
                            edgecolor='#1A1A1A', linewidth=1.5, zorder=6))
        body = mpatches.Wedge((x, y - 0.08), 0.30, 0, 180,
                              facecolor='#B0B0B0', edgecolor='#1A1A1A',
                              linewidth=1.5, zorder=6)
        ax.add_patch(body)
        ax.text(x, y - 0.42, quote, ha='center', va='center', fontsize=12.5,
                fontstyle='italic', color='#1A1A1A', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#707070', linewidth=1.0))

    # === Top panel: Traditional AI ===
    draw_panel(2.95, 'Traditional AI')
    y_top = 3.85
    a_in = draw_node(1.55, y_top, 'Input Data', 1.55, 0.7, facecolor='#F0F0F0')
    a_model = draw_node(4.1, y_top, 'Black-Box\nModel', 1.8, 1.0,
                        facecolor='#4D4D4D')
    a_out = draw_node(6.6, y_top, 'Prediction', 1.55, 0.7, facecolor='#F0F0F0')
    draw_flow_arrow(a_in['right'], a_model['left'])
    draw_flow_arrow(a_model['right'], a_out['left'])
    draw_flow_arrow(a_out['right'], (8.35, y_top))
    draw_user(8.85, y_top, '"Why? Unclear"')

    # === Bottom panel: Explainable AI ===
    draw_panel(0.15, 'Explainable AI (XAI)')
    y_bot = 1.05
    b_in = draw_node(1.55, y_bot, 'Input Data', 1.55, 0.7, facecolor='#F0F0F0')
    b_model = draw_node(4.1, y_bot, 'Model +\nExplanation\nInterface', 1.8, 1.3,
                        facecolor='#FFFFFF', hatch='///', text_bbox=True)
    b_out = draw_node(6.6, y_bot, 'Prediction +\nExplanation', 1.7, 0.95,
                      facecolor='#F0F0F0')
    draw_flow_arrow(b_in['right'], b_model['left'])
    draw_flow_arrow(b_model['right'], b_out['left'])
    draw_flow_arrow(b_out['right'], (8.35, y_bot))
    draw_user(8.85, y_bot, '"I understand why"')

    plt.tight_layout()
    plt.savefig('ai_vs_xai_vector.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('ai_vs_xai_vector.png', format='png', bbox_inches='tight')
    print("Created: ai_vs_xai_vector.pdf and .png")
    plt.close()


if __name__ == '__main__':
    print("Generating figures for XAI Survey Paper...")
    print("=" * 50)
    create_taxonomy_diagram()
    create_decision_tree()
    create_explainability_approaches()
    create_accuracy_interpretability()
    create_ai_vs_xai()
    print("=" * 50)
    print("All figures generated successfully!")
