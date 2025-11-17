"""
HTML utilities for visual component extraction and IoU calculation.

This module provides functions to extract visual components from HTML files
and compute layout similarity using Intersection over Union (IoU) metrics.
"""

import io
import json
import os
import traceback
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from shapely.geometry import box
from shapely.ops import unary_union


def boxes_adjacent(
    box1: Dict[str, float],
    box2: Dict[str, float],
    align_tolerance: float = 8,
    adj_tolerance: float = 4
) -> bool:
    """
    Determine if two boxes are adjacent considering horizontal and vertical tolerance.

    Boxes are considered adjacent if:
    - They are nearly aligned vertically (vertical centers close within y_tolerance)
    - They are horizontally sequential without a large gap
    - OR they are vertically sequential

    Args:
        box1: First bounding box with x, y, width, height
        box2: Second bounding box with x, y, width, height
        align_tolerance: Tolerance for alignment checking
        adj_tolerance: Tolerance for adjacency checking

    Returns:
        True if boxes are adjacent, False otherwise
    """
    # Calculate vertical centers and horizontal centers
    vertical_center1 = box1['y'] + box1['height'] / 2
    vertical_center2 = box2['y'] + box2['height'] / 2
    horizontal_center1 = box1['x'] + box1['width'] / 2
    horizontal_center2 = box2['x'] + box2['width'] / 2

    # Check vertical alignment
    vertically_aligned = abs(vertical_center1 - vertical_center2) <= align_tolerance

    # Check horizontal adjacency
    horizontally_adjacent = (
        (box1['x'] + box1['width'] + adj_tolerance >= box2['x'] and box1['x'] < box2['x']) or
        (box2['x'] + box2['width'] + adj_tolerance >= box1['x'] and box2['x'] < box1['x'])
    )

    # Check horizontal alignment for vertical adjacency
    horizontally_aligned = abs(horizontal_center1 - horizontal_center2) <= align_tolerance

    # Check vertical adjacency
    vertically_adjacent = (
        (box1['y'] + box1['height'] + adj_tolerance >= box2['y'] and box1['y'] < box2['y']) or
        (box2['y'] + box2['height'] + adj_tolerance >= box1['y'] and box2['y'] < box1['y'])
    )

    return (vertically_aligned and horizontally_adjacent) or (horizontally_aligned and vertically_adjacent)


def merge_boxes(box1: Dict[str, float], box2: Dict[str, float]) -> Dict[str, float]:
    """
    Merge two adjacent boxes into a larger box.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        Merged bounding box
    """
    x1 = min(box1['x'], box2['x'])
    y1 = min(box1['y'], box2['y'])
    x2 = max(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = max(box1['y'] + box1['height'], box2['y'] + box2['height'])

    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1
    }


def is_within(box1: Dict[str, float], box2: Dict[str, float]) -> bool:
    """
    Check if box1 is within box2.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        True if box1 is completely within box2
    """
    return (
        box1['x'] >= box2['x'] and
        box1['y'] >= box2['y'] and
        box1['x'] + box1['width'] <= box2['x'] + box2['width'] and
        box1['y'] + box1['height'] <= box2['y'] + box2['height']
    )


def extract_visual_components(
    url: str,
    save_path: Optional[str] = None,
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Extract visual components from an HTML file using Playwright.

    Args:
        url: Path to HTML file or URL
        save_path: Optional path to save annotated screenshot
        viewport_width: Optional viewport width in pixels (default: browser default)
        viewport_height: Optional viewport height in pixels (default: browser default, or full page height)

    Returns:
        Dictionary mapping component types to lists of component data
    """
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    screenshot_image = None
    element_data = {}

    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch()

            # Create page with optional viewport
            viewport_dict = None
            if viewport_width is not None:
                # Use provided height or a large default to allow full page rendering
                # The actual page height will be determined by the content
                height = viewport_height if viewport_height is not None else 2000  # Large default for full page rendering
                viewport_dict = {
                    "width": viewport_width,
                    "height": height
                }

            if viewport_dict:
                page = browser.new_page(viewport=viewport_dict)
            else:
                page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Wait a bit for rendering
            page.wait_for_timeout(500)

            # Get actual rendered dimensions (will be full page height)
            total_width = page.evaluate("() => document.documentElement.scrollWidth")
            total_height = page.evaluate("() => document.documentElement.scrollHeight")

            # Define selectors for specific types of elements
            selectors = {
                'video': 'video',
                'image': 'img',
                'text_block': 'p, span, a, strong, h1, h2, h3, h4, h5, h6, li, th, td, label, code, pre, div',
                'form_table': 'form, table, div.form',
                'button': 'button, input[type="button"], input[type="submit"], [role="button"]',
                'nav_bar': 'nav, [role="navigation"], .navbar, [class~="nav"], [class~="navigation"], [class~="menu"], [class~="navbar"], [id="menu"], [id="nav"], [id="navigation"], [id="navbar"]',
                'divider': 'hr, [class*="separator"], [class*="divider"], [id="separator"], [id="divider"], [role="separator"]',
            }

            for key, value in selectors.items():
                element_data[key] = []
                elements = page.query_selector_all(value)

                for element in elements:
                    if not element.is_visible():
                        continue

                    box_data = element.bounding_box()
                    if box_data and box_data['width'] > 0 and box_data['height'] > 0:
                        text_content = None

                        if key == 'text_block':
                            is_direct_text = element.evaluate("""
                                (el) => Array.from(el.childNodes).some(node =>
                                    node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '')
                            """)
                            tag_name = element.evaluate("el => el.tagName.toLowerCase()")

                            if tag_name == 'div' and not is_direct_text:
                                continue

                            text_content = element.text_content().strip()
                            if not text_content:
                                continue

                        element_data[key].append({
                            'type': key,
                            'box': {
                                'x': box_data['x'],
                                'y': box_data['y'],
                                'width': box_data['width'],
                                'height': box_data['height']
                            },
                            'text_content': text_content,
                        })

            # Process adjacent text blocks
            text_blocks = element_data['text_block']
            text_blocks.sort(key=lambda block: (block['box']['y'], block['box']['x']))

            merged_text_blocks = []
            while text_blocks:
                current = text_blocks.pop(0)
                index = 0
                add = True

                while index < len(text_blocks):
                    if is_within(text_blocks[index]['box'], current['box']):
                        # Skip nested text blocks
                        del text_blocks[index]
                        continue
                    elif is_within(current['box'], text_blocks[index]['box']):
                        # Current box is nested, skip it
                        add = False
                        break

                    if boxes_adjacent(current['box'], text_blocks[index]['box']):
                        if current['box']['x'] < text_blocks[index]['box']['x'] or current['box']['y'] < text_blocks[index]['box']['y']:
                            current['text_content'] += " " + text_blocks[index]['text_content']
                        else:
                            current['text_content'] = text_blocks[index]['text_content'] + " " + current['text_content']
                        current['box'] = merge_boxes(current['box'], text_blocks[index]['box'])
                        del text_blocks[index]
                    else:
                        index += 1

                if add:
                    merged_text_blocks.append(current)

            element_data['text_block'] = merged_text_blocks

            # Take full page screenshot
            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)
            image_buffer = io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)

            # Draw bounding boxes and labels
            if save_path:
                draw = ImageDraw.Draw(screenshot_image)
                font = ImageFont.load_default()

                for key in element_data:
                    for item in element_data[key]:
                        x = item['box']['x']
                        y = item['box']['y']
                        width = item['box']['width']
                        height = item['box']['height']
                        draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)
                        draw.text((x, y), item['type'], fill="red", font=font)

                screenshot_image.save(save_path)

            # Normalize positions and sizes to relative values
            for key in element_data:
                for item in element_data[key]:
                    box_data = item['box']
                    item['box'] = {
                        'x': box_data['x'] / total_width,
                        'y': box_data['y'] / total_height,
                        'width': box_data['width'] / total_width,
                        'height': box_data['height'] / total_height
                    }

            browser.close()

    except Exception as e:
        print(f"Failed to extract components due to: {e}. Returning empty data.")
        print(traceback.format_exc())
        if save_path:
            screenshot_image = Image.new('RGB', (1280, 960), color='white')
            screenshot_image.save(save_path)

    return element_data


def bounding_box_to_polygon(bbox: Dict[str, float]):
    """Convert bounding box to shapely polygon."""
    return box(bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])


def compute_list_iou_shapely(listA: List[Dict], listB: List[Dict]) -> Tuple[float, float]:
    """
    Compute IoU between two lists of components using shapely.

    Args:
        listA: List of components from first layout
        listB: List of components from second layout

    Returns:
        Tuple of (iou_score, union_area)
    """
    if not listA and not listB:
        return 0.0, 0.0
    elif not listA:
        polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]
        unionB = unary_union(polygonsB)
        return 0.0, unionB.area
    elif not listB:
        polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
        unionA = unary_union(polygonsA)
        return 0.0, unionA.area

    # Convert bounding boxes to shapely polygons
    polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
    polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]

    # Create union of all polygons
    unionA = unary_union(polygonsA)
    unionB = unary_union(polygonsB)

    # Compute intersection and union
    intersection_area = unionA.intersection(unionB).area
    union_area = unionA.union(unionB).area

    total_iou = intersection_area / union_area if union_area > 0 else 0

    return total_iou, union_area


def compute_weighted_iou_shapely(
    elementsA: Dict[str, List[Dict]],
    elementsB: Dict[str, List[Dict]]
) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    """
    Compute weighted IoU across all component types.

    Args:
        elementsA: Components from first layout
        elementsB: Components from second layout (reference)

    Returns:
        Tuple of (weighted_iou, multi_score_dict)
        multi_score_dict maps component type to (score, weight)
    """
    areas = {}
    ious = {}
    all_keys = set(elementsA.keys()).union(set(elementsB.keys()))

    for key in all_keys:
        if key not in elementsA:
            elementsA[key] = []
        if key not in elementsB:
            elementsB[key] = []

        ious[key], areas[key] = compute_list_iou_shapely(elementsA[key], elementsB[key])

    total_area = sum(areas[key] for key in all_keys)
    weighted_iou = sum(areas[key] * ious[key] for key in all_keys) / total_area if total_area > 0 else 0

    multi_score = {}
    for key in all_keys:
        multi_score[key] = (ious[key], areas[key] / total_area if total_area > 0 else 0)

    return weighted_iou, multi_score


def take_and_save_screenshot(url: str, output_file: str = "screenshot.png", overwrite: bool = False):
    """
    Take and save screenshot of HTML page.

    Args:
        url: Path to HTML file or URL
        output_file: Path to save screenshot
        overwrite: Whether to overwrite existing screenshot
    """
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if os.path.exists(output_file) and not overwrite:
        print(f"{output_file} exists!")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=60000)
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)
            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        img = Image.new('RGB', (1280, 960), color='white')
        img.save(output_file)

