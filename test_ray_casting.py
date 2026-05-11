#!/usr/bin/env python3
"""Test Ray Casting algorithm for polygon point-in-polygon detection."""

from services import point_in_polygon

def test_point_in_triangle():
    """Test with a simple triangle."""
    triangle = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 0.5, "y": 1.0},
    ]
    
    # Point inside triangle (center)
    assert point_in_polygon(0.5, 0.5, triangle), "Center of triangle should be inside"
    
    # Point outside triangle
    assert not point_in_polygon(0.1, 0.9, triangle), "Top-left should be outside"
    print("✅ Triangle tests passed")


def test_point_in_rectangle():
    """Test with a rectangle."""
    rectangle = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.1, "y": 0.9},
    ]
    
    # Point inside
    assert point_in_polygon(0.5, 0.5, rectangle), "Center should be inside"
    assert point_in_polygon(0.2, 0.2, rectangle), "Offset center should be inside"
    
    # Point outside
    assert not point_in_polygon(0.05, 0.5, rectangle), "Left of rectangle should be outside"
    assert not point_in_polygon(0.95, 0.5, rectangle), "Right of rectangle should be outside"
    print("✅ Rectangle tests passed")


def test_edge_cases():
    """Test edge cases."""
    zone = [
        {"x": 0.0, "y": 0.0},
        {"x": 0.4, "y": 0.0},
        {"x": 0.4, "y": 1.0},
        {"x": 0.0, "y": 1.0},
    ]
    
    # Left 40% of frame (original ZONE_INTERDITE)
    assert point_in_polygon(0.2, 0.5, zone), "0.2, 0.5 should be inside (40% left)"
    assert point_in_polygon(0.3, 0.9, zone), "0.3, 0.9 should be inside"
    assert point_in_polygon(0.0, 0.0, zone), "0.0, 0.0 should be inside (corner)"
    
    # Outside 40% zone
    assert not point_in_polygon(0.5, 0.5, zone), "0.5, 0.5 should be outside (right 60%)"
    assert not point_in_polygon(0.9, 0.5, zone), "0.9, 0.5 should be outside (far right)"
    print("✅ Edge case tests passed (40% left zone)")


def test_tuple_points():
    """Test with tuple coordinates instead of dict."""
    triangle = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 1.0),
    ]
    
    assert point_in_polygon(0.5, 0.5, triangle), "Center should be inside triangle (tuples)"
    print("✅ Tuple coordinates test passed")


if __name__ == "__main__":
    test_point_in_triangle()
    test_point_in_rectangle()
    test_edge_cases()
    test_tuple_points()
    print("\n🎉 All Ray Casting tests passed!")
