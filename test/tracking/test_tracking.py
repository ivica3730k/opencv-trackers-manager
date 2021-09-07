import unittest

import numpy as np

from src.tracking.tracking import *

test_image = np.zeros((640, 480, 3), np.uint8)


class TestPointAndRectangleRepresentations(unittest.TestCase):
    def test_initialize_point(self):
        p = Point(100, 200)
        self.assertEqual(p.x, 100)
        self.assertEqual(p.get_x(), 100)
        self.assertEqual(p.y, 200)
        self.assertEqual(p.get_y(), 200)

    def test_compare_points(self):
        p1 = Point(100, 100)
        p2 = Point(100, 100)
        p3 = Point(150, 150)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p2, p3)

    def test_initialize_rectangle_from_two_points(self):
        p1 = Point(100, 100)
        p2 = Point(200, 200)
        r1 = Rectangle(p1, p2)
        self.assertEqual(r1.point1, p1)
        self.assertEqual(r1.point2, p2)
        self.assertEqual(r1.get_midpoint(), Point(150, 150))
        self.assertEqual(r1.width, 100)
        self.assertEqual(r1.height, 100)
        self.assertEqual(r1.get_xywh(), (100, 100, 100, 100))

    def test_initialize_rectangle_from_point_and_dimensions(self):
        p1 = Point(100, 100)
        width = 100
        height = 100
        p2 = Point(200, 200)
        r1 = Rectangle(p1, width=width, height=height)
        self.assertEqual(r1.point1, p1)
        self.assertEqual(r1.point2, p2)
        self.assertEqual(r1.get_midpoint(), Point(150, 150))
        self.assertEqual(r1.width, 100)
        self.assertEqual(r1.height, 100)
        self.assertEqual(r1.get_xywh(), (100, 100, 100, 100))


class TestTrackingModule(unittest.TestCase):

    def test_clear_all(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(300, 300)))
        obj2 = TrackedObject(test_image, Rectangle(Point(300, 300), Point(500, 500)))
        assert len(get_tracked_objects()) == 2
        assert get_tracked_objects()[0] == obj1
        assert get_tracked_objects()[1] == obj2
        clear_all()
        assert len(get_tracked_objects()) == 0
        with self.assertRaises(IndexError):
            _ = get_tracked_objects()[0] == obj1
            _ = get_tracked_objects()[0] == obj2

    def test_get_tracked_objects(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(300, 300)))
        obj2 = TrackedObject(test_image, Rectangle(Point(300, 300), Point(500, 500)))
        assert len(get_tracked_objects()) == 2
        assert get_tracked_objects()[0] == obj1
        assert get_tracked_objects()[1] == obj2
        clear_all()

    def test_clear_tracked_objects_list(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(300, 300)))
        obj2 = TrackedObject(test_image, Rectangle(Point(300, 300), Point(500, 500)))
        assert len(get_tracked_objects()) == 2
        clear_tracked_objects_list()
        assert len(get_tracked_objects()) == 0
        clear_all()

    def test_remove_object_from_tracked_objects_list(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(300, 300)))
        obj2 = TrackedObject(test_image, Rectangle(Point(300, 300), Point(500, 500)))
        remove_object_from_tracked_objects_list(obj2)
        assert len(get_tracked_objects()) == 1
        assert get_tracked_objects()[0] == obj1
        clear_all()

    def test_is_object_tracked(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(300, 300)))
        self.assertTrue(is_object_tracked(100, 100, 200, 200))  # Assert bounding box full intersection
        self.assertTrue(is_object_tracked(50, 50, 200, 200))  # Assert bounding box intersection right top
        self.assertTrue(is_object_tracked(150, 150, 200, 200))  # Assert bounding box intersection left bottom
        self.assertFalse((is_object_tracked(301, 301, 200, 200)))  # Assert no intersection
        self.assertTrue((is_object_tracked(300, 300, 200, 200)))  # Assert single border (line) intersection

        self.assertTrue(is_object_tracked(
            area=Rectangle(Point(100, 100), width=200, height=200)))  # Assert bounding box full intersection
        self.assertTrue(is_object_tracked(
            area=Rectangle(Point(50, 50), width=200, height=200)))  # Assert bounding box intersection right top
        self.assertTrue(is_object_tracked(
            area=Rectangle(Point(150, 150), width=200, height=200)))
        self.assertFalse(is_object_tracked(
            area=Rectangle(Point(301, 301), width=200, height=200)))  # Assert no intersection
        self.assertTrue(is_object_tracked(
            area=Rectangle(Point(300, 300), width=200, height=200)))  # Assert single border (line) intersection
        clear_all()


class TestTrackedObject(unittest.TestCase):
    def test_get_starting_point(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(0, 0), Point(100, 100)))
        self.assertEqual(obj1.get_start_point(), Point(0, 0))
        clear_all()

    def test_get_end_point(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(0, 0), Point(100, 100)))
        self.assertEqual(obj1.get_end_point(), Point(100, 100))
        clear_all()

    def test_get_midpoint(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(0, 0), Point(100, 100)))
        self.assertEqual(obj1.get_midpoint(), Point(50, 50))
        clear_all()

    def test_get_id(self):
        obj1 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(200, 200)))
        obj2 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(200, 200)))
        obj3 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(200, 200)), new_id=10)
        obj4 = TrackedObject(test_image, Rectangle(Point(100, 100), Point(200, 200)))
        self.assertEqual(get_tracked_objects()[0].get_id(), 0)
        self.assertEqual(get_tracked_objects()[1].get_id(), 1)
        self.assertEqual(get_tracked_objects()[2].get_id(), 10)
        self.assertEqual(get_tracked_objects()[3].get_id(), 11)
        clear_all()
