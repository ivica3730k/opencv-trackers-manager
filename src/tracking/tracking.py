import cv2

_trackedObjects = []
_sampleObtainingAreas = []
_sampleObtainingAreaId: int = 0
_trackedObjectId: int = 0


class Point:
    x: int = 0
    y: int = 0

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False


class Rectangle:
    point1 = Point(0, 0)
    point2 = Point(0, 0)
    width: int = 0
    height: int = 0

    def __init__(self, p1: Point, p2: Point = None, width: int = None, height: int = None):
        self.point1 = p1
        if p2 is None:
            self.width = width
            self.height = height
            self.point2 = Point(self.point1.x + self.width, self.point1.y + self.height)
        if width is None and height is None:
            self.point2 = p2
            self.width = self.point2.x - self.point1.x
            self.height = self.point2.y - self.point1.y

    def get_midpoint(self):
        return Point(int((self.point1.x + self.point2.x) / 2), int((self.point1.y + self.point2.y) / 2))

    def get_xywh(self):
        return self.point1.x, self.point1.y, self.width, self.height


class TrackedObject:
    _area: Rectangle
    _tracker: cv2.Tracker = None
    _original_mid: Point
    _id: int = -1

    def get_start_point(self):
        """
        Get the start point of the tracked area
        """
        return self._area.point1

    def get_end_point(self):
        """
        Get the end point of the tracked area
        """
        return self._area.point2

    def get_midpoint(self):
        """
        Get the get_midpoint of the tracked area
        :return: Tuple defining the _mid-point of the tracked area in (x,y) format.
        """
        return self._area.get_midpoint()

    def get_width(self):
        """
        Get the _width of the sample area

        :return: Width value
        """
        return self._area.width

    def get_height(self):
        """
        Get the _height of the sample area

        :return: Height value
        """
        return self._area.height

    def get_id(self):
        return self._id

    def __init__(self, frame, area: Rectangle, new_id=None):

        self._area = area
        # self._tracker = cv2.TrackerGOTURN_create()
        self._tracker = cv2.TrackerKCF_create()
        self._tracker.init(frame, (self._area.get_xywh()))
        self._original_mid = self._area.get_midpoint()
        global _trackedObjectId
        if new_id:
            if new_id > _trackedObjectId:
                _trackedObjectId = new_id + 1
            self._id = new_id
        else:
            self._id = _trackedObjectId
            _trackedObjectId += 1
        global _trackedObjects
        _trackedObjects.append(self)
        pass

    def update(self, frame):
        """
        Update the tracker with the new frame

        :param frame: Frame used to update the tracker
        :return: Success status of the tracker update operation
        """
        ok, bbox = self._tracker.update(frame)
        # Draw bounding box
        if ok:
            # Tracking success
            self._area = Rectangle(Point(int(bbox[0]), int(bbox[1])), width=bbox[2], height=bbox[3])
            return True
        return False

    def get_xywh(self):
        return self._area.get_xywh()


def get_tracked_objects():
    """
    Get the list of all tracked objects by the tracking module

    :return: List of tracked objects
    """
    return _trackedObjects


def clear_tracked_objects_list():
    """
    Clear the list off all tracked objects
    """
    global _trackedObjects
    _trackedObjects.clear()


def remove_object_from_tracked_objects_list(tracker):
    """
    Remove the tracked object from the list
    :param tracker: Tracked object to remove
    """
    global _trackedObjects
    _trackedObjects.remove(tracker)


def get_sample_obtaining_areas():
    """
    Get the list of sample obtaining areas
    :return: List of sample obtaining areas
    """
    return _sampleObtainingAreas


def _intersection(a, b):
    """
    Returns intersect point between two provided rectangles.

    Use (x,y,w,h) tuple format.
    :param a: First rectangle
    :param b: Second rectangle
    :return: Intersection point of two rectangles
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()  # or (0,0,0,0) ?
    return x, y, w, h


def is_object_tracked(x=None, y=None, w=None, h=None, area: Rectangle = None):
    """
    Function that check if the object is currently on the tracking list.

    Returns True if the provided object intersects with any of the objects
    in the trackedObjects list.
    :param x: X start point of rectangle
    :param y: Y start point of rectangle
    :param w: Width of rectangle
    :param h: Height of rectangle
    :param area: Area of interest
    :return: Boolean value, true if provided polygon is intersection with any polygon on tracking list
    """

    if area:
        a = area.get_xywh()
    else:
        a = (x, y, w, h)
    for i in _trackedObjects:
        i: TrackedObject
        b = i.get_xywh()
        intersection = _intersection(a, b)
        if intersection:
            return True
    return False


def clear_all():
    """
    Clears everything
    """
    global _trackedObjectId
    global _trackedObjects
    _trackedObjectId = 0
    _trackedObjects = []
