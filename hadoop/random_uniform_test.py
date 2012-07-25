try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Cheat Sheet (method/test) <http://docs.python.org/library/unittest.html>
#
# assertEqual(a, b)       a == b   
# assertNotEqual(a, b)    a != b    
# assertTrue(x)     bool(x) is True  
# assertFalse(x)    bool(x) is False  
# assertRaises(exc, fun, *args, **kwds) fun(*args, **kwds) raises exc
# assertAlmostEqual(a, b)  round(a-b, 7) == 0         
# assertNotAlmostEqual(a, b)          round(a-b, 7) != 0
# 
# Python 2.7+ (or using unittest2)
#
# assertIs(a, b)  a is b
# assertIsNot(a, b) a is not b
# assertIsNone(x)   x is None
# assertIsNotNone(x)  x is not None
# assertIn(a, b)      a in b
# assertNotIn(a, b)   a not in b
# assertIsInstance(a, b)    isinstance(a, b)
# assertNotIsInstance(a, b) not isinstance(a, b)
# assertRaisesRegexp(exc, re, fun, *args, **kwds) fun(*args, **kwds) raises exc and the message matches re
# assertGreater(a, b)       a > b
# assertGreaterEqual(a, b)  a >= b
# assertLess(a, b)      a < b
# assertLessEqual(a, b) a <= b
# assertRegexpMatches(s, re) regex.search(s)
# assertNotRegexpMatches(s, re)  not regex.search(s)
# assertItemsEqual(a, b)    sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)      all the key/value pairs in a exist in b
from random_uniform import LeakyPriorityQueue

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        a = LeakyPriorityQueue(5)
        for x in range(1000):
            a.add(x, str(x))
        self.assertEqual(list(a.items()),
                         [(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')])
        
        a = LeakyPriorityQueue(5)
        for x in range(1000)[::-1]:
            a.add(x, str(x))
        self.assertEqual(list(a.items()),
                         [(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')])

        a = LeakyPriorityQueue(5)
        self.assertEqual(list(a.items()),
                         [])
        self.assertEqual(list(a.values()),
                         [])
        a.add(1, 1)
        self.assertEqual(list(a.values()),
                         [1])
        self.assertEqual(list(a.items()),
                         [(1, 1)])

if __name__ == '__main__':
    unittest.main()
