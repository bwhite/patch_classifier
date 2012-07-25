try:
    import unittest2 as unittest
except ImportError:
    import unittest

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
