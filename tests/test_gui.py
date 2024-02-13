import logging

try:
    from loadscompare import compare
    from modelviewer import view
except ImportError:
    pass


class TestLoadsCompare():

    def test_gui(self):
        logging.info('Testing Loads Compare')
        c = compare.Compare()
        c.test()


class TestModelViewer():

    def test_gui(self):
        logging.info('Testing Model Viewer')
        m = view.Modelviewer()
        m.test()
