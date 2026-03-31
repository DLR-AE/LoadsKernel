import logging

try:
    from loadscompare import compare
    from modelviewer import view as modelviewer
    from responseviewer import view as responseview
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
        m = modelviewer.Modelviewer()
        m.test()


class TestResponseViewer():

    def test_gui(self):
        logging.info('Testing Response Viewer')
        m = responseview.ResponseViewer()
        m.test()
