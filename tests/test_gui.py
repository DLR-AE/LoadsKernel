import logging

# The imports of the gui modules are moved into the test such that they are part of the test.


class TestLoadsCompare():

    def test_gui(self):
        logging.info('Testing Loads Compare')
        from loadscompare import compare
        c = compare.Compare()
        c.test()


class TestModelViewer():

    def test_gui(self):
        logging.info('Testing Model Viewer')
        from modelviewer import view as modelviewer
        m = modelviewer.Modelviewer()
        m.test()


class TestResponseViewer():

    def test_gui(self):
        logging.info('Testing Response Viewer')
        from responseviewer import view as responseview
        m = responseview.ResponseViewer()
        m.test()
