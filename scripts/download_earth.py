

import numpy as np
from cookielib import CookieJar
import urllib2
import cartopy.crs as ccrs
from cartopy.io.srtm import srtm_composite
import pickle
# from mayavi import mlab

def get_earth():
        
        # The user credentials that will be used to authenticate access to the data
        username = "arnevoss"
        password = "Caolila12"
        
        # Create a password manager to deal with the 401 reponse that is returned from
        # Earthdata Login
        password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
        
        # Create a cookie jar for storing cookies. This is used to store and return
        # the session cookie given to use by the data server (otherwise it will just
        # keep sending us back to Earthdata Login to authenticate).  Ideally, we
        # should use a file based cookie jar to preserve cookies between runs. This
        # will make it much more efficient.
        cookie_jar = CookieJar()
        # Install all the handlers.
        
        opener = urllib2.build_opener(
            urllib2.HTTPBasicAuthHandler(password_manager),
            #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
            #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
            urllib2.HTTPCookieProcessor(cookie_jar))
        urllib2.install_opener(opener)
        
        print('start downloading earth data...')
        elev, crs, extent = srtm_composite(10, 51, 1, 1)
        print('downloading finished.')
        
        extensions_x = (extent[1]-extent[0])*1852.0*60.0/2.0
        extensions_y = (extent[3]-extent[2])*1852.0*60.0/2.0
        res_y = extensions_y*2.0 / elev.shape[0]
        res_x = extensions_x*2.0 / elev.shape[1]
        x = np.arange(-extensions_x, extensions_x, res_x)
        y = np.arange(-extensions_y, extensions_y, res_y)
        return x, y, elev




if __name__ == "__main__":
    x, y, elev = get_earth()
    with open('earth.pickle', 'w') as f:# open response
        pickle.dump((x,y,elev), f, pickle.HIGHEST_PROTOCOL)
    print('Done.')

# surf = mlab.surf(np.arange(-extensions_x, extensions_x, res_x), np.arange(-extensions_y, extensions_y, res_y), elev, colormap='gist_earth')
# mlab.orientation_axes()
# mlab.show()
# surf.module_manager.scalar_lut_manager.reverse_lut = True


