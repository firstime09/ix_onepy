import datetime, glob, os, math, gdal
from datetime import datetime, date
from osgeo import gdal
import numpy as np

class allFunc():
    # def readData(path):
    #     path = 'FORESTS2020/M_8_1'
    #     path_f = []
    #     for path in glob.glob('/*.txt'):
    #         root, folder, file = next(os.walk(path))
    #     return folder

    def readFolder(path):
        """Read location directory File"""
        path_f = []
        root, folder, file = next(os.walk(path))
        for i in folder:
            Rfolder = os.path.join(root, i)
            path_f.append(Rfolder)
        return path_f

    def build_data(f):
        output = {}
        for line in f.readlines():
            if "=" in line:
                l = line.split("=")
                output[l[0].strip()] = l[1].strip()
        return output

    def year_date(data):
        year_file = data['DATE_ACQUIRED']
        date_file = data['SCENE_CENTER_TIME']
        date_file2 = date_file [1:16]
        all = year_file+" "+date_file2
        parsing = datetime.strptime(all, '%Y-%m-%d %H:%M:%S.%f')
        return parsing

    def zone(s):
        WIT = [102061, 102062, 102063, 102064, 102065, 102066, 109058, 109059, 109060, 109061, 109062, 109063, 109064,
               109065, 109066, 109067, 100062, 100063, 100064, 100065, 100066, 107059, 107060, 107061, 107062, 107063,
               107064, 107065, 107066, 105060, 105061, 105062, 105063, 105064, 105065, 103061, 103062, 103063, 103064,
               110057, 110058, 110059, 110060, 110061, 110062, 110063, 110064, 110065, 110066, 110067, 101061, 101062,
               101063, 101064, 101065, 101066, 108060, 108061, 108062, 108063, 108064, 108065, 108066, 106060, 106061,
               106062, 106063, 106064, 106065, 106066, 104060, 104061, 104062, 104063, 104064, 104065, 111060, 111061,
               111062]
        WIB = [118060, 118061, 118062, 118063, 118064, 118065, 118066, 125059, 125060, 125061, 125062, 125063, 125064,
               123057, 123058, 123059, 123060, 123061, 123062, 123063, 123064, 123065, 130056, 130057, 130058, 130059,
               121060, 121061, 121062, 121063, 121064, 121065, 128057, 128058, 128059, 128060, 128061, 119060, 119061,
               119062, 119063, 119064, 119065, 119066, 126059, 126060, 126061, 126062, 126063, 124058, 124059, 124060,
               124061, 124062, 124063, 124064, 124065, 131056, 131057, 131058, 122058, 122059, 122060, 122061, 122062,
               122063, 122064, 122065, 129057, 129058, 129059, 129060, 120060, 120061, 120062, 120063, 120064, 120065,
               120066, 127058, 127059, 127060, 127061, 127062]
        if s in WIB:
            k = int(7)
        elif s in WIT:
            k = int(9)
        else:
            k = int(8)
        return k

    def hour(dt, z):
        h = dt.hour + z
        return h

    def second(dt):
        s = float(dt.microsecond) / 1000000 + dt.second
        return s

    def leap(dt):
        if (dt.year % 4) == 0:
            if (dt.year % 100) == 0:
                if (dt.year % 400) == 0:
                    a = int(366)
                else:
                    a = int(365)
            else:
                a = int(366)
        else:
            a = int(365)
        return a

    def cos(x):
        cos = np.cos(np.deg2rad(x))
        return cos

    def sin(x):
        sin = np.sin(np.deg2rad(x))
        return sin

    def day(dt):
        day_date = date(dt.year, dt.month, dt.day)
        sum_of_day = int(day_date.strftime('%j'))
        return sum_of_day

    def hitGama(leap, day, hour, dt, second):
        gamma = ((2 * math.pi) / leap) * ((day - 1) + (((hour + dt.minute / 60 + second / 3600) - 12) / 24))
        return gamma

    def sundecAngle(gamma, sin, cos): # ----- for to know about sun position
        decl = (0.006918 - 0.399912 * ((cos)*(gamma)) + 0.070257 * ((sin)*(gamma)) - 0.006758 * ((cos)*(2 * gamma)) \
               + 0.000907 * ((sin)*(2 * gamma)) - 0.002697 * ((cos)*(3 * gamma)) + 0.00148 * ((sin)*(3 * gamma)))  # radians
        decl_deg = (360 / (2 * math.pi)) * decl
        return decl_deg

    def pixel2coord(x, y, band):
        xoff, a, b, yoff, d, e = band.GetGeoTransform()
        xp = a*x + b*y + xoff
        yp = d*x + e*y + yoff
        return(xp, yp)

    def eqoftime(gamma, sin, cos):
        timeeq = (229.18 * (0.000075 + 0.001868 * ((cos)*(gamma)) - 0.032077 * ((sin)*(gamma)) \
                - 0.014615 * ((cos)*(2*gamma))))
        return timeeq

